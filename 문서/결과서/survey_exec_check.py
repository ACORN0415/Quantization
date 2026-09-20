"""G1 실행 확인 — 저장 연산자의 Q2-a..d′ 를 코드 구조가 아니라 측정으로 판정.

명세가 분리하라고 한 다섯 가지 중 네 가지를 각각 잰다:
  재양자화 호출 / 코드 변화 / 복원값 변화 ‖x̂₂−x̂₁‖ / 원본 대비 오차 증감 ‖x̂−x‖
(품질 영향은 이 조사에서 판정하지 않는다 — L1에서 잰다.)

세 저장 체제:
  R1 같은 격자 · 입력 = 복원값      (정렬 ON, fused 세그먼트)
  R2 8토큰 밀린 격자 · 입력 = 복원값 (Self-Forcing UCSD 패치의 rolling)
  R3 같은 격자 · 입력 = 원본 BF16    (결정성 확인용. 이것만으로는 아무것도 안 나온다)
  R4 밀린 격자 · 입력 = 원본 BF16    (LongLive: master가 굴러가며 격자가 바뀐다)

  **R3와 R4를 반드시 나눈다.** "BF16 master가 이전 오차의 재입력을 막는다"와
  "격자가 바뀌어도 출력이 안 바뀐다"는 **다른 주장**이다. R3는 앞의 것만 보이고,
  뒤의 것은 R4라야 답한다.

입력은 **통제 입력**이다(모델 활성이 아니다). 연산자의 성질은 이것으로 판정되지만
크기는 실제 분포에 따라 달라진다 — 그 점을 결과에 함께 적는다.
"""
import sys, os
import torch
sys.path.insert(0, os.path.expanduser("~/gpu/kv-quant-longhorizon"))
from kv_quant.factory import create_quantizer

torch.manual_seed(0)
B, T, H, D, SHIFT = 1, 4680, 12, 128, 8


def norm(a, b):
    return (a.float() - b.float()).norm().item()


def run(qname, bits, dist):
    q = create_quantizer(qname, bits=bits, block_size=16,
                         key_bits=bits, value_bits=bits, name=f"{qname}{bits}")
    if dist == "gauss":
        x = torch.randn(B, T, H, D, dtype=torch.bfloat16)
    else:                                   # outlier channels, as in real keys
        x = torch.randn(B, T, H, D, dtype=torch.bfloat16)
        x[:, :, :, ::16] *= 30.0
    Q = lambda t: q.quantize_kv(t, t, meta={"tensor_dtype": torch.bfloat16})
    Dq = lambda s: q.dequantize_kv(s, meta={"tensor_dtype": torch.bfloat16})[0]

    s1 = Q(x); x1 = Dq(s1)
    e1 = norm(x1, x)
    out = {}
    # R1 같은 격자, 입력 = 복원값
    s2 = Q(x1); x2 = Dq(s2)
    out["R1"] = (torch.equal(s1["k"]["q"], s2["k"]["q"]), norm(x2, x1), e1, norm(x2, x))
    # R2 밀린 격자, 입력 = 복원값 (앞을 SHIFT만큼 잘라 격자를 어긋냄)
    xs = x1[:, SHIFT:]
    s3 = Q(xs); x3 = Dq(s3)
    out["R2"] = (torch.equal(s1["k"]["q"][:, :, :, :, :][:, : s3["k"]["q"].shape[1]],
                             s3["k"]["q"]) if s1["k"]["q"].shape[1] == s3["k"]["q"].shape[1] else False,
                 norm(x3, xs), norm(x1[:, SHIFT:], x[:, SHIFT:]), norm(x3, x[:, SHIFT:]))
    # R3 입력 = 원본 BF16, 같은 격자 — 결정성만 확인한다
    s4 = Q(x); x4 = Dq(s4)
    out["R3"] = (torch.equal(s1["k"]["q"], s4["k"]["q"]), norm(x4, x1), e1, norm(x4, x))
    # R4 입력 = 원본 BF16, 밀린 격자 — LongLive처럼 master가 굴러가는 경우.
    # 누적은 없지만 격자가 바뀌므로 복원값과 오차가 어떻게 되는지는 별개 질문이다.
    xo = x[:, SHIFT:]                       # 원본을 SHIFT만큼 잘라 격자를 어긋냄
    s5 = Q(xo); x5 = Dq(s5)
    out["R4"] = (False,                      # 토큰 수가 달라 코드 비교는 성립 안 함
                 norm(x5, x1[:, SHIFT:]),    # 밀린 격자 복원값 vs 원래 격자 복원값
                 norm(x1[:, SHIFT:], xo),    # 원래 격자의 원본 대비 오차
                 norm(x5, xo))               # 밀린 격자의 원본 대비 오차
    return out


print("Q2-d(복원값 변화) 와 Q2-d′(원본 대비 오차) 는 다른 양이다 — 따로 잰다.\n")
hdr = f"{'양자화기':10s} {'분포':8s} {'체제':4s} {'코드 동일':>9s} {'‖x̂₂−x̂₁‖':>10s} {'‖x̂₁−x‖':>10s} {'‖x̂₂−x‖':>10s} {'오차 증감':>9s}"
print(hdr); print("-" * len(hdr))
for qname, bits in (("RTN", 4), ("RTN", 2), ("KIVI", 4), ("KIVI", 2)):
    for dist in ("gauss", "outlier"):
        r = run(qname, bits, dist)
        for reg in ("R1", "R2", "R3", "R4"):
            same, dx, e_before, e_after = r[reg]
            delta = e_after - e_before
            arrow = "같음" if abs(delta) < 1e-6 else ("증가" if delta > 0 else "감소")
            print(f"{qname+str(bits):10s} {dist:8s} {reg:4s} {str(same):>9s} "
                  f"{dx:10.3f} {e_before:10.3f} {e_after:10.3f} {arrow:>9s}")
    print()
