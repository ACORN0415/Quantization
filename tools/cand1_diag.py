# -*- coding: utf-8 -*-
"""후보 1 진단 (a) — r2.1 §3.4·§3.5. 기존 캡처(consec·consec2)에서 GPU 로 재구성.

무엇을 재는가
  1. 잔차 자체: E = K − K̂0 (RTN INT2, chunk-local 격자 group16) 와 PSAC 후 E_rem 의
     head 별 평균·채널 분산·공분산 집중도, 그리고 균등 모형 예측 Δ²/12 대비 실측 분산비.
  2. R0~R4 사다리: chunk 21 step 1~3 query 에서 attention 출력의 BF16 참조 대비 상대오차,
     캐시 attention mass, head 별 보정량 b 분포(outlier head 추적).
  3. 바이트: K 쪽 head·chunk 당 INT2 / INT2+PSAC(r) / R4 통계, K4V2 대비.

팔 (K 만 바꾼다. V 는 모든 팔에서 RTN INT2 복원으로 고정. self chunk 21 은 BF16.)
  R0        K̂0, 보정 없음
  R1        K̂0 + TUM Taylor λ1 with Δ=2·scale  — **주의**: 이것은 Quantization/tools/tum_correction.py(감사 전 사본)의 정의다.
            Self-Forcing 트리의 실제 tum_correction.py 와 fused 커널은 Δ=scale 을 쓴다(감사에서 "Δ=2·scale 버그" 로 수정,
            해당 docstring: "lam=4 reproduces the pre-audit Delta=2*scale bug"). 따라서 R1 = 프로젝트의 **λ4(버그 재현)** 이고,
            **R1_l0.25 = 프로젝트의 실제 λ1** (P3 fused λ1 과 같은 보정량). 실측 잔차 분산은 Δ=scale 기준 Δ²/12 의 0.94~1.01 배.
  R1p       K̂0 + 같은 식, Δ² 를 head 단위로 풀링(블록 평균)  — 입도 효과 분리용 대조
  R1_l<λ>   K̂0 + TUM λ∈{0.5,0.25} (Δ=2·scale 기준). λ0.25 ≡ Δ=scale·λ1 = 실제 기준선.
  R4d0/R4f0 K̂0 + E0 통계 보정 (PSAC 없음) — PSAC 과 보정식의 기여 분리
  변형 N    무작위 직교 기저 PSAC — 방향 선택(query 통계)의 몫 분리
  R2_<v>    K̂1 = K̂0 + PSAC(E, U_v, r), 보정 없음.  v ∈ {B, E, D}
  R3_<v>    K̂1 + 제거 전 Δ² 기준 TUM (λ ∈ {1, 0.5, 0.25} — 스칼라 대조)
  R4d_<v>   K̂1 + 잔여 오차 대각 모형:  b = (1/2d)·Σ_c q_c² Var_rem,c   (head 당 128 분산)
  R4f_<v>   K̂1 + 잔여 오차 평균+공분산 모형: b = −q·μ_rem/√d + qᵀΣ_rem q/(2d)   (r2.1 2차 리뷰 후 부호 수정)
  R4m_<v>   K̂1 + 평균 + 대각 분산 (평균항의 몫 분리)
  PSAC_SCOPE=chunk20: chunk 20 만 팔 처리, 15~19 는 모든 팔에서 INT2+Δ=scale λ1 고정 — "저장 1회 → 다음 chunk 읽기" 온라인 대응.
  PSAC_SCOPE=all(기본): 과거 6 chunk 전부 chunk 20 query 기저로 처리 — chunk 15~19 에는 미래 정보. 오프라인 대응 진단.
  (검산: Var_c = Δ_c²/12 를 R4d 식에 넣으면 R1 과 같은 b 가 나와야 한다 — 실행 시 max 차 출력)

PSAC 방향 U_v (head 별, 상위 r 고유벡터 of query 2차 모멘트 QᵀQ/N):
  B  chunk 20 자기 query(step 0~4)                 배포 가능
  E  B 의 query 를 +3 프레임 RoPE 회전              배포 가능 (S8 §10.2 와 같은 구현)
  D  chunk 21 step 0 query                          oracle
  H  이전 chunk 들의 누적 통계 — 이 캡처에 없음. 새 캡처 후 추가. (r2.1 §3.4 (b))
PSAC 적용: 과거 6 chunk(15~20) 전부에 같은 U_v 적용, chunk 별 자기 E 로 계수 계산. 계수는 열별 int8.

출력: results/session9/cand1_diag/{tag}.json + 진행 로그(stdout). 한 (prompt, layer) 마다 한 줄.
"""
import json, math, os, sys, time
import torch

SF = os.environ.get("SF_TOOLS", "/vfsdata/gpu_data/gpu/Self-Forcing/tools")
sys.path.insert(0, SF)
sys.path.insert(0, os.environ.get("KVQ_ROOT", "/vfsdata/gpu_data/gpu/kv-quant-longhorizon"))
from factor_attention_ref import attn_full, rel                      # noqa: E402
from lr_residual_consec import rope_shift_frames                      # noqa: E402

dev = "cuda"
L, H, D, G = 4680, 12, 128, 16                                        # chunk 토큰, head, head_dim, RTN group(토큰 축)
ROOT = os.environ.get("CONSEC_ROOT", "/vfsdata/gpu_data/attr_dump/consec/points")
PROMPTS = [int(x) for x in os.environ.get("CONSEC_PROMPTS", "0,1,2").split(",")]
LAYERS = [int(x) for x in os.environ.get("CONSEC_LAYERS", "0,3,6,9,12,15,18,22,26,29").split(",")]
RANKS = [int(x) for x in os.environ.get("PSAC_RANKS", "8").split(",")]
VARIANTS = os.environ.get("PSAC_VARIANTS", "B,E,D,N").split(",")
LAMBDAS = [float(x) for x in os.environ.get("TUM_LAMBDAS", "1.0,0.5,0.25").split(",")]
OUT = os.environ.get("CAND1_OUT", "/home/devel/Quantization/results/session9/cand1_diag/consec.json")
OUTLIER = {29: [0, 2, 9]}
FINAL_BF16 = os.environ.get("K_FINAL_BF16", "0") == "1"                 # 과거 K 복원값(K̂0, K̂0+E_keep, K̂4)을 attention 직전 BF16 으로 반올림                                             # B10 의 outlier head

ld = lambda p, c, l, s: torch.load(f"{ROOT}/p{p}_c{c}_l{l}_s{s}.pt", map_location="cpu", weights_only=False)

# ----------------------------------------------------------------------------- RTN INT2, chunk-local grid
def rtn_int2(X, bits=2):
    """X [L,H,D] 한 chunk. kv_quant.utils.quantize_sym(bits, reduce over 16-token block) 과 같은 산술.
    반환 X̂ [L,H,D] (bf16 반올림 후 float), step [L,H,D] = 2·scale (저장소 사본 tum_correction 의 정의; 실제 트리는 scale)."""
    qmax = (1 << (bits - 1)) - 1
    nb = math.ceil(L / G); pad = nb * G - L                              # 4680 = 292.5 블록 → 293 블록, 8 토큰 패딩 (P3 세그먼트와 같음)
    xp = torch.cat([X, X.new_zeros(pad, H, D)], 0) if pad else X
    xb = xp.reshape(nb, G, H, D)
    scale = (xb.abs().amax(dim=1, keepdim=True) / qmax).clamp_min(1e-8).to(torch.float16).float()
    q = torch.round(xb / scale).clamp(-qmax - 1, qmax)
    xh = (q.to(torch.bfloat16) * scale.to(torch.bfloat16)).float()     # 커널과 같은 bf16(q)×bf16(scale)
    step = (2.0 * scale).expand(nb, G, H, D)
    return xh.reshape(-1, H, D)[:L], step.reshape(-1, H, D)[:L]

NB = math.ceil(L / G)
def int2_bytes_k():   # head·chunk 당 K: 코드 2bit + fp16 scale per (block,channel) — S8 §1 의 224,768 과 일치해야 함
    return L * D * 2 // 8 + NB * D * 2
def int4_bytes_k():
    return L * D * 4 // 8 + NB * D * 2
def psac_bytes(r):    # head·chunk 당 추가: int8 계수 L·r + 열 scale r·2 + 기저 r·D·2(chunk 당 보수적으로 계산)
    return L * r + r * 2 + r * D * 2
def r4d_bytes():  return D * 2                 # head·chunk 당 128 분산 fp16
def r4f_bytes():  return D * 2 + D * D * 2     # 평균 128 + 공분산 128×128 fp16 (대각만 쓰면 r4d 와 같음)

# ----------------------------------------------------------------------------- residual statistics
def resid_stats(E, step=None):
    """E [N,H,D]. head 별: 평균 노름/RMS, 채널 분산의 집중도, 공분산 고유값 참여비(유효 rank), 상위 고유값 에너지,
    균등 모형(Δ²/12) 대비 분산비(중앙·p90), 첨도 중앙."""
    out = []
    Ef = E.float()
    for h in range(H):
        e = Ef[:, h]                                        # [N,D]
        mu = e.mean(0); ec = e - mu
        var = ec.pow(2).mean(0)                             # [D]
        cov = (ec.T @ ec) / e.shape[0]                      # [D,D]
        ev = torch.linalg.eigvalsh(cov).clamp_min(0).flip(0)
        pr = (ev.sum() ** 2 / (ev.pow(2).sum().clamp_min(1e-30))).item()
        tot = ev.sum().clamp_min(1e-30)
        rec = {"mean_rel": (mu.norm() / e.pow(2).mean(0).sum().sqrt()).item(),
               "rms": e.pow(2).mean().sqrt().item(),
               "var_top12_frac": (var.topk(12).values.sum() / var.sum().clamp_min(1e-30)).item(),
               "cov_eff_rank": pr,
               "cov_top4_frac": (ev[:4].sum() / tot).item(), "cov_top12_frac": (ev[:12].sum() / tot).item(),
               "kurt_med": ((ec.pow(4).mean(0) / var.pow(2).clamp_min(1e-30)) - 3).median().item()}
        if step is not None:
            pred = (step[:, h].float() ** 2 / 12).mean(0)   # 균등 모형 분산(블록 평균)
            ratio = var / pred.clamp_min(1e-30)
            rec["var_ratio_med"] = ratio.median().item(); rec["var_ratio_p90"] = ratio.quantile(0.9).item()
        out.append(rec)
    return out

# ----------------------------------------------------------------------------- PSAC
def query_basis(Q, r):
    """Q [N,H,D] → U [H,D,r]: head 별 QᵀQ/N 의 상위 r 고유벡터."""
    U = torch.empty(H, D, r, device=Q.device)
    for h in range(H):
        S = Q[:, h].float().T @ Q[:, h].float() / Q.shape[0]
        w, V = torch.linalg.eigh(S)
        U[h] = V[:, -r:].flip(1)
    return U

def psac_keep(E, U, chunk=L):
    """E [N,H,D], U [H,D,r] → E_keep [N,H,D]. 저장 형식대로: 기저 U 는 fp16 반올림, 계수 C=E·U 는 **chunk 별** 열 scale(fp16) 의 int8,
    복원은 bf16 에서 수행(커널과 같은 저장 산술). 바이트 계산(psac_bytes)과 일치하는 형식이다."""
    out = torch.empty_like(E); Uh = U.to(torch.float16).float()
    for h in range(H):
        for j in range(E.shape[0] // chunk):
            sl = slice(j * chunk, (j + 1) * chunk)
            C = E[sl, h].float() @ Uh[h]                                             # [L,r]
            s = (C.abs().amax(0, keepdim=True) / 127).clamp_min(1e-12).to(torch.float16).float()
            Cq = torch.round(C / s).clamp(-127, 127)
            out[sl, h] = ((Cq.to(torch.bfloat16) * s.to(torch.bfloat16)).float() @ Uh[h].T).to(E.dtype)
    return out

# ----------------------------------------------------------------------------- bias terms
def bias_delta(q, step_sq, lam=1.0):
    """TUM Taylor: b[i,t] = λ·(1/24d)·Σ_c q_c² Δ_tc².  q [Lq,D], step_sq [Lk,D] → [Lq,Lk]."""
    return lam * (q.float() ** 2 @ step_sq.float().T) / (24.0 * D)

def bias_diag(q, var):
    """잔여 오차 대각 모형: b = (1/2d)·Σ_c q_c² Var_c.  var [D] (head 당) → [Lq,1] (키에 무관)."""
    return ((q.float() ** 2) @ var.float()).unsqueeze(1) / (2.0 * D)

def bias_full(q, mu, cov, use_mean=True, use_cov=True):
    """평균+공분산. K̂ = K − E 이므로 logit 오차 ε = −q·E/√d, E[e^ε] = exp(−q·μ/√d + qᵀΣq/(2d)).
    빼는 보정량 b = −q·μ/√d + qᵀΣq/(2d).  (r2.1 2차 리뷰: 이전 판은 평균항 부호가 반대였다.)  → [Lq,1].
    use_mean/use_cov 로 평균항·비대각 항의 기여를 분리한다(use_cov=False 면 대각만)."""
    qf = q.float(); b = torch.zeros(qf.shape[0], device=qf.device)
    if use_mean: b = b - (qf @ mu.float()) / math.sqrt(D)
    cv = cov.float() if use_cov else torch.diag(torch.diagonal(cov.float()))
    b = b + ((qf @ cv) * qf).sum(1) / (2.0 * D)
    return b.unsqueeze(1)

@torch.no_grad()
def attn_bias(Q, K, V, bias_fn, n_past, qchunk=585):
    """softmax(QKᵀ/√d − b)V. b 는 과거 n_past 키에만 적용(self 는 0). bias_fn(h, q[Lq,D]) → [Lq, n_past] 또는 [Lq,1].
    반환 (out [Lq,H,D], cache_mass [H] = 과거 키에 간 확률 질량 평균, b_stats [H] = (median, max) of b)."""
    Lq = Q.shape[0]; out = torch.empty(Lq, H, D, device=Q.device); mass = torch.zeros(H, device=Q.device)
    bmed = torch.zeros(H, device=Q.device); bmax = torch.zeros(H, device=Q.device)
    for h in range(H):
        kt = K[:, h].float().T; vv = V[:, h].float(); acc = 0.0; bm = []; bx = 0.0
        for s in range(0, Lq, qchunk):
            q = Q[s:s+qchunk, h].float()
            logits = (q @ kt) / math.sqrt(D)
            if bias_fn is not None:
                b = bias_fn(h, q)
                logits[:, :n_past] = logits[:, :n_past] - b
                bm.append(b.flatten()[:: max(1, b.numel() // 4096)]); bx = max(bx, b.max().item())
            P = torch.softmax(logits, dim=-1)
            acc += P[:, :n_past].sum().item()
            out[s:s+qchunk, h] = P @ vv
        mass[h] = acc / Lq
        if bm: bmed[h] = torch.cat(bm).median(); bmax[h] = bx
    return out, mass, (bmed, bmax)

# ----------------------------------------------------------------------------- one (prompt, layer)
@torch.no_grad()
def run(p, l):
    t_start = time.time()
    c20 = ld(p, 20, l, 4); s0, s1 = c20["self_rel"]; n20 = int(c20["local_end_index"])
    X20K = c20["src_k"][0, s0:s1].to(dev).float(); X20V = c20["src_v"][0, s0:s1].to(dev).float()
    q20 = torch.cat([ld(p, 20, l, s)["q"][0] for s in range(4)] + [c20["q"][0]], 0).to(dev).float()   # [5L,H,D]
    c21 = [ld(p, 21, l, s) for s in range(4)]
    t0, t1 = c21[0]["self_rel"]; n21 = int(c21[0]["local_end_index"]); seg21 = slice(t0 - L, t0)
    K21 = c21[0]["src_k"][0, :n21].to(dev).float(); V21 = c21[0]["src_v"][0, :n21].to(dev).float()
    chk = {"K": rel(K21[seg21], X20K), "V": rel(V21[seg21], X20V)}
    Kref = K21.clone(); Vref = V21.clone(); Kref[seg21] = X20K; Vref[seg21] = X20V     # 참조 창 (chunk20 은 실제 쓰기)
    n_past = t0; nch = n_past // L; assert nch * L == n_past
    Kpast = Kref[:n_past]; Vpast = Vref[:n_past]

    # --- INT2 per chunk (chunk-local grid) + V fixed
    K0 = torch.empty_like(Kpast); stepK = torch.empty_like(Kpast); V0 = torch.empty_like(Vpast)
    for j in range(nch):
        sl = slice(j * L, (j + 1) * L)
        K0[sl], stepK[sl] = rtn_int2(Kpast[sl]); V0[sl], _ = rtn_int2(Vpast[sl])
    E0 = Kpast - K0
    res = {"prompt": p, "layer": l, "int8_vs_write": chk, "n_past_chunks": nch,
           "resid_E0": resid_stats(E0[-L:], stepK[-L:]),             # chunk 20 (실제 쓰기) 기준으로 보고
           "bytes_k_head_chunk": {"INT2": int2_bytes_k(), "K4": int4_bytes_k(),
                                  **{f"INT2+PSAC{r}": int2_bytes_k() + psac_bytes(r) for r in RANKS},
                                  **{f"INT2+PSAC{r}+R4d": int2_bytes_k() + psac_bytes(r) + r4d_bytes() for r in RANKS},
                                  **{f"INT2+PSAC{r}+R4f": int2_bytes_k() + psac_bytes(r) + r4f_bytes() for r in RANKS}},
           "arms": {}}

    # --- query bases
    bases = {}
    if "B" in VARIANTS: bases["B"] = q20
    if "E" in VARIANTS: bases["E"] = rope_shift_frames(q20)
    if "D" in VARIANTS: bases["D"] = c21[0]["q"][0].to(dev).float()
    if "N" in VARIANTS:                                               # 대조: 무작위 직교 기저 (query 와 무관) — PSAC 방향 선택의 몫을 분리
        g = torch.Generator(device="cpu").manual_seed(1000 * p + l)
        bases["N"] = torch.randn(4096, H, D, generator=g).to(dev)     # query 대신 등방 가우시안 → 고유벡터가 무작위 직교 기저

    # --- Δ² 와 pooled Δ²
    step_sq = stepK ** 2                                            # [n_past,H,D]
    step_sq_pooled = step_sq.mean(0, keepdim=True).expand_as(step_sq)

    # --- 검산: 균등 모형 Var=Δ²/12 를 R4d 식에 넣으면 R1(pooled) 과 같은 b
    qchk = c21[1]["q"][0, :64].to(dev).float()
    b1 = bias_delta(qchk[:, 0], step_sq_pooled[:, 0]); b4 = bias_diag(qchk[:, 0], step_sq_pooled[0, 0] / 12).expand_as(b1)
    res["selfcheck_R4d_vs_R1pooled_maxdiff"] = (b1 - b4).abs().max().item()

    # --- 잔차 통계 함수 (chunk 별 자기 잔차의 head 별 평균·분산·공분산 — 저장 시점에 계산 가능한 양)
    def stats_fns(Er):
        f16 = lambda t: t.to(torch.float16).float()                                            # 저장 형식(fp16) 반올림
        mu_c = f16(torch.stack([Er[j*L:(j+1)*L].mean(0) for j in range(nch)]))                    # [nch,H,D]
        var_c = f16(torch.stack([Er[j*L:(j+1)*L].var(0, unbiased=False) for j in range(nch)]))     # [nch,H,D]
        cov_c = f16(torch.stack([torch.stack([((Er[j*L:(j+1)*L, h] - mu_c[j, h]).T @ (Er[j*L:(j+1)*L, h] - mu_c[j, h])) / L
                                               for h in range(H)]) for j in range(nch)]))         # [nch,H,D,D]
        def mk_diag(h, q, var_c=var_c):
            return torch.cat([bias_diag(q, var_c[j, h]).expand(q.shape[0], L) for j in range(nch)], 1)
        def mk_full(h, q, mu_c=mu_c, cov_c=cov_c):                                                # 평균 + 전체 공분산
            return torch.cat([bias_full(q, mu_c[j, h], cov_c[j, h]).expand(q.shape[0], L) for j in range(nch)], 1)
        def mk_mean_diag(h, q, mu_c=mu_c, cov_c=cov_c):                                           # 평균 + 대각만 (평균항의 몫 분리)
            return torch.cat([bias_full(q, mu_c[j, h], cov_c[j, h], use_cov=False).expand(q.shape[0], L) for j in range(nch)], 1)
        return mk_diag, mk_full, mk_mean_diag

    SCOPE = os.environ.get("PSAC_SCOPE", "all")                      # all: 과거 6 chunk 전부 팔 처리(오프라인 대응) / chunk20: 저장 1회→다음 chunk 읽기
    seg20 = slice(n_past - L, n_past)
    def wrap(Kp, bfn):
        """SCOPE=chunk20: chunk 15~19 는 모든 팔에서 K0 + Δ=scale λ1(=bias_delta λ0.25) 로 고정하고 chunk 20 구간만 (Kp, bfn) 을 쓴다."""
        if SCOPE != "chunk20": return (Kp, bfn)
        Kw = K0.clone(); Kw[seg20] = Kp[seg20]
        def bf(h, q, bfn=bfn):
            b = bias_delta(q, step_sq[:, h], 0.25)                      # [Lq, n_past]
            if bfn is None: b[:, seg20] = 0
            else:
                bb = bfn(h, q); b[:, seg20] = (bb if bb.shape[1] == n_past else bb.expand(q.shape[0], n_past))[:, seg20]
            return b
        return (Kw, bf)
    res["scope"] = SCOPE; res["final_bf16"] = FINAL_BF16
    # --- arms: K 구성과 bias 함수
    arms = {"R0": (K0, None), "R1": (K0, lambda h, q: bias_delta(q, step_sq[:, h])),
            "R1p": (K0, lambda h, q: bias_delta(q, step_sq_pooled[:, h]))}
    for lam in LAMBDAS[1:]:                                           # Δ 정의 보정 대조: Δ=2·scale 이 2배 과대면 λ0.25 가 균등 모형의 정답
        arms[f"R1_l{lam}"] = (K0, lambda h, q, lam=lam: bias_delta(q, step_sq[:, h], lam))
    d0, f0, m0 = stats_fns(E0)                                        # PSAC 없이 잔여(=E0) 통계 보정만: PSAC 과 보정식의 기여 분리
    arms["R4d0"] = (K0, d0); arms["R4f0"] = (K0, f0); arms["R4m0"] = (K0, m0)
    for r in RANKS:
        for v, Qv in bases.items():
            U = query_basis(Qv, r)
            Ek = psac_keep(E0, U); K1 = K0 + Ek; Er = Kpast - K1
            tag = f"r{r}_{v}"
            res.setdefault("resid_Erem", {})[tag] = resid_stats(Er[-L:], stepK[-L:])
            res.setdefault("psac_energy_kept", {})[tag] = [(Ek[-L:, h].norm() ** 2 / E0[-L:, h].norm() ** 2).item() for h in range(H)]
            mk_diag, mk_full, mk_mean = stats_fns(Er)
            arms[f"R2_{tag}"] = (K1, None)
            arms[f"R4m_{tag}"] = (K1, mk_mean)
            for lam in LAMBDAS:
                arms[f"R3_{tag}_l{lam}"] = (K1, lambda h, q, lam=lam: bias_delta(q, step_sq[:, h], lam))
            arms[f"R4d_{tag}"] = (K1, mk_diag)
            arms[f"R4f_{tag}"] = (K1, mk_full)

    # --- K4V2 기준선 팔 (K INT4, V INT2 동일): 무보정 / TUM Δ=scale λ1 (= 이 스크립트 단위로 λ0.25) — P3 K4V2+λ1 의 대응물
    K4 = torch.empty_like(Kpast); step4 = torch.empty_like(Kpast)
    for j in range(nch):
        sl = slice(j * L, (j + 1) * L); K4[sl], step4[sl] = rtn_int2(Kpast[sl], bits=4)
    step4_sq = step4 ** 2
    arms["K4_R0"] = (K4, None)
    arms["K4_R1s"] = (K4, lambda h, q: bias_delta(q, step4_sq[:, h], 0.25))
    arms["K4_R1"] = (K4, lambda h, q: bias_delta(q, step4_sq[:, h], 1.0))
    res["resid_K4"] = resid_stats((Kpast - K4)[-L:], step4[-L:])
    flt = os.environ.get("ARMS_FILTER")
    if flt:
        keep = tuple(flt.split(",")); arms = {k: v for k, v in arms.items() if k.startswith(keep)}
    arms = {k: wrap(*v) for k, v in arms.items()}

    # --- evaluate on chunk 21 step 0~3 (step 0 는 D 의 표본 내 → 따로 보고)
    for s in range(4):
        q = c21[s]["q"][0].to(dev).float()
        Kr = Kref.clone(); Vr = Vref.clone()
        if s > 0:
            Kr[t0:t1] = c21[s]["src_k"][0, t0:t1].to(dev).float(); Vr[t0:t1] = c21[s]["src_v"][0, t0:t1].to(dev).float()
        ref, ref_mass, _ = attn_bias(q, Kr, Vr, None, n_past)
        res.setdefault("ref_cache_mass", {})[f"step{s}"] = ref_mass.tolist()
        Vw = Vr.clone(); Vw[:n_past] = V0
        for name, (Kp, bfn) in arms.items():
            Kw = Kr.clone(); Kw[:n_past] = Kp.to(torch.bfloat16).float() if FINAL_BF16 else Kp   # K_FINAL_BF16=1: 복원 K 를 BF16 으로 저장·읽는 경로까지 재현
            out, mass, (bmed, bmax) = attn_bias(q, Kw, Vw, bfn, n_past)
            a = res["arms"].setdefault(name, {})
            a[f"step{s}"] = {"rel": rel(out, ref), "rel_head": [rel(out[:, h], ref[:, h]) for h in range(H)],
                             "cache_mass": mass.tolist(), "b_med": bmed.tolist(), "b_max": bmax.tolist()}
    res["seconds"] = time.time() - t_start
    res["peak_gb"] = torch.cuda.max_memory_allocated() / 2**30
    return res

def summarize(o):
    a = o["arms"]; m = lambda k: sum(a[k][f"step{s}"]["rel"] for s in (1, 2, 3)) / 3
    keys = [k for k in ("R0", "R1", "R1p", "R1_l0.25", "K4_R0", "K4_R1s") if k in a] + [k for k in a if k.startswith("R2_")] + [k for k in a if k.startswith("R4d_")]
    s = " ".join(f"{k} {m(k):.4f}" for k in keys)
    e0 = o["resid_E0"]; vr = sum(x["var_ratio_med"] for x in e0) / H; er = sum(x["cov_eff_rank"] for x in e0) / H
    return (f"p{o['prompt']} L{o['layer']:2d} | E0 var/uniform {vr:.2f} effrank {er:.1f} | {s} | "
            f"chk {o['selfcheck_R4d_vs_R1pooled_maxdiff']:.1e} | {o['seconds']:.0f}s peak {o['peak_gb']:.1f}GB")

if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    print(f"[cand1_diag] root={ROOT} prompts={PROMPTS} layers={LAYERS} ranks={RANKS} variants={VARIANTS} → {OUT}", flush=True)
    print(f"[cand1_diag] bytes K head·chunk: INT2 {int2_bytes_k():,} | K4 {int4_bytes_k():,} | "
          + " | ".join(f"INT2+PSAC{r} {int2_bytes_k()+psac_bytes(r):,} ({100*(int2_bytes_k()+psac_bytes(r))/int4_bytes_k():.0f}% of K4)" for r in RANKS), flush=True)
    rows = []; total = len(LAYERS) * len(PROMPTS); n = 0
    for l in LAYERS:
        for p in PROMPTS:
            torch.cuda.reset_peak_memory_stats()
            o = run(p, l); rows.append(o); n += 1
            print(f"[{n}/{total}] " + summarize(o), flush=True)
            json.dump({"root": ROOT, "layers": LAYERS, "prompts": PROMPTS, "ranks": RANKS, "variants": VARIANTS,
                       "lambdas": LAMBDAS, "H_note": "H(누적 과거 query 통계) 는 이 캡처에 없음 — 새 캡처 후", "rows": rows},
                      open(OUT, "w"), indent=1, ensure_ascii=False)
            torch.cuda.empty_cache()
    print(f"[cand1_diag] DONE {n}/{total} → {OUT}", flush=True)
