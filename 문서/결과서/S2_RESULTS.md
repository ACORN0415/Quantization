# 세션 2 결과 보고 — teacher-forced 분리, KIVI 점검, sink, 임계점 탐색

작성: 2026-09-16 · 대상 지시서: `HANDOFF_session2.md` · 선행: `RESULTS_4090_session.md`
환경: 세션 1과 동일 RTX 4090 24GB. 세션 1의 `patches/`·`results/`를 그대로 이어 사용.

**Step F·G·H·I·J 전부 완료. Step K만 하드웨어 미보유로 제외.**

---

## 0. 한 줄 요약

세션 1의 "누적 없음" 결론은 **틀렸습니다.** 누적을 잴 수 없는 지표로 낸 결론이었고, teacher-forced 분리와 reference-free 지표 양쪽 모두에서 뒤집혔습니다.

그리고 그 누적을 막는 것이 **attention sink**임을 세 단계로 확정했습니다. (1) chunk 7 민감도 계단의 원인이 암묵적 sink(첫 chunk) 상실임을 pseudo-sink 실험으로 보였고, (2) 명시적 sink를 가진 LongLive에서 그 계단이 실제로 사라지며, (3) LongLive의 INT4 KV 양자화는 캐시 오차가 2.3배 큼에도 **사실상 무손실**(MUSIQ −0.3 대 Self-Forcing −11.0)입니다. sink는 주입량이 아니라 **누적을 차단**하고, 그 효과는 내용의 정밀도가 아니라 **위치 보존**에서 옵니다.

부수적으로 UCSD 코드에서 버그를 **두 개** 찾았고(둘째는 첫째를 고쳐야 드러남), **key 오차가 value 오차보다 약 9배 비싸다**는 통제 실험 결과를 얻었습니다.

---

## 1. Step F — teacher-forced vs free-running 분리 (§1)

### 구현: lockstep (디스크 덤프 대신)

지시서는 BF16 K/V를 디스크에 덤프해 주입하는 방식을 제안했으나, 실측 용량이 **54.3GB**(chunk·레이어당 28.8MB × 30레이어 × 21chunk × 3프롬프트)로 지시서 추정("수 GB")의 10배였습니다. chunk마다 863MB를 쓰고 읽으면 I/O가 연산(chunk당 약 2초)을 압도합니다.

대신 한 프로세스 안에서 캐시 두 벌을 들고 chunk마다 ①기준 캐시 스냅샷 → ②BF16 pass(기준 캐시 전진) → ③스냅샷 위에서 양자화 pass를 도는 **lockstep** 방식으로 구현했습니다. 디스크 사용 0바이트, peak VRAM 15.2GB(24GB 여유).

반영한 조건:
- 두 pass가 같은 chunk 노이즈·re-noise generator를 사용. 재계산 전 같은 seed로 generator 재생성
- 기준 캐시는 BF16 pass로만 갱신. 양자화 pass의 K/V는 작업용 복사본에만 기록
- fake-quant는 가중치 두 벌 대신 **forward 토글**(`ToggleQuantizer`). 가중치 양자화는 forward마다 즉석 수행
- TF 정의: 슬롯 0..t−1을 BF16 K/V로 채우고 **그 캐시를 한 번 양자화**한 뒤 chunk t 생성

**UCSD 패치와의 차이 (지시서 요청 기록사항)**: 패치는 `quant_state`가 있으면 그것을 역양자화해 쓰고 없으면 BF16 `k`/`v`를 씁니다. 그대로 두면 각 chunk의 **첫 denoising step만 양자화되지 않은 깨끗한 컨텍스트**를 보게 되어 FR과 구조가 달라집니다. 따라서 `anchor_work_cache`에서 매 chunk 명시적으로 `quant_state = quantize_kv(ref_k, ref_v)`를 미리 계산합니다. "기준 캐시 복사본을 한 번 양자화"와 같은 연산이지만 **자동으로 되지는 않아** 별도 처리가 필요했습니다.

### 관문 (§8 규칙: 통과 전 다른 설정 금지)

| 관문 | 결과 |
|---|---|
| 양자화 끈 lockstep 재계산 == 기준 (`torch.equal`) | **PASS** (63/63 chunk) |
| lockstep 기준 궤적 == 세션 1 FR BF16 | **PASS** (21/21 chunk) |
| sink_size=3 설정에서도 위 관문 | **PASS** |

두 번째는 지시서에 없지만 필수입니다. 기준 궤적이 세션 1과 다르면 `FR − TF` 뺄셈 자체가 성립하지 않습니다.

### 결과 (chunk 20, 프롬프트 3개 평균)

| 설정 | 주입 TF | FR | 전파 FR−TF | MUSIQ Δ |
|---|---:|---:|---:|---:|
| RTN INT4 (KV) | 0.217 | 0.808 | 0.591 | −11.0 |
| RTN INT2 (KV) | 0.527 | 1.850 | 1.323 | −42.8 |
| W4A4 | 0.272 | 1.337 | 1.065 | −18.3 |
| W8A8 (FP8) | 0.063 | 0.937 | 0.875 | +1.4 |
| K/V proj만 W4A4 | 0.143 | 1.087 | 0.944 | −7.8 |

### 판정

**"propagation이 chunk 7 이후 증가하는가" → 예, 다섯 설정 모두.** 21 chunk 안에서 평탄해지지 않습니다. 따라서 FR의 chunk 7 꺾임은 민감도 계단만으로 설명되지 않고 **민감도 계단 + 전파 증가의 합**입니다.

**"INT2에서만 증가하는가" → 아니오.** INT4 포함 전부 증가합니다. **세션 1의 "INT4는 누적 없음"은 틀렸습니다.**

추가로 얻은 것 둘:

- **민감도 계단은 캐시 경로에 특이적입니다.** chunk 7 계단이 KV 캐시/K·V projection을 건드리는 설정(kv_int2, kv_int4, kv_only_w4a4)에서만 뚜렷하고, 모든 Linear를 균일 양자화하는 W4A4·W8A8에서는 거의 없습니다.
- **퇴출은 민감도를 올리지만 오염은 덜어냅니다.** chunk 7에서 주입은 뛰는데 propagation은 일시 하락합니다(INT2 0.334→0.265, INT4 0.381→0.357). 가장 오래되고 오염된 chunk가 버려지기 때문으로 보입니다.

### 중요한 해석 주의 — 전파량은 피해량이 아닙니다

`FR − TF`는 정의상 **캐시 전파와 카오스 이탈을 모두 포함**합니다. W8A8은 전파 0.875로 RTN INT4(0.591)보다 큰데 MUSIQ는 **+1.4로 멀쩡**하고, RTN INT4는 전파가 작은데 **−11.0으로 실제 열화**합니다. W8A8의 FR(0.937)이 카오스 바닥선(0.738) 근처라 전파의 대부분이 이탈이기 때문입니다.

**결과표에는 전파와 reference-free 지표를 반드시 함께 실어야 합니다.**

### 보조 지표 — chunk가 넘겨주는 오염량

lockstep이라 거의 공짜로 얻을 수 있어 추가했습니다(양자화 전, 새로 계산된 chunk t의 K/V를 기준 run과 비교, 레이어 평균). RTN INT4 정상 구간에서 K 0.23 / V 0.47로, 해당 chunk의 latent 오차(0.22)보다 크고 **V가 K의 2배**입니다.

---

## 2. Step G — KIVI 구현 점검 (§2)

### 점검 항목 답변

- **RoPE 후 적용 여부**: UCSD 패치가 `roped_key`가 담긴 `cache_k`를 넘기므로 **RoPE 후가 맞습니다.**
- **축 방향**: keys `reduce_dims=(2,)`는 블록 내 토큰 축 축약 → (block, head, channel)당 scale = per-channel ✓. values `reduce_dims=(3,4)`는 head·channel 축약 → per-token ✓. **둘 다 논문과 일치합니다.** 따라서 §2.3의 "(a) 채널 축 방향만 고쳐 재실행"은 해당 사항이 없습니다.
- **residual window**: 최근 토큰을 FP16으로 유지하는 장치가 **없습니다.** 논문 사양과의 차이이므로 표기는 **"KIVI-style (residual window 없음)"** 이어야 합니다.

### 버그 1 — zero-point 클램프

`kv_quant/utils.py::quantize_asym`이 zero point를 `[qmin, qmax]`로 클램프합니다. zero point는 코드가 아니라 **오프셋**이라 이 범위에 갇힐 이유가 없고, 0을 걸치지 않는 그룹(전부 양수/음수)은 통째로 한 코드로 무너집니다. RoPE 후 키는 16토큰 블록 안에서 한쪽 부호로 몰리기 쉬워 이 조건에 상시 걸립니다.

최소 재현(원본 `b4c0936`에서 확인):
```
asymmetric (KIVI/QAQ path): rel_err = 0.9296
symmetric  (RTN path)     : rel_err = 0.0399
true zero point range   : [-516, -97]
after .clamp(0, 15)     : [0, 0]
groups altered by clamp : 100.0%
```

**RTN은 `quantize_sym`을 써서 무관합니다** — 세션 1에서 RTN만 멀쩡했던 이유가 정확히 이것입니다. `qaq.py`도 같은 헬퍼를 씁니다.

### 버그 2 — 첫 버그가 가리고 있던 것

클램프만 제거하면 **NaN이 납니다.** `EPS = 1e-8`이 scale을 저장하는 fp16에서 0으로 언더플로하고(fp16 최소 subnormal ≈ 6e-8), 동시에 `zp = -x_min/1e-8`이 `inf`가 되어 `(q − inf) × 0 = NaN`입니다. 원본에서는 zp가 `[0,15]`에 갇혀 있어 NaN 대신 **그룹이 조용히 0이 되는** 형태로 숨어 있었습니다. 실제 63프레임 TF 실행 전체가 NaN으로 죽어서 발견했습니다.

**zp fp16 overflow는 원인이 아니었습니다.** 실제 post-RoPE 키에서 측정한 `|zp|` 분포는 p50 8, p99 51, p99.99 237, **max 1200**으로 fp16 한계 대비 **55배 여유**, 초과 그룹 0개입니다.

### 수정 — offset 형태

zero point 코드 대신 **그룹 최소값**을 저장합니다. 수학적으로 동일한 비대칭 양자화이고 저장 비용도 같으며(그룹당 fp16 하나), `x_min`은 데이터 값이라 fp16이 항상 표현하고 상수 그룹은 정확히 복원됩니다.

| 그룹 | 수정 후 asym | sym (RTN) |
|---|---:|---:|
| 0 중심 | **0.0650** | 0.0852 |
| 전부 양수 | **0.0013** | 0.0400 |
| 전부 음수 | **0.0013** | 0.0400 |
| bf16 해상도 이하 분산 | **0.0000**, NaN 없음 | — |

### 수정 효과 (실제 rollout)

| 방법 | key | value |
|---|---:|---:|
| RTN INT4 | 0.0278 | 0.0371 |
| RTN INT2 | 0.1298 | 0.1728 |
| KIVI INT4 (버그본) | 0.1841 | 0.0569 |
| **KIVI INT4 (수정본)** | **0.0176** | 0.0559 |
| KIVI INT2 (수정본) | 0.0770 | 0.2251 |

key 오차가 **10.5배 개선**되어 RTN INT4(0.0278)**보다 좋아졌습니다.** 같은 비트폭에서 비대칭이 대칭보다 나은 것이 기대되는 순서이므로, **세션 1의 KIVI 열세는 전적으로 구현 버그였음이 확정**됩니다.

value는 거의 안 변했습니다(0.0569→0.0559). KIVI의 value는 `reduce_dims=(3,4)`로 head·channel을 함께 묶어 그룹이 0을 걸칠 확률이 높아 버그 노출이 적었기 때문입니다.

**진짜 기법 차이도 하나 드러났습니다**: KIVI INT2의 value 오차(0.2251)가 RTN INT2(0.1728)보다 나쁩니다. KIVI는 값을 per-token(스케일당 1536값)으로 묶고 RTN은 (block,head,channel)별로 잘게 묶습니다. 2비트에서는 이 거친 그룹화가 손해입니다. 논문 기술: **"KIVI는 key에 강하고 저비트 value에 약하다."**

### 세션 1 KIVI 수치 처리

결과표에서 **"버그 있는 구현"으로 각주 처리**해야 합니다. 대체할 수정본 수치는 위 표와 §5에 있습니다.

### GitHub issue

`~/gpu/issue/ISSUE_zp_clamp.md`(161줄)에 두 버그와 `factory.py`의 `NameError`를 묶어 작성했고, 실행 가능한 최소 재현 스크립트(`repro_zp_clamp.py`)를 첨부했습니다. **게시하지 않았습니다** — `gh` 미설치이고 이 세션은 비대화형이라 OAuth 인증이 불가능하며, 공개 저장소에 계정 명의로 올라가는 작업이라 확인 없이 진행하지 않았습니다.

---

## 3. chunk 7 계단의 원인 — 암묵적 sink 상실

### 재해석

지적대로 chunk 6과 7은 캐시 슬롯 수도 양자화 오차 크기도 같으므로, 주입량이 아니라 **같은 캐시 오차에 대한 모델 민감도**가 커진 것입니다.

### 가설 (b) RoPE OOD — 약화됨

`causal_rope_apply`는 `start_frame = current_start // frame_seqlen`, 즉 **절대 프레임 인덱스**를 쓰고 퇴출 후 상대 재인덱싱이 없습니다. `freqs` 테이블은 1024개라 63프레임에 충분합니다(인덱싱 오류 없음).

그러나 **RoPE 내적은 상대 거리에만 의존**하고 캐시가 21프레임으로 제한되므로 **query–key 상대 거리는 항상 학습 범위(≤21) 안에 머뭅니다.** 벗어나는 것은 절대 위치뿐이고, 그것은 attention 점수에 직접 들어가지 않습니다. 따라서 (b)의 여지는 작습니다.

### 가설 (a) pseudo-sink 실험 — 지지됨

**코드 변경이 불필요했습니다.** `sink_size`가 이미 구현돼 있고 퇴출 로직이 이를 존중하는데(`causal_model.py:483` "we keep the first `sink_size` frames unchanged when rolling the KV cache") 기본값이 0이라 꺼져 있었을 뿐입니다. `model_kwargs.sink_size: 3` = 정확히 chunk 0(3 latent frame)을 고정합니다.

| | chunk 6→7 계단 (sink 없음) | sink=3 | 감소 |
|---|---:|---:|---:|
| RTN INT4 | +0.0727 | +0.0241 | **−67%** |
| RTN INT2 | +0.1846 | +0.0118 | **−94%** |

chunk 0~6은 두 설정이 **완전히 동일**합니다(퇴출 전이라 sink 무효 — 내부 정합성 확인). 부수적으로 INT4는 정상 구간 주입량도 0.21→0.175로 약 17% 낮아졌습니다.

**결론**: Self-Forcing에 명시적 sink는 없지만 **첫 chunk가 사실상 attention sink 역할을 하고 있었고, 그 퇴출 시점이 민감도 계단의 원인**입니다. LongLive가 명시적 sink를 넣은 이유와 맞물립니다.

**유보**: 잔여 계단(INT4 +0.024)이 남고, 그것이 (b)의 몫인지 제3의 원인인지는 이 실험만으로 가려지지 않습니다.

**참고**: sink3은 지시서가 요구한 TF만 실행했으므로 FR이 없고, 따라서 sink3의 propagation은 산출하지 않았습니다.

---

## 4. V > K 오염량 — QK-norm 가설 확인, 그러나 결론은 반대

### 코드 확인 (가설 맞음)

```python
q = self.norm_q(self.q(x))   # WanRMSNorm
k = self.norm_k(self.k(x))   # WanRMSNorm
v = self.v(x)                # norm 없음
```

`qk_norm=True`가 기본값이고 **V만 정규화가 없습니다.** K는 RMSNorm으로 동적 범위가 묶입니다.

### 그러나 통제 실험은 반대 결론 — key가 더 비쌉니다

세션 1의 "value가 더 비싸다"는 힌트는 **서로 다른 방법 간 비교**(RTN INT2 vs 버그 있던 KIVI)에서 나온 것이라 교란이 있었습니다. 같은 양자화기에서 비트 배분만 바꾼 통제 실험 결과는 반대입니다.

| 설정 | key 오차 | value 오차 | c20 rel_err | MUSIQ Δ |
|---|---:|---:|---:|---:|
| INT4 (기준) | 0.0278 | 0.0369 | 0.808 | −11.0 |
| **K4V2** (키 좋고 값 나쁨) | 0.0280 | **0.1788** | 0.973 | **−13.8** |
| **K2V4** (키 나쁘고 값 좋음) | **0.1291** | 0.0358 | 1.622 | **−36.9** |

INT4 대비 **value 오차 4.8배 → MUSIQ −2.8점**, **key 오차 4.6배 → MUSIQ −25.9점**. 단위 오차당 **key가 약 9배 비쌉니다.**

### 두 관찰의 관계

모순이 아니라 맞물립니다. **키는 softmax를 통과하므로 지수적으로 민감**하고 값은 선형으로 기여합니다. 키 오차는 "어느 토큰을 볼지"를 바꾸고 값 오차는 "무엇을 가져올지"만 흐립니다. QK-norm은 키의 동적 범위를 묶어 **같은 비트폭에서 키 오차를 낮게 유지**해 주는데, 하필 그 키가 민감한 쪽이라 유리한 설계입니다.

**실무 함의: 비트는 key에 몰아야 합니다.** K2V4는 정확히 최악의 배분입니다. 다만 value도 공짜는 아닙니다(−2.8).

---

## 5. Step I — 임계점 탐색 (§4)

| 설정 | key 오차 | value 오차 | c20 rel_err | MUSIQ Δ |
|---|---:|---:|---:|---:|
| 대조군(노이즈 섭동) | — | — | 0.738 | **+4.3** |
| KIVI INT4 (수정) | 0.0176 | 0.0559 | 0.753 | −10.3 |
| RTN INT4 b16 | 0.0278 | 0.0369 | 0.808 | −11.0 |
| RTN INT4 b64 | 0.0313 | 0.0443 | 0.785 | −11.0 |
| RTN INT4 b256 | 0.0370 | 0.0541 | 0.795 | −13.3 |
| RTN INT4 b1024 | 0.0418 | 0.0634 | 0.806 | −14.5 |
| RTN INT3 b16 | 0.0601 | 0.0807 | 0.991 | −22.4 |
| KIVI INT2 (수정) | 0.0770 | 0.2251 | 1.078 | −21.5 |
| K4V2 | 0.0280 | 0.1788 | 0.973 | −13.8 |
| K2V4 | 0.1291 | 0.0358 | 1.622 | −36.9 |
| RTN INT2 b16 | 0.1317 | 0.1814 | 1.850 | −42.8 |

### "붕괴 무릎"은 없습니다

latent rel_err로 보면 block size를 16→1024로 키워 key 오차를 50% 올려도 0.808→0.806으로 변화가 없어 무릎이 있는 것처럼 보입니다. 그러나 **MUSIQ로 보면 같은 구간에서 −11.0 → −14.5로 꾸준히 나빠집니다.** latent 지표가 카오스 바닥선에 눌려 구분하지 못한 것입니다.

MUSIQ는 key 오차에 대해 **연속적이고 거의 단조**입니다(0.028→−11.0, 0.037→−13.3, 0.042→−14.5, 0.060→−22.4, 0.129→−36.9, 0.132→−42.8). 즉 **임계점이 아니라 연속 열화**이고, 세션 1이 본 "임계"는 지표가 천장을 넘어야 비로소 보이기 시작한 시점이었습니다.

### INT3 참고

INT3(b16)은 key 0.0601 / MUSIQ −22.4로 INT4와 INT2 사이에 위치합니다. 비트폭을 줄이는 것보다 **block size를 키우는 쪽이 같은 key 오차 대비 손해가 적습니다**(b1024 key 0.0418 → −14.5 vs INT3 key 0.0601 → −22.4는 key 오차가 다르니 직접 비교는 아님).

### 구현상 필요했던 변경

- `base.py`가 key/value 비트를 `(2, 4)`로만 허용해 **INT3이 막혀 있었습니다.** 양자화 수식과 `packed_bytes` 모두 비트폭에 무관하므로 인위적 가드로 판단해 **2~8로 확대**했습니다.
- `--kv_key_bits` / `--kv_value_bits` 인자를 추가했습니다(비대칭 배분 실험용).

---

## 6. Step J — reference-free 지표 (§5)

VBench 전체 대신 `pyiqa`의 **MUSIQ**(koniq)를 썼습니다. 세션 1의 기존 영상에 후처리로 적용했고 재생성은 없습니다. chunk별 중앙 픽셀 프레임 1장에 적용했습니다.

**핵심: 대조군은 오히려 올라갑니다(+4.3).** 카오스 이탈이 품질을 해치지 않는다는 직접 증거이고, 이것이 모든 해석의 기준선입니다.

### 두 지표 중 어느 쪽도 단독으로는 부족합니다

가장 날카로운 두 사례:

- **대조군(rel_err 0.738, MUSIQ +4.3) vs 수정본 KIVI INT4(rel_err 0.753, MUSIQ −10.3)**: latent 거리가 사실상 같은데 품질은 정반대입니다. **latent L2는 품질 저하를 원리적으로 감지하지 못합니다.**
- **버그본 KIVI(rel_err 1.277, MUSIQ −3.4)**: 역방향입니다. 멀리 갔는데 MUSIQ는 멀쩡하다고 합니다. 육안 확인 결과 버그본은 선명·고대비지만 **BF16과 다른 장면**이고, 수정본은 BF16과 거의 같은 장면이되 약간 부드럽습니다. MUSIQ가 선명도를 보상하는 no-reference 지표라 "다르지만 쨍한" 출력에 높은 점수를 준 것입니다.

**결론: 반드시 두 지표를 함께 읽어야 하며, 경계 사례는 육안 확인이 필요합니다.** latent L2는 "다른 데로 갔는가", MUSIQ는 "무너졌는가"를 답하고, 둘 다 반대쪽 질문에는 답하지 못합니다.

---

## 7. 세션 1 결론 정정 요약

| 세션 1 주장 | 세션 2 결과 |
|---|---|
| "캐시에서 오차가 누적되지 않는다" | **주입량**은 상수가 맞으나 **전파**는 chunk 축을 따라 증가. 다섯 설정 모두 |
| "RTN INT4는 바닥선과 구분 불가라 안전" | MUSIQ −11.0으로 **실제 열화**. latent 지표가 천장에 막혀 못 본 것 |
| "KIVI가 RTN보다 나쁘다" | **구현 버그**. 수정 후 KIVI INT4가 RTN INT4보다 좋음 |
| "value 오차가 key보다 비싸다" | **반대.** 통제 실험에서 key가 단위 오차당 약 9배 비쌈 |
| "붕괴 임계점이 있다(2.8% 안전 / 13% 붕괴)" | 무릎 없음. **연속 열화**이며 임계처럼 보인 것은 지표 천장 |
| in-situ 캐시 오차가 상수 = 누적 없음 | 상수인 것은 맞으나 이는 **주입량 지표**이지 전파 지표가 아님 |

---

## 8. Step H — LongLive (§3), 완료

지시서 §9에 따라 F/G/I/J 완료 후 착수했고, **끝까지 실행했습니다.**

### 지시서 가정 정정 (중요)

현재 기본 브랜치는 **LongLive 2.0**이며 Wan2.2-TI2V-**5B** 기반, `local_attn_size: 32`, `sink_size: 8`입니다. 지시서가 적은 "Wan2.1-1.3B 기반, sink 3 + window 9(effective 12)"는 **LongLive 1.0**이고 `v1.0` 브랜치로 이동했습니다.

**`v1.0`을 선택했습니다.** Self-Forcing과 같은 Wan2.1-1.3B 베이스라야 차이를 sink/window 설계 탓으로 돌릴 수 있고, 5B는 4090에서 무리입니다. commit `e52d9ef`, 체크포인트 `Efficient-Large-Model/LongLive-1.3B` 8.2GB.

### 캐시 구조 (지시서 요청 도식)

```
캐시 = 12 latent frame = 18720 토큰 = 4 chunk
  [ chunk 0 (sink, 3 frame, 고정) ][ chunk t-2 ][ chunk t-1 ][ chunk t ]
                                    <------ rolling window 9 frame ------>

퇴출 시작: chunk 4   (Self-Forcing은 chunk 7)
```

### 이식

Self-Forcing과 달리 LongLive는 attention 계산용 `temp_k`/`temp_v` **복사본**을 쓰고, 실제 캐시 갱신은 `cache_update_info`를 반환해 `_update_kv_cache`에서 **중앙집중 처리**합니다. 따라서 UCSD 패치를 그대로 적용할 수 없고 훅을 나눠 걸었습니다.

- **읽기 2곳** (`temp_k = kv_cache["k"].clone()`) → `_kv_read()`로 교체, `quant_state`가 있으면 역양자화해 반환. `KV_SINK_BF16`이 켜지면 sink 슬롯만 BF16 원본으로 되돌림
- **쓰기 1곳** (`_update_kv_cache` 말미) → 캐시 재양자화 + in-situ 오차 기록
- seed 고정(chunk별 2줄기 generator)과 chunk별 latent 덤프는 Self-Forcing 패치가 거의 그대로 이식됨 (chunk 루프 구조 동일)

**OOM 수정**: 텍스트 인코더가 `device` 위에 fp32로 생성되어(umT5 fp32 ≈ 22GB) `pipeline.to(dtype=bfloat16)` 단계에서 죽습니다. 변환 전 CPU 경유로 고쳤습니다.

패치: `patches/07_longlive_port.patch` (320줄, 3개 파일)

### 관문

| 관문 | 결과 |
|---|---|
| seed 고정 후 재실행 bit-exact | **PASS** (63/63 chunk) |
| quantizer 훅 추가 후 BF16 회귀 | **PASS** (63/63, bit-exact) |

### 결과 (chunk 20, 프롬프트 3개 평균)

각 모델의 **자기 대조군(카오스 바닥선) 대비 비율**이 비교 가능한 양입니다. 절대 rel_err은 두 모델의 이탈 속도가 달라 직접 비교할 수 없습니다.

| 설정 | rel_err | 바닥선 대비 | MUSIQ Δ |
|---|---:|---:|---:|
| 대조군 1e-3 | 0.792 | 1.00× | +0.3 |
| **RTN INT4** | 0.781 | **0.99×** | **−0.3** |
| RTN INT2 | 0.887 | 1.12× | −22.7 |
| sink만 BF16 + 나머지 INT2 | 0.878 | 1.11× | −21.9 |
| W4A4 | 1.088 | 1.37× | −10.5 |

### §3 질문 판정 — **(a) 꺾임이 사라집니다**

window 경계로 옮겨가지 않습니다. 퇴출 시점(chunk 3→4) 계단을 보면:

| | 계단 |
|---|---:|
| 대조군 | +0.0774 |
| INT4 | +0.0887 |
| **INT2** | **+0.0269** |

Self-Forcing에서 INT2는 chunk 7에서 위로 꺾인 뒤 선형 상승해 1.850까지 갔는데, LongLive에서는 퇴출 계단이 **대조군보다도 작고** 곡선이 평평해집니다.

### 모델 간 비교

| | Self-Forcing | LongLive |
|---|---:|---:|
| 캐시 설계 | sink 없음, window 21 | sink 3 + window 9 |
| 퇴출 시작 | chunk 7 | chunk 4 |
| 카오스 바닥선 (c20) | 0.738 | 0.792 |
| INT4 캐시 오차 (key) | 0.0278 | **0.0640** |
| INT4 바닥선 대비 | 1.09× | **0.99×** |
| INT4 MUSIQ Δ | −11.0 | **−0.3** |
| INT2 바닥선 대비 | **2.51×** | **1.12×** |
| INT2 MUSIQ Δ | −42.8 | −22.7 |

**LongLive는 캐시 오차를 2.3배 더 많이 주입받으면서도 피해는 훨씬 작습니다.** INT4는 사실상 무손실(MUSIQ −0.3)이고, INT2도 피해가 절반입니다.

**sink는 주입량을 줄이는 게 아니라 누적을 차단합니다.** LongLive INT2의 바닥선 대비 비율이 chunk 1의 2.66배에서 chunk 20의 1.12배로 **떨어집니다**. Self-Forcing에서는 반대로 계속 올라갔습니다. 이것이 세션 2의 두 발견을 하나로 잇습니다 — 전파는 실재하고(Step F), 그것을 막는 것이 sink이며(pseudo-sink 실험), 그 예측이 다른 모델에서 재현됩니다(Step H).

### sink 정밀도 가설 — **지지되지 않음**

지시서는 "sink 오차가 상수 편향으로 작용한다"는 가설의 직접 검증을 요청했습니다. `sink만 BF16 + 나머지 INT2`(1.11×, MUSIQ −21.9)가 `전부 INT2`(1.12×, −22.7)와 **사실상 동일**합니다. chunk 4 이후 차이가 노이즈 수준입니다.

**sink의 보호 효과는 내용의 수치 정밀도가 아니라 위치가 보존된다는 사실 자체에서 옵니다.** 실무적으로는 좋은 소식으로, sink를 특별 취급(BF16 유지)할 필요가 없습니다.

### 주의사항

- **W4A4는 모델 간 직접 비교 불가.** LongLive는 LoRA 어댑터 때문에 fakequant 대상 Linear가 **900개**(Self-Forcing 300개)이고, `lora_A`/`lora_B` 가중치까지 양자화됩니다. 모델 간 비교의 주축은 캐시만 건드리는 **KV 설정**입니다.
- **절차상 실수 2건 기록.** (1) sweep 실행 중 `run_ll.sh`를 수정해 bash가 재읽기하며 `kv_int2`가 중복 실행됨(결정적 실행이라 결과는 동일). (2) `--output_folder` 인자를 추가해 놓고 config 덮어쓰기를 연결하지 않아 초기 3개 설정의 영상이 같은 폴더에 덮어써짐. latent는 설정별로 정상 저장되어 있었으므로 `decode_latents.py`로 재디코딩해 복구했습니다(재실행 대비 GPU 시간 대폭 절약).

### 산출물

`results/longlive_vs_selfforcing.csv/png` (3분할: SF 곡선 / LL 곡선 / 바닥선 대비 비율), `results/iq_longlive.csv/png`, 설정별 `results/ll_*/` (kv_err, run.log, videos)

## 9. Step K — 제외

Wan 14B 활성화 outlier 진단은 **H100 80GB가 필요**하며 이 장비는 4090 24GB입니다(14B BF16은 가중치만 약 28GB). 지시서도 별도 인스턴스를 전제했습니다. **이 세션 산출물에서 제외**되며, 별도 H100 세션에서 수행해야 합니다.

---

## 10. 산출물

```
~/gpu/Self-Forcing/results/
  propagation.csv / .png       Step F 핵심 (TF·FR·propagation 3분할)
  threshold.csv / .png         Step I (키/값 분리 산점도)
  iq_curves.csv / .png         Step J (세션 1 전체 설정)
  iq_threshold.csv / .png      Step I 설정 MUSIQ
  iq_kivi.csv / .png           KIVI 버그본 vs 수정본
  error_curves.csv / .png      세션 1 (유지)
  kv_cache_error.csv / .png    세션 1 (유지)
  tf_*/tf.json                 TF 원자료 (kv_int4/int2, w4a4, w8a8, kv_only_w4a4,
                               sink3_kv_int4/int2, kivi_fix_*, gate/gate2)
  thr_*/                       Step I 실행 (stats, kv_err, videos, run.log)
  kivi_fix/                    KIVI 수정본 FR (stats, kv_err, videos)

~/gpu/patches/
  08_kivi_asym_fix.patch       utils.py(offset 형태) + factory.py(kwargs)
  (신규) teacher_forced.py measure_propagation.py measure_threshold.py
         iq_eval.py diag_zp.py run_tf.sh run_sink.sh run_threshold.sh
         self_forcing_dmd_long_sink3.yaml

~/gpu/issue/
  ISSUE_zp_clamp.md            GitHub issue 초안 (미게시)
  repro_zp_clamp.py            최소 재현 스크립트
```

---

## 11. 다음 세션 과제

1. **Step H 완료** — §8의 착수 순서대로. 특히 "sink만 BF16 + 나머지 INT2" 설정
2. **Step K** — H100 별도 세션, 비용 상한 3시간
3. **GitHub issue 게시** — `gh` 설치 + 인증 후 `~/gpu/issue/ISSUE_zp_clamp.md`
4. **residual window** — KIVI 논문 사양의 나머지 절반. 구현 후 재측정하면 "KIVI-style"이 아니라 진짜 KIVI와 비교 가능
5. **chunk 7 잔여 계단** — pseudo-sink 후에도 남는 +0.024의 정체. sink_size를 6, 9로 키우며 잔여가 줄어드는지 보면 (b)와 제3원인을 더 가를 수 있음
6. **더 긴 rollout** — 63프레임은 퇴출 후 14 chunk. propagation이 21 chunk 안에서 평탄해지지 않으므로 어디서 멈추는지 미확인
7. **MUSIQ 보완** — no-reference 지표도 이탈을 품질로 오인할 수 있음(버그본 KIVI 사례). reference 기반 지표(LPIPS 등)를 BF16 대비로 하나 더 붙이면 삼각 측량 가능
