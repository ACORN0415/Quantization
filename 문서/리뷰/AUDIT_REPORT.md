# 감사 보고서 — 세션 1~4 결론 C1~C14 검증

> ⚠ **이 문서는 세션 5 시점 스냅샷이다.** 세션 6·7에서 아래 주장들이 철회·축소됐다:
> C-I(실측 ε 균일·ε=0) **철회** · C-D(구조는 INT4에서만) **철회** · C-R(오라클 정책) **분해** ·
> C-M(0.75GB 8×) **철회, 실측 4.00×** · C-J/C-B/C-H/C-L **범위 축소** · C-A **n=3 예비**.
> **먼저 `CORRECTIONS_2026-09-19.md`를 읽고 해당 항목을 대체해서 읽을 것.**


작성: 2026-09-17 · 지시서: `HANDOFF_audit.md` · 산출물: `~/gpu/audit/`
방침: 기존 결론의 재현·반증만 수행. 새 실험 설계 없음. 원본 결과 파일은 덮어쓰지 않음.

## 0. 판정 요약

| # | 결론 | 판정 | 한 줄 근거 |
|---|---|---|---|
| C1 | INT2 chunk7 꺾임 후 붕괴, INT4는 바닥선 위 소폭 | **통과** | error_curves.csv 수치 전부 재현 + A1_int4/int2 재실행 bit-exact(§B1) |
| C2 | 노이즈 섭동 대조군 L2 ~0.74 포화 | **통과** | CSV 재현 + seed 1·2에서 0.736/0.760 재현(§B5) |
| C3 | 활성화 outlier 채널 없음 | **통과** | diag_act 재실행: per-tensor≈per-token(0.026대), amax/med 1.2~4.6 재현(§B12) |
| C4 | UCSD asym 버그 2개; 수정 후 KIVI INT4 key < RTN | **통과** | 수정 asym = numpy 참조 일치(≤7e-6), 재현 스크립트 존재, kv_cache_error 수치 재현 |
| C5 | TF 주입 상수, FR-TF 전파 증가; TF-MUSIQ 정합 | **통과** | tf_musiq_per_prompt 전 수치 재현(BF16 +3.73, TF_int4 +1.10, FR_int4 −11.03 등) |
| C6 | key 오차가 value보다 단위당 ~9배 비쌈 | **통과(주의)** | threshold.csv 재현. 단 n=3 프롬프트, K4V2/K2V4 각 1설정 — 외삽 주의 |
| C7 | 캐시 구조만으로 INT4 피해 70% 회복; LL INT4=BF16 | **통과** | §5 LOO: A1→A4 +7.69(t=5.36, 견고), LL INT4 +0.11(무효주장 견고) |
| C8 | 첫 chunk sink 아님(8.5%<14.3%); self 48/최근 19/최고령 7% | **통과** | records 재집계 일치 + raw Q/K 독립 full-softmax 재계산 18/18 일치(§B7) |
| C9 | INT2 분포 역전; chunk1부터; key 단독 원인 | **통과** | gateG 표 재현 + K4V2/K2V4 mass 재집계 일치 + §B7 |
| C10 | attention compute-bound(AI=쿼리수); +24%=재구성 | **통과** | CUDA event 독립 재측정: attention 불변, 증가분의 99.6%가 quant+dequant(§B9) |
| C11 | self는 항상 BF16으로 읽힘 | **통과** | causal_model.py 경로 직접 확인(기록→attention→재양자화 순서) |
| C12 | TUM 보정 INT2 54~68% 회복, INT4 무효 | **부분통과 — 수치·서술 정정 필요** | 효과·인과는 확정(경로 무해 t=−1.53, 논문 사양에서도 +10.4 t=3.2 회복). 그러나 헤드라인 +28.25는 **4배 과대 보정**(Δ=2·scale 버그)의 산물이고 논문값 회복은 그 37% (§B6·B8) |
| C13 | sink는 내용 앵커; 노이즈 sink 유해, INT2-오염 무변화 | **통과(freeze 원인 규명)** | E2 관문 latent 재검증(70/70·140/140·210/210), freeze 이상치 = SINK_TENSORS 프롬프트 간 누출 버그 |
| C14 | 슬롯 오염 나이 따라 감소; A2 고정 0.064 일정 | **통과(정의 주의)** | 측정량은 궤적이탈+양자화 총합(세션 4도 F1에서 인정). RTN 멱등성은 수학적으로 확인 |

(비고: "통과" = 수치 재현 + 반증 시도 실패. 자세한 근거와 논문에서 빼거나 고쳐야 할 문장은 아래.)

## 1. 감사에서 새로 발견한 버그 4건

### 버그 A — TUM 보정 항이 논문 값의 4배 (C12)
`tum_correction.py:43` — `step = 2.0 * scale`. RTN 대칭 양자화의 격자 간격은 `scale`이다
(코드 q는 정수 격자, dequant = q·scale). Δ=2·scale이면 Δ²이 4배, bias가 4배.
- 영향: F2의 회복(+28.25/+29.46)은 **4배 과대 보정** 하에서의 결과. F1의 bias/logit
  1.55%/8.43%/75.9%도 4배 과대(실제 0.39%/2.1%/19%). 비트폭 간 상대 비교(INT2≫INT4)는 유지.
- 논문에서 빼야 할 문장: "보정은 논문 수식 그대로이며 아무것도 맞추지 않았다".
  §B8 결과(확정): 논문 사양 Δ에서도 회복은 유의(+10.40, t=3.23)하나 4배 bias의 37%뿐.
  4배 bias가 논문값보다 유의하게 낫다(t=−3.46). → 헤드라인 +28.25/"A4+보정≈LongLive"는
  4배-스케일 결과로 명시하고, 논문-사양 수치(+10.4)를 병기할 것. (큰 노이즈 구간에서
  2차 테일러가 bias를 과소평가한다는 해석과 정합 — 후속 연구 소재.)
- 수정 패치: `patches_audit/fix_tum_delta.patch`

### 버그 B — F2의 "Δ=0 경로 관문"이 공허 (C12)
관문 실행(f2gate)은 `--tum_correct`만 켜고 kv_quant는 BF16. `causal_model.py:343`의 분기는
`quantizer is not None and quant_state is not None`을 요구하므로 **명시적 attention 경로가
한 번도 실행되지 않은 채** 210/210 bit-exact가 나왔다. `tum_correction.verify_zero_delta`는
정의만 있고 어디서도 호출되지 않는다.
- 영향: F2의 차이(+28.25)에 "FA2 → 명시적 fp32 softmax 경로 교체" 효과가 섞여 있을 가능성이
  배제되지 않았었음. §B6(INT2 + 명시 경로 + Δ강제0)이 진짜 컨트롤.
- 논문에서 고쳐야 할 문장: 세션 4 §3 "관문: BF16(Δ=0)에서 보정을 켜도 210/210 bit-exact —
  경로 자체는 결과를 바꾸지 않으므로" → 이 관문은 검증력이 없었다. §B6 결과로 대체할 것.

### 버그 C — freeze의 SINK_TENSORS 프롬프트 간 누출 (C13 이상치의 원인, §7 해결)
`causal_model.py` 전역 `SINK_TENSORS`가 프롬프트 사이에 리셋되지 않는다. freeze 모드에서
프롬프트 0이 캡처한 sink가 프롬프트 1~9에 주입된다 — 의도치 않게 "다른 프롬프트의 chunk 0
override" 조작이 된 것.
- 증거: (1) 코드에 리셋 없음. (2) 저장 latent에서 freeze vs A2_int4의 chunk 8 rel_err가
  프롬프트 0만 0.041, 나머지 0.13~0.35. (3) §B2 프롬프트 5 단독 재실행 결과.
- 프롬프트 0조차 0이 아닌 이유: 양자화 블록(16토큰)이 sink 경계(4680 = 292.5블록)를
  걸쳐서, 경계 블록의 scale이 rolling 내용에 따라 변한다. freeze는 이 8토큰까지 고정하므로
  A2와 미세하게 달라지고 카오스가 증폭한다.
- 세션 4가 freeze를 결과에서 뺀 판단은 옳았다. C13의 다른 주장은 영향 없음.
- 수정 패치: `patches_audit/fix_sink_tensors_reset.patch`

### 버그 D — A1/A2 attn_mass JSON에 구버전 라벨 잔존 (C8 provenance)
세션 3이 "슬롯→chunk 라벨 버그를 고쳤다"고 했지만, 저장된 `attn_mass/A1_*`, `A2_bf16`
JSON은 수정 전 스크립트 산출물이다(chunk 7·14·20에서 slot0..5가 전부 0..5로 라벨).
mass 값 자체는 무영향(세션 4의 g_*와 bit-exact 일치)이고 보고서 표는 올바른 해석으로
작성되었으나, JSON을 그대로 쓰는 후속 분석은 라벨을 믿으면 안 된다.

## 2. §별 상세

### §2 측정 훅 부작용 (hook_side_effects.csv)
| 훅 | 검사 | 결과 |
|---|---|---|
| 패치 16 누적 상태(모든 훅 코드 포함, 비활성) | 현재 코드로 A1_int4 재실행 vs 세션 3 latent | **210/210 bit-exact** |
| KV_ERR_LOG ON/OFF | A1_int4 --kv_err_log 제거 재실행 vs 세션 3 | **210/210 bit-exact** |
| ATTN_MASS_HOOK | 코드 검사: in-place 없음, RNG 소비 없음, .float() 복사 연산만 | 통과(코드) + §B7 교차검증 |
| 16_tum Δ=0 | f2gate는 공허(버그 B). §B6로 대체 | **MUSIQ Δ −0.93, t=−1.53 (유의하지 않음) — 경로 무해 실증** |
| 15_sink 조작 게이트 | 퇴출 전 70/70 bit-exact, 후 140/140 상이 — latent 재검증 | **통과** |
| e2c INT2 멱등 no-op | 210/210 bit-exact — latent 재검증 | **통과** |
| noise_perturb eps=0 경로 | 코드: eps>0일 때만 곱함 | 통과(코드) |
| FA2 등가 검증(attn_mass) | 15개 실행 전부 같은 값 0.01392 — chunk 0에서 1회만 수행 | **주의**: 검증은 실행당 1회, 양자화 상태에선 미검증. 값 동일 자체는 결정성의 방증 |

### §3 attention mass 독립 재구현
- records → summary 재집계: A1·A2·f1·g 전 태그에서 mass_mean 오차 0, mass_sum=1.
- 세션 3(A1_*) vs 세션 4(g_*, f1_*) 동일 설정 재실행: **max|Δmass| = 0 (bit-exact)**.
- 핵심 수치 재현: chunk0 8.52%(보고 8.5), INT2 c20 self 19.21/최고령 29.78(보고 19.2/29.8),
  관문 G 표 전부 일치. 토큰 기준선 7×4680=32760 확인.
- raw Q/K 덤프 + full-softmax 독립 재계산: **max|Δmass| = 3.9e-6, 18/18 PASS** (§B7).
- 라벨: 독립 계산과 g_*/f1_* 일치, A1/A2 JSON은 버그 D.

### §4 negative controls (negative_controls.csv)
| 검사 | 기대 | 결과 |
|---|---|---|
| 교차 프롬프트 rel_err | 무상관(≥1.0), 진짜 짝과 분리 | **PASS** 가짜 1.042~1.561 vs 진짜 0.606~0.845, 겹침 없음 |
| chunk 0 진짜 짝 | 정확히 0 | **PASS** max 0.00e+00 |
| Identity quantizer(양자화 경로 배관 no-op) | BF16과 bit-exact | **PASS** 63/63 bit-exact |
| 대조군 seed 1·2 | c20 ≈ 0.74±0.05 | **PASS** 0.736 / 0.760 |
| TUM Δ=0 (INT2, 명시 경로) | A1_int2와 품질 동등 | **PASS** Δ −0.93, t=−1.53 |
| 노이즈 sink on BF16 | 양자화 무관하게 유해 | **PASS** Δ −3.67, t=−2.51 (유의) |
| (지시서의 16-bit no-op) | — | **수행 불가**: q가 int8 저장이라 16-bit는 원리적으로 no-op 불가(가드 2..8의 이유). Identity로 대체 — 더 강한 검사 |

### §5 프롬프트별·leave-one-out (per_prompt/per_prompt_audit.csv)
- 21개 핵심 비교의 평균·t 전부 보고서와 일치(±0.02 이내).
- 유의 주장 13건: 전부 LOO 견고(min|t| > 2.306). TUM INT2 A1은 10/10 부호 일치.
- 무효 주장 5건 중 2건 FRAGILE:
  - `INT2 A1→A5`(+2.42, t=2.05): 한 프롬프트 제외 시 t=2.97로 유의. "sink 6도 무효"
    주장은 약함 — 논문에는 "판단 유보"로 쓸 것.
  - `E2-e 노이즈 sink INT2`(+3.80, t=1.72): 프롬프트 2(−14.4) 제외 시 t=5.86.
    "INT2에서 노이즈 sink 무해" 방향의 서술은 피할 것(보고서는 이미 결론에 안 씀).
- 원 Δ값 10개(관문 A 표) 전부 재현.

### §6 MUSIQ·디코딩 파이프라인
- 프레임 인덱스: 결정적 함수, 모든 비교 대상 영상이 249프레임으로 동일 확인(ffprobe).
- 디코딩: inference와 재디코딩 모두 `decode_to_pixel(use_cache=False)`, VAE bf16 동일.
- pyiqa 0.1.16 단일 venv. (requirements-frozen.txt에는 pyiqa 부재 — 세션 1 시점 동결.
  세션 2~4 사이 버전 변경 증거는 없음.)
- 대조군 MUSIQ Δ: 10프롬프트 A1_ctrl +0.16(3프롬프트 세션 2 값 +4.3은 표본 차이).
  "카오스 이탈이 품질을 해치지 않는다"는 방향은 유지되나 +4.3을 인용하지 말 것.
- TF_w4a4 절대값 함정(c0=58.25) 재확인.

### §7 freeze 이상치 → 버그 C로 해결. 4개 세부 질문 답:
1. sink 슬롯 발산 시점: chunk 7(조작 시작)부터, 원인은 프롬프트 간 누출(주) + 경계 블록(부).
2. 비-sink 슬롯 off-by-one: 없음(`[:, :4680]` 정확). 단 블록 292가 경계를 걸침.
3. 노이즈 norm 매칭: **레이어별 전체 슬롯 하나의 norm**(head별 아님). −12.37의 방향은
   §B10(BF16 컨트롤)으로 뒷받침되지만, head별 크기 왜곡 가능성은 논문에 명시할 것.
4. E2-c INT2 오염: key와 value 둘 다 적용 확인(quantize_kv(k_slot, v_slot)).

### §8 양자화기 수학
- 수정 asym(offset) vs numpy 참조(클램프 없는 zp): rel_err 차 ≤ 7e-6 — 수학적 동일.
- 전부 양수/음수 그룹에서 asym 0.0013 vs sym 0.0400 — 수정 효과 재확인(C4).
- RTN(sym) 재양자화 멱등성: **bit-exact** (INT2/4, 4개 분포) — E1·E2-c 해석의 전제 성립.
- KIVI(asym)는 멱등 아님(max|Δ| 3e-2) — 세션들이 asym 멱등성에 기댄 곳은 없음. 단
  "RTN 계열 반올림이 거의 멱등"(세션 1 §8)을 KIVI에 일반화하지 말 것.
- KIVI 축: keys per-channel `(2,)`, values per-token `(3,4)` — 논문 사양 일치.
- zero-padding(4680→pad 8): 마지막 블록 오차 1.3배 — 전역 영향 무시 가능(8/4680 토큰).

### §9 프로파일링
- CUDA event 독립 재측정(§B9): attention 906→900ms 불변, 증가분 +1611ms 중 1604ms가
  quantize+dequantize. 3-chunk 비율 1.57× (세션 3: 1.59×). "+24%"는 63f 전체(VAE 포함) 기준.
- AI 수식 재계산: FLOP = 4·q·kv·d·H, 바이트 = 4·kv·d·H (BF16 K,V) → AI = q = 4680.
  4090 균형(165 TFLOPS bf16 / 1.008 TB/s ≈ 164) 대비 28배 — compute-bound 주장은 수식으로 성립.

### §1 provenance (provenance.csv)
- 63개 실행 stats.json 카탈로그: 전부 seed 0, torch 2.6.0+cu124 일관.
- 작업 트리 = Self-Forcing 33593df + 01 + 02 + 16(누적) 정확히 일치.
  kv-quant = b4c0936 + 08 + 17(내용) 일치.
- **주의**: 패치 시리즈는 README 순서로 기계 적용 불가 — causal_model.py 패치(03/09/15/16)는
  각각 pristine 기준 누적 diff라 최신 것 하나만 적용해야 하고, 17은 08과 훅이 겹침.
  재현 절차 문서에 명시할 것.
- 재현: A1_int4 210/210 bit-exact (§B1). 나머지 §B 참조.

## 3. §B — GPU 재실행 결과

- **B1 재현(세션 3 gateA)**: 현재 코드(패치 16 누적, 훅 비활성)로 원 커맨드 재실행 —
  A1_int4 **210/210 bit-exact**, A1_int2 **210/210 bit-exact**. C1·C9의 궤적이 오늘의
  코드 상태에서 정확히 재현되고, 세션 3 이후 추가된 패치(15·16)가 비활성 상태에서
  결과를 바꾸지 않음을 동시에 증명.
- **B2 freeze 프롬프트 5 단독**: 프롬프트 5만으로 e2d freeze 재실행(자기 sink 캡처) —
  A2_int4 대비 발산이 chunk 7에서 0.1731 → **0.0179 (10배 감소)**, chunk 8에서 0.2934 →
  0.0525. **SINK_TENSORS 프롬프트 간 누출(버그 C)이 freeze 이상치의 주원인임을 인과 확정.**
  잔여 발산은 양자화 블록(16토큰)이 sink 경계(4680토큰 = 292.5블록)를 걸치는 효과.
- **B3 KV_ERR_LOG OFF**: A1_int4에서 --kv_err_log 제거 재실행 — 세션 3(훅 ON)과
  **210/210 bit-exact**. in-situ 캐시 오차 계측은 결과에 영향 없음.
- **B4 Identity quantizer**: 원본 텐서를 그대로 저장·복원하는 passthrough quantizer로
  A1 실행 — BF16과 **63/63 bit-exact**. 양자화 경로 배관 전체가 투명함. (지시서의 16-bit
  no-op은 q가 int8 저장이라 원리적으로 불가 — 이 검사가 같은 취지의 더 강한 대체.)
- **B5 대조군 seed 1·2**: c20 rel_err seed1 0.736, seed2 0.760 (seed0 0.738) —
  카오스 바닥선 ±0.05 내 재현. **C2 통과.**
- **B1 보강**: A1_bf16도 **210/210 bit-exact** — gateA 3종 전부 재현 완료.
- **B6 TUM Δ=0(명시 경로 컨트롤)**: INT2 + 명시적 attention 경로 + Δ 강제 0, 10프롬프트 —
  A1_int2(FA2) 대비 MUSIQ Δ = **−0.93 ± 1.18, t=−1.53 (유의하지 않음)**. 경로 교체 자체는
  결과를 사실상 바꾸지 않는다. 버그 B로 공허했던 관문을 실증으로 대체 — C12의 인과 귀속
  ("차이는 보정 항") 성립.
- **B7 raw Q/K full-softmax 대조**: A1_bf16/A1_int2/A2_bf16 × chunk {6,20} × layer {0,15,29} —
  훅(online-softmax·GPU) vs 독립 full-softmax(CPU fp32) **max|Δmass| = 3.9e-6, 18/18 PASS**.
  슬롯→chunk 라벨 독립 재계산 일치, 캐시 = 정확히 7×4,680 토큰 확인. **C8·C9의 측정이
  독립 구현으로 검증됨.**
- **B8 TUM Δ=논문값(bias 1/4)**: 회복 **+10.40 ± 6.32 (t=+3.23, 부호 9/10)** — 논문 사양에서도
  유의한 회복. 그러나 4배 bias의 +28.25 대비 37%뿐이고, 4배가 논문값보다 유의하게 낫다
  (B8b: −17.85, t=−3.46). tum_zero(−0.93)와의 차이가 보정 항이 실제로 logit을 움직임을 증명
  (지시서 §4의 "보정 항이 실제 작동하는가" 취지 충족).
- **B9 프로파일링(CUDA event 독립 재측정, 3 chunk)**: BF16 2817ms → INT4 4427ms (1.57×,
  세션 3의 3-chunk 1.59×와 일치). attention 906→900ms **불변**. 증가분 +1611ms 중
  quantize 1261 + dequantize 343 = **1604ms가 재구성 비용, 잔여 +7ms**. AI 수식 재계산:
  AI = q = 4,680 ≫ 4090 균형 164 — compute-bound 성립. **C10 통과.**
- **B10 노이즈 sink on BF16**: A2_bf16 대비 **−3.67 ± 2.87 (t=−2.51, 유의)** — 노이즈 anchor는
  양자화 없이도 유해. C13의 "내용 생존" 해석과 정합(INT4 문맥의 −12.37보다 작음 — 캐시가
  오염될수록 앵커 의존이 커진다는 해석과도 정합).
- **B12 diag_act 재실행(C3)**: per-tensor 0.0263~0.0290 vs per-token 0.0257~0.0265,
  amax/med 1.2~4.6 — 세션 1 수치 재현. **C3 통과** (원 로그 부재는 provenance 결함으로만 기록).
- **B11 재현(세션 1 BF16 21f)**: **21/21 bit-exact** — 세션 1의 기준 궤적이 4개 세션 치의
  패치가 쌓인 현재 코드에서 그대로 재현.

## 4. 논문에서 빼거나 고쳐야 할 문장 목록

1. 세션 4 §3(F2): "보정은 양자화기 스케일만으로 유도했고 아무것도 맞추지 않았습니다" —
   **정정 필요**(버그 A: 실효 보정이 논문 값의 4배).
2. 세션 4 §3(F2): "관문: BF16(Δ=0)에서 210/210 bit-exact — 경로 자체는 결과를 바꾸지 않음" —
   **삭제/대체 필요**(버그 B: 공허한 관문). §B6 결과로 대체.
3. 세션 4 F1 표의 "bias/logit 1.55% / 8.43% / 75.87%" — 4배 축소 필요(0.39/2.1/19%).
   "INT2에서만 bias가 신호 크기에 접근"이라는 정성 결론은 유지.
4. 세션 3 §1(관문 A): "A5(sink 6)가 낫지 않음(INT2 +2.42 유의하지 않음)" — LOO 취약.
   "판단 유보(n=10에서 검정력 부족)"로 완화할 것.
5. 세션 2 §6 "대조군은 오히려 올라갑니다(+4.3)" — 3프롬프트 값. 10프롬프트에서는 +0.16.
   인용 시 10프롬프트 값으로.
6. 세션 1 §8 "RTN 계열 반올림이 거의 멱등" — RTN(sym)은 정확히 멱등(감사 확인),
   KIVI(asym)는 멱등 아님. "RTN(대칭)"으로 한정할 것.
7. 세션 4 §2(F2) "실무적으로 의미 있는 수치: A4+보정 −13.93으로 LongLive INT2에 근접.
   캐시 구조 변경과 논문 수식 한 줄이면 됩니다" — "논문 수식 한 줄"이 아니라 4배 스케일
   보정이었음(버그 A). 레시피 자체는 유효하나 서술 정정 필요.
8. 세션 3 §4 "attention은 memory-bound가 아니라 compute-bound" 및 재구성 비용 귀속 —
   감사 재측정으로 그대로 성립(수정 불요). 다만 "+24%"에는 63f 전체 기준임을 병기할 것.

## 5. 산출물 목록

```
audit/
  AUDIT_REPORT.md              이 문서
  provenance.csv               63 실행 × 설정
  per_prompt/audit_s5.py, per_prompt_audit.csv
  attn_mass_indep/recompute_from_records.py, qk_dump.py, indep_full_softmax.py, dumps/
  quantizer_math.py
  negative_cross_prompt.py
  compare_replication.py
  run_replication.sh, run_batch2.sh
  tum_zero_driver.py, tum_paper_delta_driver.py
  profile_recheck.py, profile_recheck.out
  patches_audit/               발견 버그 수정 패치
  rerun/                       재실행 산출물 (원본 미변경)
```
