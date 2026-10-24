# S9 — 균일 PSAC8 참고 백엔드 품질 실험: 사전 등록 (실행 전 고정, 2026-10-25 00:50 KST)

근거: [S9_cand1_diag.md](../결과서/S9_cand1_diag.md) §5.5 7 (6차 리뷰 합의) · [r2.1 지시서](HANDOFF_method_candidates_2026-10-24_r2.md) §0.1·§3.6.
**목적(한정).** 10% 예산 안의 유일한 고정 후보 "균일 PSAC8 + 잔여 대각 분산 보정" 이 **영상 품질에서 K4V2 λ1 에 비열등한가**를, fused 커널 개발 **전에** 복원 경로의 참고 백엔드로 판별한다. 속도·peak 우위는 이 실험으로 주장하지 않는다. 실패하면 후보 1 의 구현 투자를 종료한다.

## 1. 고정 사항

| 항목 | 고정값 |
|---|---|
| 후보 | **PSAC8**: K RTN INT2(chunk-local group 16) + chunk 확정 시 **자기 query 2차 모멘트**의 상위 8 고유벡터(fp16)에 잔차 투영, 계수 int8·열 scale fp16(chunk 별), 잔여 대각 분산 fp16(head 당 128); V RTN INT2; 보정 b = (1/2d)Σ q_c² Var_c (segment 상수). 추가 선택 없음 |
| 비교 팔 (같은 복원·명시적 attention 경로) | **INT2 λ1**: K2 V2, rank 0, Δ=scale Taylor λ1 · **K4V2 λ1**: K4 V2, rank 0, Δ=scale Taylor λ1 · **BF16**: 양자화 없음(기본 FA2 경로) |
| 경로 통제 | 양자화 세 팔은 모두 `--attn_backend psac` 의 **같은 코드 경로**(복원 K̂·V̂ + fp32 명시적 softmax). BF16 만 기본 경로 — 그 차이는 관문 G1 로 크기를 잰다 |
| 프롬프트 | `prompts/quant3/prompts_eval_psac10.txt` 10개. VBench `all_dimension.txt` 에서 **기존 8개 파일(20 문장)과 중복 0** 인 936 문장을 seed 0 으로 섞어 앞 10개. 어떤 실행에도 쓰인 적 없음 |
| 생성 | seed 0 · 63 프레임(21 chunk) · `configs/gateA_A1.yaml`(sink 0, 창 21f) · EMA · 디코드 포함 · 진단 훅 OFF |
| 평가 | MUSIQ(`metrics_all.py`, 프롬프트별 chunk 평균) · VBench `subject_consistency`·`temporal_flickering`(`vbench_dims.py`, CPU) · **움직임**: VBench `dynamic_degree`(RAFT 가중치 다운로드 가능 시) **또는** 대체 지표 = 디코드 프레임의 인접 프레임 평균 절대차(`tools/motion_proxy.py`). 어느 쪽을 썼는지 기록 |
| 통계 | 프롬프트 짝지은 차이, t(9,.975) CI, 양수 개수, 프롬프트별 값 |

## 2. 판정 규칙 (결과를 보기 전에 고정)

**비열등성 (PSAC8 − K4V2 λ1, 95% CI 하한):**

| 지표 | 한계 | 근거 |
|---|---|---|
| subject_consistency | **≥ −0.010** | K4V2 λ1 자체의 BF16 대비 잔여(−0.011 [−0.024, +0.001], S8 계획 §4)와 같은 크기 |
| MUSIQ | **≥ −1.0** | 설계 A·K4V2 의 BF16 대비 CI 폭(≈±1) |
| temporal_flickering | **≥ −0.003** | S8 계획 §4 의 K4V2 flicker CI 폭 |
| 움직임 | dynamic_degree: PSAC8 의 동적 판정 영상 수 ≥ K4V2 − 1 / 대체 지표: PSAC8/K4V2 비 ≥ 0.9 인 프롬프트 ≥ 8/10 | 정적 영상으로 일관성만 올리는 경우 차단 |

**통과** = 네 항목 모두 만족. 하나라도 어긋나면 **실패** — 후보 1 의 fused 구현 투자 종료, 후보 2 로 이동.

**선행 조건(sanity):** PSAC8 − INT2 λ1 이 subject_consistency 또는 MUSIQ 에서 CI 가 0 을 제외하고 양수여야 한다. 아니면 "INT2 λ1 의 개선조차 비검출" 로 **실패**(S9 §5 의 attention 오차 개선이 품질로 옮겨지지 않았다는 뜻).

**해석 규칙.** 통과해도 "K4V2 와 동등" 이 아니라 "사전 한계 안에서 비열등" 이다. 통과 시 다음은 (i) 실제 packed 저장량 ≥ 10% 절감 확인, (ii) peak·시간, (iii) fused 투자 판단이며 이 실험이 그것을 대신하지 않는다. 결과를 본 뒤 한계를 옮기지 않는다. 이 10개 프롬프트는 이후 개발 표본이 된다.

## 3. 관문 (본 실험 전)

| 관문 | 내용 | 기준 |
|---|---|---|
| G1 경로 | `--attn_backend psac` + `PSAC_NOQUANT=1`(세그먼트에 BF16 K/V 그대로, 보정 없음) 1 프롬프트 63f 의 latent 를 기본 BF16 경로와 비교 | chunk 별 상대오차 **≤ 1e−2**(fp32 명시적 softmax vs FA2 bf16 의 수치 차이 허용; S5 Gate B 와 같은 성격). 초과하면 세그먼트 관리나 마스크 오류 |
| G2 chunk 0 | 양자화 세 팔의 chunk 0 latent | **bit-exact 일치**(전부 self, 캐시 미사용) |
| G3 상태 | 각 chunk 확정 후 세그먼트 수·chunk_id·창 용량(7 slots − self) · 프롬프트 리셋 시 `psac` 상태 pop | 세그먼트 ≤ 6, chunk_id 연속, 리셋 후 부재 |
| G4 바이트 | PSAC8 의 세그먼트 구성 요소(코드·scale·U·C·Cs·var) 형상·dtype 기록 → 해석식 바이트(head·chunk 당 K 264,528 B) 와 대조 | 일치. **단 참고 백엔드는 코드를 int8 로 들고 있어 resident_bytes 는 packed 상주가 아니다** — 보고는 해석식으로, 실측 상주 판정은 fused 단계로 미룬다 |

## 4. 시간 상한

- 구현 + 관문: **6 시간** (2026-10-25 07:00 KST 까지). 초과 시 중단하고 상태를 기록.
- 전체(생성 4 팔 × 10 프롬프트 + 지표): **24 시간** (2026-10-26 01:00 KST). 명시적 경로는 fused 보다 느리다(S5: TUM explicit 63f×10p ≈ 48 분/팔 기준).

## 5. 소스·산출물

- Self-Forcing `5741aef` + 미커밋 diff(이 실험에서 추가: `causal_model.py` psac 백엔드, `inference.py` 플래그, `causal_inference.py` 리셋 pop). 실행 시 `kv_bytes_report` 에 diff sha256 기록.
- 드라이버 `run_s9_psac_quality.sh`, 결과 `Self-Forcing/results/session9/psac_quality/`, 분석 `Quantization/tools/analyse_psac_quality.py`, 결과서 S9 §6 에 추가.

## 6. 프로토콜 변경 기록 (본 실험 시작 전, 2026-10-25 01:20 KST)

- **G1 원안은 실패했다.** NOQUANT(세그먼트 BF16 그대로, 명시적 fp32 softmax) vs 기본 BF16(FA2) 의 rollout latent 상대오차가 chunk 0 에서 0, 이후 단조 증가해 chunk 20 에서 **6.2e−2** (기준 1e−2 초과). 이것은 rollout 에 걸친 수치 차이의 누적이며(README 의 카오스 바닥선 0.74 보다 한 자릿수 아래), 한 호출의 경로 오류와 구분되지 않는다. 원안 기준은 rollout 누적에 맞지 않게 정한 것이었다.
- **대체 관문 G1′**: 같은 입력에서 **한 호출**의 명시적 경로 출력 vs FA2 출력 상대오차 ≤ 1e−2 (4 지점: chunk 1 층 0, chunk 6 층 15, chunk 10·20 층 29). S5 Gate B 와 같은 성격. 원안 수치도 기록으로 남긴다.
- 이 변경은 **품질 결과를 보기 전**에, 관문 도구의 오류 수정과 함께 이뤄졌다. 판정 규칙 §2 는 바뀌지 않았다.
- 관문 1차 실행에서 발견한 도구 오류 둘: inference.py 의 psac 상태 기록 블록이 바이트 보고의 출력문을 삼켜 NOQUANT 팔이 종료 시 NameError 로 중단(생성·latent 는 완료) → 블록 위치 수정; 관문 스크립트의 형상 비교가 JSON list 와 tuple 을 비교해 G4 를 오판 → 수정. G4 의 해석식 바이트는 일치했다(3,174,336 B = 12 × 264,528).
- 속도(1 프롬프트 63f, 디코드 포함): BF16 51 s · PSAC8 282 s · INT2 λ1 346 s · K4V2 λ1 346 s (delta 모드는 per-key Δ² bias 계산이 추가). 본 실험 4 팔 × 10 프롬프트 ≈ 2.9 h + 지표 ≈ 1 h — 24 h 상한 안.
