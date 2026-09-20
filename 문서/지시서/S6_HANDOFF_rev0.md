# 세션 6 작업 지시서 — 격자 가설 검증, 스칼라 보정 정식화, 일반화(LongLive·SkyReels-V2), 논문용 그림

작성: 2026-09-18 · 선행: `RESULTS_session5.md`, `RESULTS_session5_gates.md`, `NOTES_session5.md` · 현황: `STATUS_2026-09-18.md`
환경: 세션 5와 동일 4090 24GB. 기준 커밋: Self-Forcing `f2e831e`(패치 22~25 포함), kv-quant `650ce90`. 프롬프트 10개 고정. **1차 지표 = 21 chunk 평균 MUSIQ**, 낙폭(c20−c0)은 병기. **모든 핵심 표에 VBench 3차원(imaging quality · subject consistency · temporal flickering)을 병기**한다 — 이 분야 표준이며, 세션 5에서 λ=1~4가 attention 분포는 크게 다른데 MUSIQ가 동률이었으므로 MUSIQ 단독으로는 부족하다. TF-LPIPS도 병기. 모든 비교 짝지은 t, 무효 주장은 효과 크기 CI 병기.

이 세션의 목적은 세션 5가 만든 기여 넷(이해·정정·레시피·커널)을 **논문으로 세울 수 있게** 하는 것이다. 새 발견보다 **검증과 일반화**가 우선이다. 순서: 격자 가설(우리 결론 두 개가 걸려 있음) → 스칼라 보정 정식화 + 왜 스칼라로 충분한지 유도(방법 1) → **보정이 못 잡는 66%의 정체와 그것을 잡는 것(관문 O — 이 연구의 원래 문제 제기)** → LongLive·SkyReels-V2 반복(일반화) → 그림 생성.

---

## 0. 시작 확인
- `git log` 두 저장소 해시 일치, 플래그 전부 OFF에서 A1_int4 21f×3p가 세션 3 latent와 bit-exact(회귀 관문, 매 패치 후 반복).
- 실행당 소요 기준: FA2 경로 63f×10p ≈ 56분, 명시(보정) 경로 ≈ 60~70분, TF ≈ 70분. GPU 예산 **약 32시간**(K 8, L 8, O 9, M 8 — 이틀 밤). 배치는 nohup, 폴링 최소화.

---

## 1. 관문 K — 격자 어긋남(lineage) 가설 (최우선, 결론 두 개가 걸려 있다)

**가설.** chunk 4,680 토큰 = 16토큰 블록 292.5개. 모델의 rolling은 퇴출마다 캐시를 8토큰 밀린 격자에서 재양자화하고, 밀린 격자에서 RTN은 멱등이 아니므로 오차가 매 퇴출마다 쌓인다. 이것이 (i) "INT2에서 sink 무효"(관문 A: A2≈A1)와 (ii) 앵커 오라클의 "정책 효과 +11.2"(선택이 A2와 같은데도)를 만든 아티팩트일 수 있다. 세션 5의 lineage 효과(driver-oldest − A1: INT4 +1.29, INT2 +2.81)가 그 하한.

### K1. bit 단위 원인 확정 (GPU 20분)
driver-oracle과 driver-oldest(둘 다 INT2, 프롬프트 1개)에서 chunk 7·14·20의 캐시 K/V를 덤프. 앵커 선택이 {0,1,2}로 같은 구간에서 **어느 슬롯·어느 토큰 블록이 다른지** 표. 예측: 차이는 sink 슬롯의 경계 블록(292번째 블록의 8토큰)과 퇴출 후 재양자화된 슬롯에 집중. 차이가 예측 밖에 있으면 정책 효과는 다른 원인(RoPE 위치? 순서?) — 그 경우 여기서 멈추고 보고.

### K2. 격자 정렬 모델 경로 (핵심 실험)
모델 경로(`ucsd_dequant` 백엔드)에 **격자 정렬** 옵션 추가. 두 방식 중 하나(둘 다 되면 둘 다):
- (α) 패딩: chunk를 4,688(=293 블록)로 패딩해 블록 경계가 chunk 경계와 일치 → 퇴출이 블록 단위로 일어나 격자가 안 밀림.
- (β) 세그먼트 양자화: 캐시를 chunk 세그먼트 단위로 저장하고 퇴출은 세그먼트 단위, 재양자화 없음(fused 경로와 같은 성질, dequant 백엔드에 이식).
관문: BF16에서 A1과 bit-exact(정렬 옵션이 BF16 경로를 바꾸면 안 됨). INT4에서 정렬 ON이 세션 5 driver-oldest와 **MUSIQ ±0.3 이내**(같은 성질이면 같은 값이 나와야 함).

### K3. 걸려 있는 결론 재측정 (정렬 ON, 10프롬프트)
| 실행 | 답하는 질문 |
|---|---|
| A1 INT2, A2 INT2 (정렬 ON) | **sink가 INT2에서 살아나는가** — A2−A1이 세션 5 오라클의 +11~15 수준이면 "INT2 sink 무효"는 아티팩트였음 → C7·C13 정정 |
| A1 INT4, A2 INT4, A4 INT4 (정렬 ON) | INT4 결론(구조 회복)이 정렬 후에도 유지되는지, 크기 변화 |
| A1 INT2 정렬 ON, **TF 모드** | 전파 5:1·2:1에서 lineage 성분 분리 — 정렬 ON FR − TF가 새 "순수 전파", 정렬 OFF와의 차가 lineage 몫 |
| A4 + λ=1 + K3V2 (정렬 ON) | 레시피 최종 수치 갱신(세션 5 −1.11이 얼마나 좋아지는가) |
- 판정: 정렬 ON에서 A2 INT2 − A1 INT2가 유의하면 격자 정렬은 **공짜 개선이자 새 발견**(논문 A §격자, 논문 B 설계 근거). 유의하지 않으면 오라클 +11.2는 다른 원인 — K1 표를 근거로 NOTES에 가설.
- 예산: 약 8실행 ≈ 8h.

---

## 2. 관문 L — 스칼라 캐시 보정 정식화 (방법 1)

세션 5 flatten: per-key 구조를 지우고 캐시 전체를 스칼라로 낮춰도 taylor와 동일(+10.49 vs +10.04). 이를 **파라미터 1개 방법**으로 정식화한다.

### L1. 스칼라 c 직접 sweep
캐시 logit에서 상수 c를 뺀다(self 제외). c를 Δ에서 유도하지 않고 직접: c ∈ {0.05, 0.1, 0.2, 0.4, 0.8, 1.6} (logit 단위; flatten λ=1의 실효 스칼라 값을 먼저 기록해 그 근방을 포함시킬 것). A1 INT2 FR, 10프롬프트. INT4에서 c 3점.
- 출력: c vs 평균 MUSIQ, c vs self mass. c*(INT2), c*(INT4). c*가 비트폭에 따라 어떻게 변하는지 — Δ² 비례(TUM 예측)인지 확인. 비례하면 "c = κ·Δ̄²"로 캘리브레이션 없는 규칙, 아니면 비트폭별 상수 1개.
### L2. TF에서 c*
TF 모드 c 3점(c*_FR의 0.5·1·2배). c*_TF ≈ c*_FR/2면 세션 5의 "전파가 실효 노이즈 2배" 해석 확정.
### L3. 왜 스칼라로 충분한가 — 집중(concentration) 유도 (GPU 불필요, 반나절)
TUM bias b_i = (1/24d) Σ_c q_c² Δ_ic². d=128 채널의 합이므로 key마다의 b_i는 평균 주위에 집중할 것이다. 유도할 것:
- q_c²Δ_ic²를 채널별 독립 항으로 보고 b_i의 평균 μ_b와 표준편차 σ_b를 식으로 쓴다. σ_b/μ_b가 채널 수 d와 Δ의 채널 간 변동계수(CV)로 어떻게 정해지는지(대략 CV/√d 꼴).
- **실측 대입**: 세션 5 gateH_eps와 attn 덤프의 실제 q, Δ로 b_i 분포를 슬롯·레이어별로 계산해 σ_b/μ_b 히스토그램. 예측: 0.1 이하.
- 그 σ_b가 softmax를 통과했을 때 attention 재배분에 미치는 크기(1차 근사: mass 변화 ∝ p_i·(b_i − μ_b))를 계산해, 실측 flatten−taylor 차이(+0.45, n.s.)와 같은 자릿수인지 확인.
- 결과: "채널 수가 충분히 크면 per-key Jensen 보정은 스칼라로 수렴한다"를 정리(proposition)로 쓸 수 있는지, 조건(d, CV)은 무엇인지. 이것이 되면 관문 L은 경험적 관찰이 아니라 **유도된 단순화**가 된다.

### L4. 구조 성분 무효의 일반성
flatten vs taylor 비교를 **A4 INT2**와 **K3V2**에서 1회씩 반복(세션 5는 A1 INT2만). 셋 모두 차이 n.s.면 "구조 기여 0"을 일반 진술로.
- 예산: 약 10실행 ≈ 10h → 시간 부족 시 L1을 4점, L4를 A4만.

---

## 2b. 관문 O — 보정이 못 잡는 66%: 정체와 해법 (이 연구의 원래 문제 제기)

세션 5: λ=1 보정은 INT2 피해의 34%를 회복하고 attention 분포를 되돌리지만, 그 이상은 못 한다. 세션 5의 배분에서 (a)(c)는 기각됐고 (b) 전파 오염만 남았다. 보정은 attention **배분**을 고치는 도구이고, 캐시 **내용**이 이미 틀어진 것은 고칠 수 없다. 남은 66%가 그 내용 오염의 몫이라는 가설을 세우고, 내용을 다루는 세 후보가 그 잔여를 각각 얼마나 깎는지 잰다.

### O1. 잔여의 정체 — 보정 후 TF/FR 분해
A1 INT2 + λ=1(self 제외)에서 TF와 FR을 둘 다 돌린다(TF는 세션 5 H1 TF λ=1 재사용). 보정 후 TF 손실 = 보정이 못 잡는 **주입** 몫, 보정 후 FR − TF = 보정이 못 잡는 **전파(내용 오염)** 몫. 예측: 잔여의 대부분이 FR−TF. 주입 몫이 크면 가설 수정(양자화 자체가 내용을 망가뜨리는 몫 — key 비트로만 해결).
### O2. 후보별 잔여 감소 (전부 λ=1 보정 위에서, A1 또는 A2 INT2, 10프롬프트)
| 후보 | 잡으려는 것 | 실행 |
|---|---|---|
| 격자 정렬(K2) | 재양자화 lineage 오염 | 보정 + 정렬 ON |
| 앵커 관리(세션 5 오라클, 정렬 ON에서 재측정) | 덜 오염된 내용 유지 | 보정 + 오라클 앵커 |
| key 비트 (K3V2) | 오염 발생량 자체 | 보정 + K3V2 |
| 전부 | 가산성 | 보정 + 정렬 + 앵커 + K3V2 |
- 각 행에서 O1처럼 TF/FR을 분해해 **어느 몫이 줄었는지** 기록(정렬은 전파 몫만, key 비트는 주입·전파 둘 다 줄어야 함 — 예측과 맞는지).
- 출력: 막대 하나 — BF16 대비 손실을 [보정이 잡은 몫 | 정렬이 잡은 몫 | 앵커가 잡은 몫 | key 비트가 잡은 몫 | 남은 몫]으로 쌓기. 이 그림이 논문 A의 결론 그림이다.
- 판정: 남은 몫이 LongLive INT2(재학습, 세션 3 −16.9 낙폭 기준 → 평균 지표로 재측정 필요, M1에서) 이하면 "재학습 없이 재학습 수준" 주장 성립.
- 예산: 약 8실행(TF 포함) ≈ 9h. 시간 부족 시 "전부" 행과 K3V2 행 우선.

## 3. 관문 M — 일반화 (모델 2·3)

### M1. LongLive v1.0 (같은 코드베이스, w9s3 재학습)
핵심 표 반복, 10프롬프트, 정렬 ON/OFF 중 K3 결과가 더 나은 쪽 하나로:
- BF16 / INT4 / INT2 / K4V2 / K2V4 (key 단독 원인 재확인)
- INT2 + λ=1 (보정이 재학습 모델에서도 듣는가)
- TF INT4·INT2 (전파:주입 비율이 재학습으로 어떻게 바뀌는가 — 예측: 전파 몫 감소)
- attention mass BF16·INT2 chunk 20 (역전이 재학습 모델에서도 일어나는가)
- 약 8실행 ≈ 8h.
### M2. SkyReels-V2 1.3B (TUM 코드베이스, 시간 남으면)
TUM 저장소의 SkyReels-V2 설정으로 BF16 / RTN INT2 / INT2 + TUM(λ=1) / INT2 + flatten 4실행, 그들 지표(VBench IQ) + MUSIQ. 목적 하나: **flatten = taylor가 그들 모델·그들 양자화기(asym per-token g=32)에서도 성립하는가.** 성립하면 "구조 기여 0"이 TUM 저자 반박에 견딘다. 코드 이식이 3시간을 넘으면 중단하고 NOTES에 진행 상황.

---

## 4. 관문 N — 논문용 그림 (GPU 불필요, 배치 도는 동안)

기존 CSV/JSON에서 벡터 PDF(+PNG)로, 폰트 9pt, 단일 열 폭 3.3in 기준:
1. `fig_prop_vs_inj.pdf` — TF/FR 막대(INT4·INT2), 전파 몫 표시. K3 결과 나오면 lineage 성분 분리 버전 추가.
2. `fig_attn_profile.pdf` — chunk 0~20 self/최근/최고령 곡선: BF16·INT4·INT2·K4V2·K2V4.
3. `fig_correction.pdf` — λ sweep(FR·TF 평균 MUSIQ) + self mass, flatten 점 표시.
4. `fig_gateA.pdf` — 캐시 구조 × 비트폭 격자(INT4/INT2 평균 MUSIQ), LongLive 점.
5. `fig_pareto.pdf` — 관문 I: x=30레이어 KV GB, y=Δ MUSIQ, A1 무보정 vs A4+보정, 전선 표시.
6. `fig_long126.pdf` — 126프레임 per-chunk MUSIQ 4곡선.
7. `fig_kernel.pdf` — 메모리 4막대(BF16/int8/INT4/INT2 패킹) + 종단 시간 3막대.
8. `fig_sink_content.pdf` — E2 조작(자기/타 프롬프트/INT2 노이즈/가우시안) 막대.
각 그림의 데이터 소스 파일과 생성 스크립트를 `figs/README.md`에.

---

## 5. 규칙
- 결과는 `session6-*` 커밋 기준. 새 플래그 OFF 회귀 관문 매 패치 후.
- 격자 정렬 옵션은 BF16 bit-exact 관문을 통과한 뒤에만 사용.
- 새 파라미터(c)는 10프롬프트 평균으로 고르고 다른 설정에서는 고정(프롬프트 맞춤 금지). c*를 정한 실행과 검증 실행을 구분 표기.
- 시간 부족 시 자르는 순서: M2 → L4 → L2 → L1 축소(4점) → O2 앵커 행 → M1의 attention mass. K·O1·O2 "전부"행·L3·N은 유지.
- 등가 관문 미통과 경로의 품질 수치는 보고하지 않는다.

## 6. 산출물
```
results/
  gateK_bitdiff.md                 K1: 슬롯·블록별 차이 표
  gateK_aligned.csv                K3: 정렬 ON/OFF × 설정 × 평균·낙폭·t
  gateK_propagation_split.csv      전파 = 순수 전파 + lineage 분해
  gateL_scalar_sweep.csv/.png      c × {FR, TF} × 평균 MUSIQ·self mass, c*(INT2/INT4), Δ² 비례 여부
  gateL_flatten_generality.csv     flatten vs taylor × {A1 INT2, A4 INT2, K3V2}
  gateL_concentration.md/.png      L3 유도 + 실측 σ_b/μ_b 히스토그램
  gateO_residual.csv/.png          O1 보정 후 TF/FR 분해, O2 후보별 잔여 감소 쌓인 막대
  vbench/                          모든 핵심 표의 VBench 3차원 + TF-LPIPS
  gateM_longlive.csv               M1 표
  gateM_skyreels.csv               M2 (있으면)
  figs/*.pdf|png, figs/README.md   N
  NOTES_session6.md  (1) 격자 가설 판정과 정정할 결론 목록 (2) c* 규칙 (3) 일반화 표 요약 (4) GPU 시간 (5) 미결
patches/  26_grid_align.patch, 27_scalar_bias.patch, 28_longlive_gates.patch, 29_vbench_eval.patch
```

## 7. 우선순위
K1 → K2 → K3 → L1 → L3(병행, GPU 불필요) → N(병행) → O1 → O2 → M1 → L2 → L4 → M2.

## 8. 이 세션이 답해야 하는 질문
- "INT2에서 sink 무효"와 "앵커 정책 +11.2"는 격자 어긋남 아티팩트인가? 격자 정렬이 공짜 개선인가? 전파 5:1에서 lineage 몫은 얼마인가?
- 캐시 logit 스칼라 하나(c)로 TUM 보정을 대체할 수 있는가, c는 Δ²에 비례하는가(캘리브레이션 없는 규칙), TF에서 절반인가? 왜 스칼라로 충분한지 채널 수 집중으로 유도되는가?
- 보정이 못 잡는 66%는 내용 오염(전파)인가 주입인가? 격자 정렬·앵커·key 비트가 그 잔여를 각각 얼마나, 어느 몫에서 깎는가? 전부 합치면 재학습 LongLive INT2 수준에 닿는가?
- key 단독 원인·attention 역전·보정 효과·전파 비율이 LongLive(재학습)에서도 성립하는가? flatten = taylor가 SkyReels-V2·TUM 양자화기에서도 성립하는가?
