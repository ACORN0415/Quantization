# GPU 서버 Codex 인수인계 — 연구 맥락과 독립 리뷰

작성: 2026-09-19. 사용자가 서버 Codex에 전달하기 위해 요청한 단일 인수인계 파일.

## 0. 범위와 사용 방법

이 문서는 9월 18일에 읽은 세션 1~6 자료, 주요 원자료 재계산, 일부 코드 확인, 그리고 HANDOFF_session6_addendum.md에 대한 후속 검토를 합친 것이다. **9월 19일 현재 서버 상태를 새로 확인한 보고서는 아니다.** 아래 대기·미구현·잠정 표시는 읽은 시점 기준이다. 이후 완료된 작업은 최신 코드·로그·원자료로 확인하고 중복 실행하지 말 것.

사용자의 직전 요청은 인수인계 파일 작성이다. 전달 자체가 GPU 실험 시작, 진행 중 배치 변경, 코드 수정, 논문 게시까지 새로 승인하는 것은 아니다. 실제 후속 작업은 서버 세션에서 사용자가 맡긴 범위 안에서 수행하라. 문서의 과거 지시·예측과 완료된 실험 결과를 구분하라.

리뷰 과정에서는 서버 원본과 실행 중 실험을 변경하지 않았다. 전체 코드 감사나 GPU 재실험을 수행하지 않았다. 주요 프롬프트별 JSON에서 평균·paired t·t 기반 CI·leave-one-out을 독립 재계산했고 오라클·보정·통계·패킹·저장 코드 일부를 대조했다.

## 1. 연구 목적

스트리밍 비디오 diffusion에서 과거 chunk의 KV 캐시를 저비트로 압축할 때 장기 생성 품질이 왜 나빠지는지 분석하고, 개선 조합을 실제 packed 캐시와 fused attention 구현까지 연결한다.

- 모델: Self-Forcing, LongLive v1.0. 주로 Wan2.1 1.3B 계열.
- 장비: 주로 RTX 4090 24GB. 커널 트랙에 RTX 3080 개발 계획도 존재.
- 일반 실험: MovieGenBench 앞 10개 prompt, seed 0, 63 latent frames = 21 chunks.
- chunk 하나: latent frame 3개, 토큰 4,680개. block size 16이면 292.5 blocks.
- 주 지표: chunk 평균 MUSIQ. VBench subject consistency·temporal flickering 등 보완.
- 단위 주의: latent frame과 pixel frame은 다르다. 기존 감사의 63 latent frame 출력은 249 pixel frames였다.
- 측정 도구: BF16 미세 섭동 대조군, TF/FR 대조, attention mass, cache 오차 진단.

현재 가장 방어력 있는 중심 주장:

> 저비트 KV의 품질은 비트폭뿐 아니라 저장·재양자화 이력에 좌우된다. 이 이력을 통제하면 sink와 캐시 구조의 효과 및 기존 메커니즘 해석이 달라진다.

‘전파가 유일 원인’, ‘상수 하나로 충분’, ‘8배 압축·BF16 품질·속도를 한 구현에서 달성’은 검증 전 헤드라인으로 쓰지 말 것.

## 2. 먼저 읽을 서버 자료

루트: /home/devel/gpu (devel@172.20.10.148, 별칭 gpu-node).

- RESULTS_4090_session.md, RESULTS_session2.md, RESULTS_session3.md, RESULTS_session4.md
- RESULTS_session5.md, RESULTS_session5_gates.md
- audit/AUDIT_REPORT.md
- Self-Forcing/results/session6/RESULTS_session6.md
- Self-Forcing/results/session6/NOTES_session6.md
- Self-Forcing/results/session6/STATUS_LIVE.md
- Self-Forcing/results/session6/gateK_bitdiff.md
- Self-Forcing/results/session6/gateL_concentration.md
- Self-Forcing/results/session6/storage_semantics.md
- Self-Forcing/results/*_per_prompt.json 및 results/session6/*_per_prompt.json

첨부 자료 CODEX_BRIEF.md, STATUS_2026-09-18.md, THESIS_by_section.md, HANDOFF_session6.md, HANDOFF_kernel_track.md, rw_D_novelty_check.md, HANDOFF_session6_addendum.md의 핵심은 아래에 통합했다. 첨부 파일들이 서버에도 존재한다고 가정하지 말 것.

구문서는 버그 수정 전 결론을 포함한다. 특히 보정의 +28.25, 구 +10.40, 4배 보정 우위는 bug E 전 수치로 현재 주장에 재사용하지 말 것.

## 3. 확인된 주요 문제와 해석

### 3.1 오라클과 A2의 앵커 수가 다르다 — 최우선

Self-Forcing/oracle_anchor.py의 n_anchors=3은 **3 chunks = 9 latent frames**다. results/gateH_anchor/oracle_int2/oracle_log.json도 n_anchors=3과 {0,1,2} 유지를 기록한다.

반면 configs/gateA_A2.yaml의 sink_size: 3은 **3 latent frames = 1 chunk**다. 모델은 sink_tokens = sink_size * frame_seqlen으로 계산한다.

따라서 ‘같은 chunk를 유지하는데 +11.2가 남으므로 재구성이 원인’이라는 전제가 틀렸다. +11.20은 oracle−driver-oldest, +15.55는 oracle−A2로 기준선도 다르다.

필요한 대조: 같은 driver·격자·RoPE·총 용량에서 oldest / fixed{0} / fixed{0,1,2} / oracle-3. 총 용량 고정에서는 앵커 증가와 최근 문맥 감소가 동시에 일어나므로 **앵커와 최근 문맥의 배분 효과**라고 부른다.

선택 이력이 완전히 같다면 해당 실행의 oracle은 고정 정책과 같은 판단을 했다. 품질의 비유의만으로 적응 선택의 기여0을 단정하지 않는다.

### 3.2 비유의와 동등성은 다르다

flatten−Taylor: +0.448, t=0.617, 95% CI [−1.194, 2.091], 90% CI [−0.883, 1.779]. ±1 MUSIQ TOST 동등성 기준을 통과하지 못한다.

FR λ4−λ1: +2.078, t=0.685, 95% CI [−4.782, 8.938]. λ1은 이론 기본값으로 채택할 수 있지만 최적점 확정은 아니다.

TOST 허용폭은 실용적 근거를 설명하고 확인용 평가 전에 정한다. n=10이면 95% CI는 2.262×SE, 90% CI는 1.833×SE지만 다른 표본수에 그대로 적용하지 말 것. MUSIQ 동등성은 영상 품질 전체의 동등성이 아니다.

### 3.3 8배·품질·속도는 아직 같은 구현의 결과가 아니다

확인 시점 Self-Forcing/kernels/triton_kv_quant.py는 assert bits in (2, 4)를 사용했다. K3V2의 1344 B/token·0.75GB는 이 커널에서 실측한 값이 아니다. 이후 K3 구현이 완료됐다면 코드와 실제 실행 증거로 갱신한다.

8배 = 토큰당 약4.57배 × window 21→12의1.75배다. 문맥 축소를 포함한 전체 캐시 감소이며 토큰당8배가 아니다.

INT4 fused의51.3초·9.84GB를 K3V2 품질 행에 결합하지 말 것. 70.1→51.3초는 기존 INT4 경로 대비이며 pristine BF16 대비가 아니다. 8.17ms/6.9ms−1은 **18.4%**로 문서의13.1%와 불일치한다.

우선 A4+K4V2+λ1에 정렬 상태·보정 형식·backend까지 고정하고 품질·실제 bytes·시간을 연결한다. 성능 측정에서는 진단 hook을 끄고 warmup·동기화·VAE 포함 여부·다른 GPU 작업 여부를 기록한다.

### 3.4 구 TF/FR 5:1·2:1은 n=3

구 tf_musiq_per_prompt.json은3 prompts다.

- BF16 drop +3.727, TF INT4 +1.098, FR INT4 −11.028 → 주입2.629, TF−FR12.125, 비율4.61:1.
- INT2는 주입14.564, TF−FR31.997, 비율2.20:1.

TF와 FR는 history 분포가 달라 history drift·query 변화·현재 양자화와의 상호작용·저장 이력이 함께 변한다. MUSIQ의 산술적 차이가 독립적인 물리 오차의 인과 분해가 되지는 않는다.

세션6 보정 후19.18=4.36+14.82는 재현된다. ‘BF16-history 대조군 대비 추가 rollout 손실이 잔여의77%’라고 표현하고 ‘내용 오염77% 인과 확정’은 피한다.

### 3.5 추가 재양자화 오차0은 원본 대비 오차0이 아니다

현재 writer 진단은 cache_k와 이번 k_hat의 차이다. Q(x̂)=x̂여도 최초 원본 대비 x̂−x는 남는다. empirical 보정 실패만으로 Jensen 기전·균일 오차 가정·평균 편향의 기여를 기각할 수 없다.

동일 내용의 최초 양자화 오차, 추가 저장 오차, FR/BF16 history의 궤적 차이를 구분한다. FR에서 생성한 BF16 키도 이미 FR history의 산물이다.

‘정렬 OFF에서만 재양자화 오차 비0’은 검증한 RTN 조건에 한정한다. KIVI·scale 재산정·다른 양자화기로 일반화하지 않는다. ε/Δ, q·ε 조건부 통계, clipping, 채널 상관을 확인한다. 분산비≈1과 pooled 평균≈0만으로 균일성·무편향을 증명하지 않는다.

### 3.6 key는 주요 병목이지만 value도 중요하다

K4V2−K4V4 = −2.624, t=−4.754, 95% CI [−3.872, −1.375], 10/10 악화. ‘key 단독·value 무관’은 틀리다. RTN K2 실패로 보편적3비트 하한을 주장하지 않는다. ‘검증한 RTN 구성과 품질 조건에서 K2가 부족’으로 한정한다.

### 3.7 flatten은 query·head별 상수다

flatten은 per-key 차이만 없애고 query/head 차이를 보존한다. 고정 head 상수나 전역 상수와의 동등성은 별도 문제다. 세션6의 d_eff≈12, 상대 산포≈0.15는 ‘d=128이니 자동으로<0.1’이라는 설명을 지지하지 않는다.

q 고정, X_ic=Δ_ic², w_c=q_c²/(24d)라 두면:

    E[b] = Σ_c w_c E[X_c]
    Var(b) = Σ_c Σ_e w_c w_e Cov(X_c,X_e)

독립 채널일 때 a_c=w_c E[X_c], κ_c=sd(X_c)/E[X_c]로:

    σ_b/μ_b = sqrt(Σ a_c² κ_c²) / Σ a_c
             ≤ κ_max / sqrt(d_eff)
    d_eff = (Σ a_c)² / Σ a_c²

CV는 Δ²의 CV다. 독립성·유효차원·공통요인을 명시한다. 상대 산포가 작아도 평균 bias가 크면 절대 logit 산포는 클 수 있다.

캐시 전체 mass를 맞추는 query/head별 상수:

    c* = log Σ_cached exp(s_i) − log Σ_cached exp(s_i − b_i)

이 값은 캐시 내부 attention 분포나 전체 출력을 일치시키지 않는다. 단순 산술평균과도 일반적으로 다르다.

### 3.8 품질·통계·시스템 해석

- 정렬ON recipe의 flickering은 BF16보다 유의하게 낮다. 그러나 ON−OFF 직접 비교는95% 기준 비유의다. 한쪽만 유의하다는 사실은 두 효과의 차이가 유의하다는 뜻이 아니다.
- VBench imaging_quality는 MUSIQ 기반이므로 독립 증거로 세지 않는다.
- subject consistency/flickering은 정지·움직임 감소를 보상할 수 있다. motion·prompt fidelity·blind 영상 평가를 보완한다.
- LongLive는 확인 시점 BF16 master를 유지했다. 목표 packed bytes와 실제 상주 메모리는 별개이며 재학습과 저장 방식도 교란된다.
- gateO1_residual.py:paired()는 CI에1.96×SE를 사용했다. 평균·t가 맞아도 CI는 수정한다.
- LOO는 다중비교나 동일10 prompts에서 hyperparameter를 선택한 편향을 해결하지 않는다.
- AI=4680은 이상적 KV 재사용 가정이다. 실제 HBM traffic·tiling·L2·unpack·occupancy 없이 compute-bound를 확정하지 않는다.

## 4. 주요 독립 재계산 값 — 9월18일 snapshot

아래는 n=10, chunk 평균을 prompt별로 집계한 paired 차이,95% t-CI다.

| 대비 | 평균차 | t | 95% CI |
|---|---:|---:|---|
| λ1−무보정 | +10.044 | 3.270 | [3.096,16.992] |
| flatten−Taylor λ1 | +0.448 | 0.617 | [−1.194,2.091] |
| FR λ4−λ1 | +2.078 | 0.685 | [−4.782,8.938] |
| TF λ1−λ0 | +4.688 | 2.816 | [0.922,8.455] |
| 정렬OFF K3V2 recipe−BF16 | −1.110 | −1.740 | [−2.554,0.333] |
| 정렬ON K3V2 recipe−BF16 | +0.728 | 1.259 | [−0.580,2.036] |
| A2 INT2 정렬ON−OFF | +11.500 | 8.277 | [8.357,14.643] |
| A1 INT2 정렬ON−OFF | +2.282 | 2.522 | [0.235,4.329] |
| 정렬ON A2−A1 INT2 | +7.675 | 6.174 | [4.863,10.488] |
| 정렬×sink 상호작용 INT2 | +9.218 | 6.188 | [5.848,12.588] |
| K4V2−K4V4 | −2.624 | −4.754 | [−3.872,−1.375] |
| oracle INT2−driver-oldest | +11.199 | 7.267 | [7.713,14.685] |

VBench temporal_flickering:

| 대비 | 평균차 | t | 95% CI |
|---|---:|---:|---|
| 정렬ON recipe−BF16 | −0.003390 | −3.303 | [−0.005712,−0.001068] |
| 정렬ON−OFF recipe | −0.000942 | −1.897 | [−0.002066,0.000182] |

A1 INT2 정렬 효과의 LOO 최소t=2.082로 df=8 임계값2.306 미달. A2는 최소t=7.442로 견고하다. 최신 결과 전부가 LOO 견고하다고 쓰지 않는다.

KIVI +2.60/+13.32는 rw1ch 결과다. rw0 대비 RTN은 +2.465/+12.843이다. 방향은 유지되나 residual 설정을 명시한다.

## 5. 보충서에 반영해야 할 추가 수정

HANDOFF_session6_addendum.md의 방향은 타당하다. 특히 B1 앵커 대조, B2 상호작용, B6 최종 경로의 일관된 측정이 중요하다. 다음은 후속 검토 의견이다.

1. A1:63 latent frames×3 prompts로 퇴출 이후를 검사하되 OFF 회귀와 ON 새 경로 검증은 분리한다. A절의 ‘GPU 없이’와 A1은 모순이다. 기존 GPU 작업에 무단으로 끼어들지 않는다.
2. A2:CI 계수는 실제 n으로 계산한다.2.262를 전역 상수로 쓰지 않는다.
3. A6:‘RTN3비트 하한’도 검증한 구성·품질 조건 내 관찰로 한정한다.
4. B1:비유의에서 기여0을 도출하지 않는다. 선택 이력 일치와 품질 동등성을 분리하고 앵커/최근 문맥 배분 효과로 표현한다.
5. B2:기존4셀의 prompt 대응을 확인해 재집계한다. 셀 평균·조건부 효과·상호작용을 제시한다. 상호작용이 있으면 순차 개선은 순서 의존이므로 독립 원인 비중의 누적 막대로 표현하지 않는다.
6. B3:c/head 상수 선택 데이터와 TOST 평가 데이터를 분리한다. head 상수가 layer×head인지 timestep별인지 정의한다.±1 MUSIQ의 실용적 이유를 밝힌다. c*는 mass 일치만 보장한다.
7. B4:RTN 멱등성을 일반화하지 않는다. 원본의 정의·capture 시점·RoPE·scale을 고정한다.3p×3layers×3chunks는 진단 pilot이지 균일성 확정에 충분하다고 미리 정할 수 없다.
8. B5:저장 방식·seed·history 대조를 맞추고 n=10 평균 지표로 구 INT4 비율을 갱신한다.
9. B6:최종 recipe에 정렬 상태·보정 형식·backend를 명시한다. P0 BF16 정면 비교를 함께 수행한다. 품질 실행과 hook 없는 성능 실행을 같은 commit/config로 연결한다.
10. B7:새 prompt+새 seed 한 묶음은 독립 확인이며 일반화 완결이 아니다. 가능하면 prompt×seed를 교차한다. 영상5쌍은 pilot이다. 무작위 파일 생성만으로 사람 평가 완료라고 하지 않는다. 평가자·기준·동률 처리·실제 결과를 기록한다.
11. B8:bit-diff만으로 올바른 보정을 입증하지 않는다. 같은 경로 무보정 대조, self 제외, Δ 배치, cache reset과 실제 attention 입력을 검증한다. BF16 master 상주량을 메모리 표에서 빠뜨리지 않는다.

## 6. 권장 작업 순서 — 후속 작업이 승인된 경우

1. 최신 상태와 실행 중 batch, 완료된 결과를 먼저 확인한다.
2. GPU 없이 용어·단위·CI·n·baseline·bit 지원 표기를 정정하고 결과를 commit/config/source에 연결한다.
3. GPU 가용성과 승인 범위를 확인한 뒤63f 회귀 및 새 경로 검증을 한다.
4. B1 앵커 대조와 B2 상호작용을 처리한다. 기존 결과 재집계를 우선한다.
5. B6+P0:구현된 최종 경로와 BF16의 품질·메모리·속도를 일관되게 측정한다.
6. 레시피를 고정하고 독립 확인용 데이터로 평가한다.
7. H3 오차 진단, 저장 조건을 맞춘 TF/FR, flatten/head/global 단순화를 중심 주장에 필요한 만큼 보완한다.

우선 과제는 실험 수 확대보다 **주장→대응 비교→코드→원자료→통계**의 정합성을 완성하는 것이다.

## 7. 운영상 주의

- 세션6 노트에는 실행 중 shell script 편집으로 실행 위치가 어긋나고, 감시자 중복 실행이 같은 출력에 쓴 사고가 기록돼 있다. 실행 중 script를 직접 변경하지 않는다. 새 script·새 출력 경로로 재개점을 관리한다.
- 회귀 검사 중 소스를 swap하는 작업과 다른 편집을 병행하지 않는다.
- 재사용 시 prompt 순서·seed·chunk 수·지표·decode·baseline·저장 semantics를 확인한다.
- 완료/계획, 측정/가설, 목표 payload/실제 allocation을 분리한다.
- 후속 보고는 확인한 것·정정한 것·미결·다음 검증을 구분한다. 관측에서 원인 확정으로 넘어갈 때 대조 실험 근거를 제시한다.

## 8. 신규성 경계와 논문 방향

- Jensen bias·보정: https://arxiv.org/abs/2605.26266
- Key 우선 비트: https://arxiv.org/abs/2502.15075
- 저비트 video cache·fused 구현: https://arxiv.org/abs/2602.02958
- 명목 압축과 BF16 재구성 buffer 문제: https://arxiv.org/abs/2603.27469
- Training-free sink/window 관리: https://arxiv.org/abs/2512.05081

기존 요소의 조합이나4090 구현만으로 신규 방법을 주장하지 않는다. 이번 문헌 확인은 제한적이며 최초성 보장이 아니다. 저장 이력에 따른 평가 교란과 격자×캐시 구조의 상호작용을 중심으로 하나의 분석 논문을 먼저 완성하는 편이 타당하다. 별도 시스템 논문은 최종 packed 구현, BF16 정면 비교, 실제 트래픽, 용량·동시성 이득을 확보한 뒤 판단한다.
