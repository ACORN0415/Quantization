# 별도 방법 탐색 — S9 이후 (2026-10-25)

작성 대상: 다음 연구 담당자와 사용자. 상태: 문헌 재조사·가설 정리 완료, 실험 미실행.
기존 분석 원고·S9 판정을 대체하지 않는다. 이 문서는 GPU 실행·대규모 구현을 지시하지 않는다.

## 1. 출발점과 경계

- S9 균일 PSAC8+잔여 분산 보정은 실제 생성에서 INT2 λ1 대비 subject consistency +0.0532 [0.0332, 0.0732]를 보였다. K4V2 대비는 −0.0119 [−0.0217, −0.0022]로 사전 비열등성 기준을 충족하지 못했다. 기존 후보의 fused 투자는 종료 상태다.
- 약 76% 회복은 두 평균 차이의 비율이다. 방향 보존 단독 효과나 메커니즘의 완결된 분해가 아니다.
- TT/Tucker의 모든 가능성, 모든 저랭크 표현, 모든 잔차 선택을 불가능하다고 결론내리지 않는다. 시험한 구성과 투자 결정을 보존한다.
- 기존 자료는 개발 자료다. consec2와 S9 신규 평가 프롬프트도 이미 열람·사용됐다. 새 확인 표본으로 재사용하지 않는다.
- 기존 10% 기준은 전체 상주 캐시(K/V codes·scales·metadata·pending 포함) 기준이다. GPU 총 사용량·peak가 10% 줄어든다는 의미가 아니다.

## 2. 문헌 재조사에서 바뀐 판단

웹 원문은 아래 판본을 2026-10-25 확인했다. 원문 사실, 저자가 명시한 future work, 우리의 가설을 구분한다. 구현은 OSCAR·OptR의 저자 연결 저장소 존재와 공개 페이지까지 확인했으며 commit 고정·실행·줄별 감사는 하지 않았다.

| 선행 연구 | 원문에서 확인한 범위 | 방법 후보를 제한하는 점 |
|---|---|---|
| TUM, 2605.26266v1 §6 | 명시적 한계: attention이 소수 토큰에 집중되면 기대값 보정의 이득이 제한된다. 비균등·편향 오차의 유도 수정, FP/MXFP/NVFP 확장은 명시적 후속 방향 | 잔차 분산이 균등 예측과 가깝다고 소수 토큰 문제까지 해결되는 것은 아니다 |
| QuantWM, 2609.26425v1 §4·C.3·D.1 | QSAC+PSAC, 과거 query 통계의 causal 갱신, cold start 처리, packed 저장과 재구성 구현을 설명 | S9 RTN+자기 query PSAC+R4d는 QuantWM 전체 재현이 아니다. PSAC 하나의 실패를 QuantWM 반증으로 쓰지 않는다 |
| TempoKV, 2610.15324v1 §3.3 | BF16 trace에서 predictor를 맞추되, 입력은 양자화·복원된 이력이며 K를 맞춘 뒤 복원 K를 이용해 V predictor를 맞춘다 | '복원 이력으로 calibration' 자체는 이미 있다. 실제 압축 rollout으로 모델 상태가 달라지는 경우와, BF16 trace를 codec으로 재구성한 경우를 구별해야 한다 |
| OSCAR, 2605.17757v1 Appendix D | 명시적 한계: frozen-error surrogate와 실제 autoregressive 목적 사이 차이. mixed format·다른 하드웨어·일부 channel-wise 결합도 후속 과제 | covariance-aware rotation이나 clipping을 적용했다는 것만으로 새 방법이 아니다 |
| OptR, 2608.02691v1 §4 | post-W_O attention 출력 오차로 K/V 회전을 최적화하고 key offset 재매개변수화를 수행 | 'attention 최종 출력으로 학습'과 'K/V를 함께 고려'는 이미 존재 |
| Q-VDiT, ICML 2025 | 시간적 상관 보존을 위한 Temporal Maintenance Distillation | '시간 일관성 loss를 넣는다'는 주장만으로 신규성이 생기지 않는다. weight/activation 양자화와 AR KV 양자화의 차이를 구체화해야 한다 |
| ReadKV, 2610.11245v1 §7 | stored precision과 fetched precision을 구별. 명시적 후속 방향: allocation·reconstruction·attention 융합, 전체 모델 시스템 평가 | 낮은 읽기량을 낮은 상주량으로 혼동하지 않는다. 현재 메모리 목표에 그대로 맞는 해법은 아니다 |
| Attention Matching, 2602.16284v1 §3/6 | attention 출력과 질량 보존을 분리하고 attention bias를 맞춘다. 온라인 compaction은 명시적 후속 방향 | 질량/내부 출력 분해 자체는 새 기여가 아니다 |
| KVSculpt, 2603.27819v1 §3.2 | 출력 MSE와 LSE matching으로 압축 캐시를 최적화한다 | 분모를 맞추고 출력을 맞춘다는 조합도 선행 존재 |
| KVLinC, 2510.05373v1 §3 | key 양자화에 의한 attention 오차를 작은 학습 correction adapter로 보정 | 상수 크기 adapter나 학습 bias 추가만으로 신규성 부족 |
| Forcing-KV, 2605.09681v1 §4·§7·Appendix F | head 기능별 압축과 양자화 결합을 이미 다룬다. 학습 단계에서의 KV 축소는 명시적 future work | '움직임 head에 예산을 더 준다' 또는 'pruning+quantization'만으로 신규성 부족 |

기존 지시서의 TempoKV 요약은 BF16 trace 수집만 강조하면 복원 이력 calibration이 빠져 읽힐 수 있다. 이 문서에서는 위 구분을 우선한다. 기존 지시서는 수정하지 않았다.

## 3. 새 방법 후보로 세지 않을 것

1. P2/P3의 chunk-local·write-once를 다시 명명하기.
2. PSAC rank·층 배분을 조금 바꾸는 것만으로 새 기여를 주장하기.
3. 단순한 λ·clipping 튜닝, rotation, post-W_O loss, temporal loss를 각각 처음처럼 제안하기.
4. 공개 논문의 future work라는 이유로 현재도 미해결이라 단정하기.
5. 모델·카드·문맥을 바꾸는 것만으로 기존 기법의 새 방법성을 주장하기.

## 4. 탐색 가설 A — 캐시 전체 질량과 캐시 내부 선택 오류를 분리한 보정

상태: 기존 분해를 우리 상태에 적용하는 진단에서 출발한다. Attention Matching·KVSculpt가 이미 출력/질량 보존을 다룬다. 분해 자체·LSE loss·질량 bias는 신규성이 아니다. 제약된 AR diffusion 배포에서 기대값 보정의 실패를 예측하고 저비용으로 제어하는 규칙이 실제로 남는지 미확인이다. 1차 우선순위는 학습이나 커널이 아니라 분해 진단이다.

### 왜 이 질문인가

TUM의 토큰별 보정은 과거/current 질량을 바꾸고 캐시 내부 순서도 바꿀 수 있다. S9의 outlier 현상은 질량 억제의 관찰이고, PSAC은 로짓 자체를 바꾼다. 현재 둘을 하나의 attention 오차로만 비교해, 어느 부분을 고쳐야 하는지 완전히 분리하지 못했다.

### 정확한 오프라인 분해

고정된 동일 Q, 동일 self K/V, 동일 causal mask에서 과거 logits를 z_p, self logits를 z_s라 한다.
- m = Z_p/(Z_p+Z_s), p_p = softmax(z_p).
- 출력 O = m·(p_p V_p) + (1−m)·(p_s V_s).
- 따라서 과거 질량 m, 과거 내부 분포 p_p, V 복원의 차이를 독립적으로 교체해 조사할 수 있다.
- oracle 질량 보정: b(q) = logsumexp(z_hat_p) − logsumexp(z_ref_p)를 모든 과거 logit에서 뺀다. 그러면 과거 내부 분포는 그대로 두고 전체 과거 질량을 참조와 일치시킬 수 있다.
- 이 oracle은 BF16 과거 K를 요구한다. 배포 방법·품질 보장·인과적 상한으로 부르지 않는다. 각 오류 항의 비선형 상호작용 때문에 순차 교체량이 유일한 기여율도 아니다.

### 가장 작은 진단 명세 — 아직 실행 안 함

기존 캡처에서 고정 Q로 아래를 비교한다. 양자화 K/V, 참조 K/V, TUM λ1을 명시한다.
1. K2V2 무보정 / TUM λ1 / λ 및 clipping의 작은 동일 예산 탐색.
2. K2V2 + oracle 전체 질량만 교정.
3. 참조 과거 내부 분포 + 양자화 V + 후보 질량.
4. 참조 V + 후보 과거 분포·질량.
각각 post-W_O 출력 오차(가중치 필요), 과거 질량 오차, 과거 분포 JSD, head별·프롬프트별 결과를 보고한다. post-W_O만 새 지표라고 신규성을 주장하지 않는다. 기존 캡처에 없는 W_O·깨끗한 과거 상태는 원본의 출처·정밀도를 명시하고, 없으면 진단 범위를 줄인다.

### 분해 후에만 만들 수 있는 배포 후보

- 질량만 맞춰도 크게 회복하면, 현재 Q와 압축 상태에서 얻는 정보로 bounded한 cache-vs-self 보정량을 추정하는 모델을 검토한다. 추가 잔차 행·PSAC 저장 없이, calibration 파라미터·통계·읽기 계산량을 모두 포함한다.
- 질량보다 내부 토큰 선택이 지배하면 이 보정 후보는 종료한다. 총 질량 조절로 routing 문제를 해결한다고 주장하지 않는다.
- λ·clip 또는 KVLinC류 correction 기준선과 차별화되지 않으면 calibration 규칙 수준으로 분류한다. Attention Matching의 bias/LSE matching을 양자화에 옮겼다는 것만으로 새 방법이라고 쓰지 않는다.
- 질량 추정의 정확성은 폐기한 BF16 K 없이 온라인에서 유지돼야 한다. 질량을 0 근처에서 막는 것만으로 품질이 좋아진다고 가정하지 않는다.
- 수치 성공 기준은 진단 실행 전에 별도 고정해야 한다. 이 문서는 사전 등록 완료본이 아니다.

## 5. 탐색 가설 B — 실제 압축 rollout에서 시간 변화까지 보존하는 calibration

상태: A의 대안/후순위. OptR·Q-VDiT·TempoKV와 겹치는 요소가 많아 아직 신규성 낮음. 단독 λ sweep으로 확대하지 않는다.

검증할 차이는 두 축으로 분리한다.
1. 상태 수집: BF16 trace를 codec으로 재구성 / 실제 압축 캐시를 사용한 rollout의 내부 상태.
2. 목적함수: 단일 호출 post-W_O 출력 / 같은 denoising 단계에서 다음 chunk들까지의 상태 및 시간 변화 오차.

첫 진단은 같은 파라미터 수·탐색량으로 두 상태 분포가 서로 다른 고정 설정을 선택하고, 새 프롬프트·동일한 압축 실행 조건에서 유효한지 보는 것이다. 시간이 다른 denoising tensor를 같은 프레임 축으로 간주하지 않는다. clean teacher trajectory fidelity와 실제 영상 품질을 구별한다.

MSE에 단순 temporal loss를 더한 것만으로 새 방법이라고 쓰지 않는다. Q-VDiT와의 목적함수·최적화·상태 분포 차이를 식과 대조군으로 설명할 수 없으면 보류한다. λ·clip만 유효하면 작은 calibration 결과로 기록하고 강한 방법 논문으로 확대하지 않는다.

## 6. 지금 실행할 순서와 하지 않은 것

완료:
- S9 후속 방법 탐색을 분석 원고와 분리해 문서화.
- 추가 선행연구 중복 확인과 원문 링크 확보.
- A의 분해식·가장 작은 진단·종료 조건 제안.

다음 CPU/읽기 작업:
1. A와 가장 가까운 bias correction / partition correction 선행 구현을 감사한다. 저자 코드가 있으면 commit과 파일 위치를 고정하고 λ/clipping/temperature/질량 보정이 이미 있는지 확인한다.
2. 현 캡처의 원본 정밀도·query 시점·W_O 접근 가능성을 목록화하고, A 진단이 가능한 상태만 고른다.
3. 가설·비교 팔·표본·지표·숫자 기준을 별도 사전 등록한다. 기존 개발 표본의 재분석과 새 확인 실험을 구분한다.
4. 그 다음에만 GPU 진단을 제안한다. fused 개발은 진단·짧은 rollout 품질·비용 판별 뒤다.

이 문서 작성으로 GPU 배치, 모델 학습, 원본 코드 변경, 기존 데이터 삭제, commit/push를 수행하지 않았다. 새 방법의 발견·신규성·성공은 확정하지 않았다.

## 7. 원문과 공개 구현

- TUM §6: https://arxiv.org/html/2605.26266v1#S6
- QuantWM §4/C.3/D.1: https://arxiv.org/html/2609.26425v1
- TempoKV §3.3: https://arxiv.org/html/2610.15324v1
- OSCAR Appendix D: https://arxiv.org/html/2605.17757v1
- OSCAR 저자 연결 저장소: https://github.com/FutureMLS-Lab/OSCAR
- OptR §4: https://arxiv.org/html/2608.02691v1
- OptR 저자 연결 저장소: https://github.com/daniel-eai/Output-Aware-INT2-KV-Cache-Quantization
- Q-VDiT 게재본: https://proceedings.mlr.press/v267/feng25q.html
- ReadKV §7: https://arxiv.org/html/2610.11245v1
- Forcing-KV §4/7/Appendix F: https://arxiv.org/html/2605.09681v1
- 로컬 근거: 결과서 S9_cand1_diag.md §6, 지시서 S9_PSAC8_quality_prereg.md

- Attention Matching §3/6: https://arxiv.org/html/2602.16284v1
- KVSculpt §3.2: https://arxiv.org/html/2603.27819v1
- KVLinC §3: https://arxiv.org/html/2510.05373v1
