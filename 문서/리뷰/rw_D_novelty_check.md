# Related-work 신규성 점검 (2026-09-17, 세션 5 시작 시점)

대상: 세션 1~4·감사에서 확정한 발견 + 세션 5 계획(H 실측 bias 보정, I K/V 비대칭, J fused 커널).

## 이미 문헌에 있는 것 (우리 것이라 주장 불가)

| 우리 항목 | 선행 | 겹치는 정도 |
|---|---|---|
| key 양자화 → softmax Jensen bias → 캐시가 attention을 훔침, sinh 정확식 + 2차 테일러 보정, FlexAttention score_mod 커널(~5% 오버헤드) | **TUM** "Quantized Keys Steal Attention" arXiv 2605.26266 (MAGI-1·SkyReels-V2·HY-WorldPlay, RTN/QuaRot INT2·INT4) | 메커니즘·보정식·커널 모두 선점. 세션 5 H2(exact_uniform)는 그들 Eq.12 재현 = **baseline**이지 기여 아님 |
| key가 value보다 비트를 더 필요로 함(K4V2 ≫ K2V4) | **KV-AdaQuant** "More for Keys, Less for Values" arXiv 2502.15075 (LLM, 스펙트럼 노름 논리, K4V2 75.2% vs K2V4 54.7%); **AsymKV**(2024); **TurboQuant-inspired** arXiv 2605.08114 (softmax 볼록성으로 K/V 비대칭 이론화, value는 1/√S로 평균되나 key는 안 됨) | LLM에서 방법·이론 모두 존재. 비디오 diffusion에서의 비대칭 격자(관문 I)는 **전이(transfer)**이지 새 방법 아님 |
| INT4 KV의 실제 비용 = BF16 재구성 버퍼 | **UCSD 33-method study** arXiv 2603.27469 ("transient BF16 buffer reconstruction during attention") | 진단 선점. 우리 몫은 "없애는 커널"뿐 |
| 재학습 없이 sink·window 조정으로 Self Forcing을 LongLive 수준으로 | **Deep Forcing** arXiv 2512.05081 (training-free deep sink 40~60% + participative compression, LongLive보다 좋음 — 양자화 없음) | 캐시 구조 레시피는 선점. 우리 "캐시 구조가 INT4 피해 70% 회복"은 **양자화 맥락**에서만 새로움 |
| 저비트 KV를 attention에서 직접 역양자화하는 fused 커널 | **QVG** arXiv 2602.02958 (Self-Forcing-Wan-1.3B 포함, 1.5~4.3% 오버헤드, pre-RoPE key) | 커널 존재. 우리 J는 "패킹 + bias 보정 + quantize 커널 통합, 소비자 GPU 실측"으로 차별화해야 함 |
| Self Forcing 계열 KV 압축(head별 hybrid pruning) | **Forcing-KV** arXiv 2605.09681, **Future Forcing** 2605.30083, **Head Forcing** 2605.14487, **VideoMLA** | 압축 계보. 양자화 아님. 인용 필요 |

## 아직 우리 것 (2026-09-17 검색 기준)

1. **TF/FR lockstep 분해 → 전파가 주입을 5:1로 압도.** TUM은 동일 입력 fidelity(PSNR/SSIM/LPIPS vs BF16)만, UCSD는 drift 측정만. chunk 간 전파를 주입과 분리해 잰 논문 없음. 카오스 바닥선(노이즈 대조군 L2 포화)과 지표 삼각측량도 포함. → **분석 논문의 핵심**.
2. **TUM 보정이 INT2에서 과소 보정된다는 실증**(Δ=scale +10.4 vs 4배 +28.3, t=−3.46). TUM Fig.A1은 "테일러가 큰 노이즈에서 과대 추정"이라 정반대 방향을 주장. 원인 후보: 우리 RTN sym 그룹16 vs 그들 asym per-token g=32의 Δ 정의 차이, 또는 INT2 클리핑으로 균일 가정 붕괴. TUM 스스로 "nonuniform or biased error may require modified derivation"을 한계로 명시 → **H3(실측 2차 모멘트 online 보정)이 정확히 그 문을 연다.** 방법 논문의 1번 기여 후보.
3. **sink는 attention 자석이 아닌 내용 앵커**(chunk 0 mass 8.5% < 14.3%, 노이즈 sink 유해, 오염은 슬롯 나이에 따라 감소). Deep Forcing은 위치 기반 해석. 양자화 오염과 결합한 분석은 없음.
4. **key 단독 원인의 직접 조작 증명**(K2V4 붕괴·K4V2 = BF16, chunk 1부터, 퇴출 무관)과 attention 분포 역전(self 48→19%, 최고령 7→30%) 궤적. TUM은 mass shift 중앙값만.
5. **compute-bound(AI = 쿼리 수) → KV는 대역폭 아닌 용량 문제** 프레이밍과, int8 저장이 아닌 진짜 4비트 패킹의 소비자 GPU 메모리 실측.

## 세션 5 지시서에 반영할 것

- H2 `exact_uniform` = TUM Eq.12 재현으로 **재명명**(baseline). 새 기여는 H3만.
- H1·H3에서 TUM과 Δ 정의를 맞춘 비교 1회 추가(asym per-token g=32) — "과소 보정"이 우리 양자화기 설정의 산물인지 확인하지 않으면 TUM 저자 반박에 무너진다.
- 관문 I 서술: "KV-AdaQuant/AsymKV의 K/V 비대칭을 스트리밍 비디오 diffusion에서 검증하고, 전파 축(chunk)에서 Pareto를 처음 측정"으로. 방법 주장 아님.
- Deep Forcing을 캐시 구조 baseline으로 추가(가능하면 deep sink 설정 1개를 INT4에서 실행).
- J의 비교 대상: QVG fused(dequant 중심)·TUM FlexAttention(보정 중심) 둘 다. 우리 커널의 차별점 = 패킹 직접 소비 + 보정 + quantize 커널 + 4090.

## 논문 포지셔닝 결론

분석 논문(arXiv): 1·3·4·5로 자립 가능. "왜 무너지는가"를 전파/주입 분해와 조작 실험으로 답한 유일한 논문.
방법 논문: 축을 "비대칭 K/V + 보정 + 커널"에서 **"실측 분산 기반 online 보정(TUM의 균일 가정 한계 해결) + 그것을 패킹 저비트 attention 안에서 공짜로 수행하는 커널"**로 좁혀야 한다. 비대칭 격자는 그 위의 실험 축.
