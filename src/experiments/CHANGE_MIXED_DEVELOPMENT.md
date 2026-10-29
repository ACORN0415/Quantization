# 변화량에 따른 부분 정밀도 배분: 작은 개발 진단

2026-10-30. 사용자 제안인 ‘변화가 큰 부분은 높은 정밀도, 작은 부분은 낮은 정밀도’를 검사한다. 현재 KV를 과거 위치에 맞추는 예측·잔차 방식이 아니다. 기존 SpatialMixed와 큰 개념이 가까우므로 신규성은 주장하지 않는다. [참조 구현](https://github.com/suraj-ranganath/kv-quant-longhorizon/blob/b4c09363735c89947222781db899862bc0dc55c5/kv_quant/spatial_mixed.py)은 V 절댓값 평균의 시간 분산을 사용하며, 이번에는 전체 V 벡터의 시간 차이를 쓴다. 참조 코드를 복사하지 않고 원래 그룹 보존·실제 패킹 코어를 독립 작성했다.

## 입력·분류 규칙

- 개발 p0/p4·seed101, 층0/14/29, 이미 생성한 청크6 최종 refresh 캡처만 사용한다. 기존 motion 캡처의 pre-RoPE K/V와 postwo 캡처의 K/V/Q/W_O를 재사용한다. motion V6가 postwo 새 V6와 비트 일치하고, pre-K에 원래 RoPE를 적용한 값이 postwo 새 K와 비트 일치해야 한다.
- 청크는 3잠재프레임·공간1560토큰·전체4680토큰이다. 현재 청크가 완성된 encode 시점에 세 프레임이 모두 알려져 있다. 생성 중 KV 일부를 영구 동결하는 실험은 아니다.
- 층0 V의 같은 공간 위치에서 `change = RMS(V[f+1]−V[f])`, `magnitude=RMS(V[f])`를 heads/channels와 해당 frames에 대해 구한다. `score=change/max(magnitude,0.1*전체 V RMS,1e−12)`로 고정한다. 작은 특징의 미세 잡음이 무한히 확대되는 것을 막는 분모 floor다. 이 score는 물리적 움직임·예측 불확실성 자체가 아닌 **특징 변화의 대리 지표**다.
- score 상위780개 공간 위치를 INT4, 나머지를 INT2로 정하고 같은 공간 마스크를 세 프레임·모든 층·K/V에 공유한다. 동률은 원래 공간 index가 작은 순서다. 반대 배분은 이 마스크의 정확한 complement다.

## 고정 11팔과 예산

1. 변화 큰 곳 INT4/작은 곳 INT2 (`change_high`).
2. 반대 배분 (`change_low`).
3. 단순 V magnitude 상위 절반 INT4 (`magnitude_high`): 변화 특유의 신호인지 살피는 보조 대조.
4. 무작위 절반 INT4 5개 (`random_9101`..`random_9105`), CPU Generator의 고정 seed로 spatial randperm을 만들고 세 프레임에 반복.
5. 균일 INT2/INT3/INT4 세 기준선.

혼합 팔은 모두 정확히2340토큰 INT4/2340토큰 INT2이며 실제 비트 payload가 같다. 균일3은 mask가 없어 조금 더 작으며, 같다고 부르거나 dummy padding으로 예산을 맞추지 않는다.

원래 token 순서의 g16 그룹에서 공통 절댓값 최댓값 M을 한 번 구해 FP16로 저장하고, 해당 토큰 비트폭의 scale=M/qmax로 복원한다. FG/BG로 나누어 그룹의 이웃·최댓값을 재계산하지 않는다. M metadata는 비트폭마다 중복 저장하지 않는다. 2/3/4비트 코드는 실제 uint8 bitstream에 패킹하고, 마스크도1bit/token으로 저장한다. K/V·층 사이 마스크 tensor를 실제 공유하며 unique storage로 센다. 내부 코드 범위는 symmetric −qmax..qmax로 통일한다. 과거 quantizer와 같은 산술이라는 주장은 하지 않는다.

## 평가 범위·진행 조건

기존 자료에는 과거 전체 BF16 V가 없어 K 양자화가 전체 attention 가중치에 미치는 영향을 정확히 재구성할 수 없다. 따라서 **주지표는 K를 BF16으로 유지한 V-only 추가 출력 오차**다. 선택에 쓰지 않은 다음 청크7 첫 denoise Q32개를 기존 K0..6/삭제 후1..6에 각각 적용한, 고정 FP32 post-W_O attention 오차 D를 측정한다. 두 상태 평균 D를 query×출력차원으로 정규화하고 3층 동일 가중해 프롬프트별로 집계한다. 실제 전체 전환·FR·영상 품질의 결과는 아니다. 다음 query는 마스크 선택에 쓰지 않는다.

K의 pre/post-RoPE 상대 L2, V 상대 L2, J, 동일 공간·층별 지표를 보조로 기록한다. K 복원 오차를 전체 KV attention 결과로 바꾸어 부르지 않는다. 전체66행(2입력×3층×11팔)을 공개한다. 무작위5개의 평균·범위를 함께 보고 p값은 계산하지 않는다. magnitude 대조도 사후에 추가한 것으로 만들지 않는다.

두 프롬프트 모두에서 `change_high`의 D가 (i)무작위5개 평균보다5% 이상, (ii)반대 배분보다5% 이상 낮고, (iii)균일3의1.02배 이하이며, (iv)균일3 대비 실제 V payload 증가가1% 이내일 때만 후속 검토로 간다. magnitude 결과는 별도 보조 비교이며 이를 못 이기면 시간 변화 특유의 장점이나 신규성은 주장하지 않는다. 모든 finite·비트 수·같은 예산·항등 게이트도 통과해야 한다. 이 기준은 개발 투자 기준이며 통계적 유의성·완성된 방법의 성능 기준이 아니다. 실패하면 score·비율·비트폭·seed를 바꾸어 구제하지 않는다.

## 검증·비용

CPU pack/unpack·tail·mask 공유·groupmax 불변·복원 위치·실제 bytes 검사를 먼저 통과한다. CUDA에서도 1/2/3/4bit 왕복을 확인하고, 같은 비트4/4를 혼합 mask로 나눠도 균일4 복원과 비트 일치함을 실제 입력에서 확인한다. 이는 영역 분할 때문에 그룹이 달라지지 않았다는 대조다.

mask/score 시간, encode/decode(1워밍업+3반복 중앙값), packed payload, 진단 peak allocated/reserved를 분리한다. 현재 attention은 복원 buffer를 사용하므로 packed payload 절감과 전체 추론 peak 절감을 동일시하지 않는다. 큰 입력은 재사용하고 새 영상은 생성하지 않는다. 결과는 `src/experiments/change_mixed_development/`, 로컬은 `~/sfq_data/change_mixed/`, 로그는 `~/sfq_data/change_mixed.log`다. 계획·코드 커밋 뒤 측정하고 기존 결과를 바꾸지 않는다.
