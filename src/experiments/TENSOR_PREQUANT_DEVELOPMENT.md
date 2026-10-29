# 先양자화 뒤 방법 적용의 추가 손상 검사

2026-10-30 사용자 명확화: 원본→양자화→방법 vs 원본→양자화.
인자 재양자화 비교가 아니다. 2/3/4비트 각각 동일 기존 개발 p0/p4 seed101,
layers0/14/29 chunk6 V와 chunk7 query32를 사용. BF16 K는 고정.

직접 RTN(group16 실제 packed 코드)을 복원한 값만 방법의 입력으로 준다.
원본은 평가에만 사용하며 column 선택/계수 fitting에는 제공하지 않는다.
이것은 복원된 수치에 대한 손실 근사이며 정수 코드의 무손실 압축이 아니다.
최종 상태에는 FP16 열/계수 및 int32 인덱스만 남긴다. 원래 양자화 코드나
원본이 배포상 함께 상주해야 한다는 뜻이 아니다. 이번 평가는 별도 GPU 작업
버퍼를 갖는 진단이므로 deployment memory 감소는 주장하지 않는다.

각 비트의 원래 RTN payload 이하인 rank22/29/36을 쓴다. 이들은 같은 예산
내 최대 정수 rank이며 unused bytes는 보고한다. 최종 FP16 factor가 원래
RTN보다 크게 작아지는 설계는 아님을 명시한다.
팔: plain RTN, static common, SVD, adaptive argmax, T-ARP 5개 시드9101..9105.
각 비트54행 총162행. 다른 seed로 좋은 결과만 고르지 않는다.

평가: 원본 대비 최종 tensor relL2와 post-Wo D, plain 대비 tensor relL2 및
추가 출력오차 ||O_method-O_plain||². D_final/D_plain과 그 차이도 보고한다.
추가 오차 norm²는 총 오차 증가량이 아니다(교차항 존재). 최종 D 감소도
가능하므로 결과의 방향을 사전 단정하지 않는다. query는 평가에만 사용.
원래 두 BF16 창에서 새 V 기여만 평균, 전체 모델 품질/출력 아님.

각 make는 양자화+복원+방법 전부 포함하여 encode 시간으로 보고한다.
최종 decode 시간은 따로 측정. warmup1 + 3회 median. 직접 RTN 대비 차이는
근사 추가 비용과 중간 복원 비용 모두를 포함. CUDA 동기화.
기존 3개 비트 직접 RTN 기준값 재현, source/input SHA, finite, payload 이하,
CPU/GPU 복원 일치, synthetic 기존 gate를 확인한다. 결과 파일 새 폴더에 보존.
