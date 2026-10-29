# 전환 rounding의 post-W_O 확장: head 혼합 대수 게이트

2026-10-29. 실제 연구 입력·모델·GPU를 사용하지 않는 실수식 유도와 CPU float64 property check다. 이전 pre-W_O 동등성의 예외를 검사한다. **일반 이차형식으로 표현된다는 사실만으로 신규성 부재나 쓸모없음을 판정하지 않는다.** 선행연구 비교와 실행 가능한 알고리즘의 가치는 별도 검토 대상이다.

## 가정과 표기

- 같은 query·K에서 토큰 삭제만 일어난다. 살아남은 토큰의 logit·위치·코드는 변하지 않는다.
- 원본을 가진 새 생존 청크의 V 복원 오차만 가변 항으로 본다. 과거 실제 오차가 없다고 주장하는 것은 아니다.
- query 하나에서 head별 pre-W_O 출력 오차를 이어 붙인 **열벡터**를 e라 한다.
- 살아남는 attention 질량의 역수 c_h를 각 head의 channel에 반복한 대각행렬을 C라 한다. 전환 후 오차는 Ce다.
- 실제 출력 projection을 열벡터에 작용하는 W로 표기한다. PyTorch linear weight도 같은 방향이다. M=WᵀW다. bias는 원본·양자화 출력의 차에서 소거된다.

## 목적과 정확한 이차형식

\[
 L(e)=\tfrac12(\|We\|^2+\|WCe\|^2)
       +\lambda\|W(C-I)e\|^2
       =e^T H e,
\]

\[
 H=\tfrac12(M+C^TMC)+\lambda(C-I)^TM(C-I).
\]

이 항은 일반적으로 query별 **전체 head/channel metric**이다. pre-W_O의 독립 head 제곱합과 달리 M에 head 간 교차항이 있으면 e_h와 e_g의 방향·부호도 목적에 영향을 준다. 같은 head별 norm이라도 head 간 상쇄가 달라질 수 있다. H의 표현은 각 query마다 합하거나 평균할 수 있다.

이 관계는 **새로운 목적이 scalar head 가중치만으로 대체되지 않을 수 있음**을 보인다. 그러나 quadratic-metric 표현 자체는 일반적인 least-squares 구조다. 기존 output-aware 알고리즘이 해당 metric을 이미 구현하는지, 이 metric으로 고른 코드가 보지 않은 query에서 유리한지, 적은 encode 비용으로 선택할 수 있는지를 따로 확인해야 한다. 이 유도만으로 새 연구 기여가 생겼다는 결론은 내리지 않는다.

## scalar head 가중치가 부족한 명시적 반례

두 head가 각각 1차원이고 W=[1,1], C=diag(1.5,2), λ=1이라 하자. 두 head의 삭제 질량은 각각 1/3과 1/2이므로 유효한 삭제 상태다. 이때

\[
 M=\begin{bmatrix}1&1\\1&1\end{bmatrix},\qquad
 H=\begin{bmatrix}1.875&2.5\\2.5&3.5\end{bmatrix}.
\]

e⁺=(1,1), e⁻=(1,−1)는 각 head의 norm이 같다. head-local scalar 가중치 방식은 두 경우 모두 5.375를 주지만 full 목적은 각각 **10.375와 0.375**다. norm만 보면 잃어버리는 교차항이 있음을 보인다. 또한 M의 rank는 1, H는 2여서 단일 query scalar를 M에 곱해서 H를 얻을 수도 없다.

이는 인위적으로 구성한 대수 반례다. 실제 모델의 W_O나 캐시에서 이런 상쇄가 중요하다는 실험 증거는 아니다.

## 단순화되는 특수 경우

1. **모든 head의 c가 같음:** C=cI이면 `H=[½(1+c²)+λ(c−1)²]M`이다. 이때 full post-W_O 오차의 query별 scalar 가중치로 정확히 환원된다. M이 head를 섞을 수 있으므로 독립 head Euclidean norm으로 환원된다는 뜻은 아니다.
2. **M이 head별 block diagonal:** head 간 교차항이 없으므로 `H_h=w_h M_hh`, `w_h=½(1+c_h²)+λ(c_h−1)²`다. 가중된 head-local projected quadratic metric이 된다. M_hh가 상수배 항등행렬이라는 조건 없이는 일반 head norm과 같다고 말하면 안 된다.

## 과거 전체 오차의 offset

삭제되는 과거 토큰의 실제 pre-W_O 오차 기여를 r라 하고 전체 이전 오차를 E라 하면, 이후 오차는 `CE+d`, `d=−Cr`다. full 목적은

\[
 L(E)=E^THE+2E^Tg+(\tfrac12+\lambda)d^TMd,
\]

\[
 g=[\tfrac12 C^T+\lambda(C-I)^T]Md.
\]

따라서 고정된 과거 오차는 선형항과 상수항을 추가한다. 원본 과거 V를 버린 encoder가 r 또는 d를 정확히 아는 것은 아니다. 이 offset을 BF16 원본에서 계산한 offline 비교를 그대로 배포 가능한 방법이라고 부르지 않는다. H가 singular일 수 있어 역행렬이 항상 존재한다고 가정한 완전제곱식도 여기서는 사용하지 않는다.

## CPU 확인

`tools/postwo_transition_gate.py`는 실제 데이터 없이 고정 seed의 CPU float64에서 다음을 검사한다.

- 무작위 32개 합성 W·query별 c에 λ 4개: direct 목적과 full metric의 목적·gradient 128건 일치.
- 같은 head norm의 부호 반례와 M/H rank 차이.
- head별 c 동일 특수 경우, block-diagonal M 특수 경우의 정확한 환원.
- 이웃 rounding 16개 후보에서 direct와 full metric이 같은 후보 선택. 구성한 toy 문제에서는 head-local surrogate와 다른 후보를 고름.
- 고정 과거 offset이 있는 목적과 quadratic+linear 전개의 λ 4개 목적·gradient 일치.

float64 오차는 테스트 허용 범위에서 비교한다. 이는 모든 입력의 형식 증명이 아니며 실수식 유도를 보조하는 property check다. 실제 양자화 부호화·배포 메모리·attention kernel·영상 품질·원본 모델 W_O를 검사하지 않았다.

```bash
cd ~/Quantization
~/sfq_data/py tools/postwo_transition_gate.py
```

판단 범위는 **post-W_O head 혼합에서는 이전 scalar 동등성으로 후보를 바로 탈락시킬 수 없으며, 전체 metric과 강한 기준선의 비교가 필요하다**는 데 그친다. 이 구조를 일반적인 quadratic output 목적이라고 부를 수 있다는 사실만으로 신규성이 없다고 판정하거나, scalar 기준선과 다르다는 이유만으로 신규성을 주장하지 않는다.
