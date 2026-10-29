# post-W_O 전환 rounding: 새 청크 오차의 개발 진단

2026-10-29. 실제 Q/K/V 캡처·측정 전에 고정하는 계획이다. 이전 pre-W_O 대수 중단은 그대로 유지한다. [대수 검사](POSTWO_TRANSITION_GATE_KO.md)는 scalar head별 손실과 post-W_O 전체 손실의 차이를 보였지만, 새로운 알고리즘의 신규성·효과를 입증하지 않았다.

## 질문·범위

실제 W_O가 섞는 head 사이 오류 상쇄를 반영하면, 현재 query에서 고른 V 코드가 다음 청크 query에서도 캐시 삭제 전후 출력 오차 변화를 줄이는가? 확인한 OptR 논문 §4.4–4.5의 목적은 head별 투영 오차의 norm을 더한다. 이번 full 출력 오차와 다르지만, 이것만으로 모든 선행연구와 차별성이 입증된 것은 아니다. 일반 이차 metric이라는 이유만으로 쓸모없다고 결론내리지도 않는다.

- 개발 p0/p4·seed101, 층 0/14/29. BF16 첫 8청크 재실행이 기존 126잠재프레임 기준 prefix와 비트 단위로 같아야 한다.
- 청크6의 최종 context refresh에서 전체 post-RoPE K(청크0..6), 새 청크6 V와 W_O를 캡처한다. K는 BF16 고정이다. 새 청크 V만 압축해 생기는 **추가 출력 오차**를 측정한다. 기존 과거 캐시 오차를 0이라고 가정하는 surrogate이며, 과거 RTN 오차가 실제로 없다는 뜻은 아니다. 과거 BF16 V 원본을 loss 계산에 사용하지 않는다.
- 선택용 Q는 청크6 refresh, 평가용 Q는 청크7 첫 denoise의 실제 post-RoPE Q다. 각각 4680개 중 고정 linspace 32개를 고른다. 평가 Q는 코드 선택에 쓰지 않는다. 두 Q 모두 BF16 teacher 실행에서 얻으므로 압축 rollout/FR 일반화를 주장하지 않는다. 이미 사용한 개발 입력이므로 보류 연구도 아니다.
- 각 Q를 같은 청크0..6 K에 적용한 상태와, 맨 앞 청크0의 4680토큰만 지운 상태를 비교한다. 미래 청크7 K/V를 추가하지 않는 **동일 query의 반사실적 삭제 진단**이다. 실제 전환 전체가 아니다. 원래 RoPE 위치는 바꾸지 않는다.
- 명시적 FP32 QK softmax와 FP32 AV/W_O를 사용한다. BF16 flash attention과 비트 단위 출력이 같다는 주장은 하지 않는다. 모든 팔이 같은 진단 산술을 쓴다.

## 팔·탐색

K16/V4, g16, 저장 FP16 scale 고정·uint8 코드 컨테이너. 새 청크 V 코드만 이웃 floor/ceil 중 선택한다. 기존 캐시 코드를 고치지 않는다. 새 code/scale의 실제·가상 packed 바이트가 모든 팔에서 같아야 한다.

1. RTN 초기 코드.
2. 전체 post-W_O 평균 출력 오차 D 최소화(λ0).
3. 전체 post-W_O `D+J` 최소화(λ1).
4. head별 post-W_O 기여 norm을 더한 `D+J` 최소화(λ1): coupling 보조 비교. OptR 전체 알고리즘 재현이라고 부르지 않는다.

`D=(||We0||²+||We1||²)/2`, `J=||W(e1−e0)||²`. 고정 projected-gradient 32회, 같은 RTN 초기 후보를 포함하고 목적별로 관찰한 최선 hard 후보를 선택한다. scale은 저장 FP16 값으로 공유한다. 각 팔에 실제 탐색 벽시계를 기록한다. 같은 반복 수가 같은 FLOPs라는 뜻은 아니다. 이웃 탐색이 전역 최적해라는 주장도 하지 않는다.

## 사전 판단

주 비교는 전체 λ1 대 전체 λ0다. head별 팔을 이긴 것만으로 전환 항의 이득이라고 주장하지 않는다. 평가 Q의 D/J를 각 층에서 query·출력 channel 평균 후 3층 동일 가중하여 프롬프트별로 계산한다.

**두 프롬프트 모두 J가 λ0보다 10% 이상 작고, D는 λ0의 1.02배 이하일 때만** 다음 단계 검토로 간다. baseline J가 `1e-12*max(D,1e-30)` 이하이면 비율로 성공을 주장하지 않고 해당 입력은 미결정/진행 안 함으로 센다. 모든 finite/동일 byte/게이트도 통과해야 한다. 10%/2%는 작은 개발 투자 기준이며 통계적 유의성·영상 품질 기준이 아니다. 실패하면 λ·입력·query sampling을 바꾸어 GO를 구제하지 않는다.

출력은 전체 norm과 head별 norm을 구분해 공개한다. W_O의 실제 사용, 이차 metric과 직접 목적/gradient 일치, 합성 headmix 반례, CPU 코드 불변성·후보 범위·바이트·최선 후보 검사를 먼저 통과한다. 실제 결과는 2프롬프트×3층×4팔=24행이며 train/eval을 같은 행의 별도 열로 기록한다.

## 자원·기록

캡처/탐색의 CUDA allocated/reserved peak·CPU high-water, 파일 용량, spectral norm 계산·query 확률 준비·팔별 탐색 시간을 분리한다. 새 V payload와 고정 K/복원 작업공간을 구분한다. 배포 속도·메모리 절감을 주장하지 않는다. 대형 원본은 `~/sfq_data/postwo_rounding/`, 로그는 `~/sfq_data/postwo_rounding.log`, 작은 결과는 `src/experiments/postwo_rounding_development/`에 둔다. source·입력 해시를 확인하고 기존 결과를 덮어쓰지 않는다.
