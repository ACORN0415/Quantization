# 메커니즘 실험의 측정 보완 명세

상태: 실행 전 설계. 기존 결과를 다시 측정한 값이 아니며 v0.3의 진행 조건·효과 판단은 바꾸지 않는다.
목적은 저장 오차의 해석과 실제 구현 비용을 구분하는 것이다. 새 보류 프롬프트는 성능 개발에 쓰지 않는다.

## 1. 빠져 있던 구분

| 항목 | 기존 계획 상태 | 보완 |
|---|---|---|
| 캐시 저장 바이트 | 구성요소가 정의됨 | 시점·view 중복·baseline·최대치까지 정의 |
| GPU peak | 기록 이름만 있음 | allocated/reserved/장치 전체 샘플과 단계 분리 |
| CPU 메모리 | 재생 예산만 있음 | 프로세스 high-water와 단계별 표본 RSS, 전송량 |
| 속도 | 실측 후 예산 산정만 있음 | 계측 OFF 실행, warmup·반복·순서·측정 범위 고정 |
| 반복 연산의 실제 비용 | encode 횟수 기록 | 처리 원소 수·바이트·restore/cat/roll 비용·할당까지 진단 |
| 오차가 생기는 위치 | 층/호출·영역이 정의됨 | 토큰 나이·encode 누적 횟수·첫 eviction 전후 곡선 |
| KV L2의 attention 관련성 | 추가 가능성만 언급 | 고정된 작은 query 진단, 미구현 시 미검사 표시 |

## 2. 측정 모드 세 개

**정확도/메커니즘 모드:** v0.3의 전체 과거 KV 진단과 insertion BF16 원본을 유지한다.
층별 CPU 재생·원본 shadow·전체 norm·로그·동기화 비용이 들어간다.
이 모드의 시간을 모델 추론 속도나 실사용 VRAM으로 보고하지 않는다.

**성능 모드:** 같은 모델/저장 정책·RNG·정밀도에서 KV 오차·norm 기록·shadow·teacher replay·단계별 CPU 복사를 끈다.
기존 `RecordingQuantizer`의 norm/.item()와 `ChunkStore`의 무조건적 `error_stats` 계산도 제거해야 한다.
`record=None`만으로 진단이 꺼졌다고 가정하지 않는다. 변경은 새 어댑터에 두고 기존 봉인 소스는 유지한다.
진단 ON/OFF의 최종 잠재 텐서가 같은 기존 개발 입력에서 bit-exact인 게이트를 통과해야 한다.
성능 모드의 GPU 출력 버퍼/캐시 유지 규칙은 팔 간에 같게 하고, 최종 D2H 복사는 별도 측정한다.

**연산 프로파일 모드:** encode, restore, eviction 복사, concat, attention의 비용을 조사한다.
CUDA event/프로파일러의 오버헤드가 있으므로 이 실행 시간을 주 성능 결과에 섞지 않는다.
중첩되는 연산 시간의 단순 합을 전체 지연으로 주장하지 않는다.

## 3. 소규모 성능 벤치마크

기존 p00, seed101 한 입력을 고정하고 BF16/L16/C-repeat16/C-once16/L24 다섯 팔을 측정한다.
다른 g의 메모리는 전체 재생에서 기록하되, 측정하지 않은 팔의 추론 속도는 주장하지 않는다.
새 보류 데이터는 사용하지 않는다. 결과는 이 개발 입력/장치의 구현 성능이며 일반성 추론은 하지 않는다.
이 15회 성능 생성은 v0.3의 모든 구현 게이트와 GO 조건을 통과했을 때만 진행한다.
GO 미통과 시 성능 반복/새 보류 FR을 실행하지 않고 추론 속도는 미측정으로 남긴다.

- 각 팔·반복은 새 프로세스에서 같은 모델·config·입력 조건으로 시작한다. 모델 로딩은 별도 측정한다.
- 해당 팔로 8청크를 warmup해 첫 eviction까지 거친 뒤 캐시와 RNG를 완전히 초기화한다.
  warmup의 상태/출력은 본 측정에 넣지 않는다. 초기화와 임시 참조 해제는 측정 구간 밖이다.
- 팔별 126프레임 생성 전체를 3회 측정한다. 총 15개의 측정 궤적과 15개의 8청크 warmup이며 비용을 숨기지 않는다.
- 팔 순서 기준은 [BF16,L16,C-repeat16,C-once16,L24]다. 반복별로 왼쪽으로 0/2/4칸 회전한다.
  가장 빠른 반복만 골라 쓰지 않고 세 결과·중앙값·범위를 모두 공개한다. 세 반복의 p95를 성능 안정성 지표로 쓰지 않는다.
- 동일 GPU의 다른 작업과 겹치면 성능 실행을 보류/무효로 표시한다. 다른 사람의 프로세스를 종료하지 않는다.
  모델 품질이 나쁘다는 이유로 성능 결과를 버리거나 반복 횟수를 늘리지 않는다.
- 메모리 부족/실행 실패도 팔별로 남기며 그 팔만 해상도·창·dtype을 낮춰 비교하지 않는다.

### 시간의 범위와 단위

1. `model_load_s`, `text_encode_s`: cold/warm 상태를 표시한 준비 비용. OS 디스크 캐시까지 통제한 cold load라고 주장하지 않는다.
2. `generation_s`: 모델과 조건 임베딩은 준비된 상태에서 빈 KV 캐시 초기화·동일 잡음 준비부터
   42청크의 모든 denoise/context-refresh와 출력 조립 완료까지. VAE·MP4·디스크 기록·품질 평가·진단은 제외한다.
   시작 전과 끝에서 CUDA 동기화를 하고 monotonic wall clock으로 잰다. chunk마다 강제 동기화하지 않는다.
3. `latent_d2h_s`, `vae_decode_s`, `mp4_encode_s`, `artifact_write_s`, `quality_eval_s`: 각각 별도 기록한다.
   준비된 모델에서 실제 영상이 준비될 때까지의 전체 wall 시간도 별도로 잰다. 중첩한 단계의 합을 wall 시간으로 대체하지 않는다.
4. 첫 `run_chunk` 반환(첫 refresh 포함)까지의 latent-ready 지연과 청크별 지연은 별도 프로파일에서 구한다.
   streaming 픽셀 출력이 구현돼 있지 않으므로 이것을 첫 영상 프레임이 사용자에게 보이는 시간이라고 부르지 않는다.
5. 청크 0–6과 7–41, 첫 eviction(c7), 반복 eviction의 지연을 구분한다. 청크 p50/p95/max는
   같은 실행의 시계열 기술 통계로만 보고하며 독립 표본의 신뢰구간으로 해석하지 않는다.
6. 처리량은 latent frame/s=126/generation_s, chunk/s=42/generation_s와 전체 영상 준비 시간 기준
   pixel frame/s=501/video_ready_wall_s를 구분한다. 잠재 126프레임과 픽셀 501프레임을 섞지 않는다.

## 4. 메모리와 전송량

각 단계 시작의 baseline, 그 단계의 peak, 종료 상태를 기록한다.
준비/생성/복원·eviction/디코딩/평가를 분리한다. warmup 뒤에도 예약 allocator 메모리가 남을 수 있으므로
절대 baseline과 peak를 함께 표시한다. `empty_cache()`를 매 호출 안에 넣어 순위에 영향을 주지 않는다.

- GPU: PyTorch allocated 및 reserved peak를 따로 기록한다. NVML 장치 전체 메모리를 10Hz로
  샘플링해 컨텍스트·라이브러리/외부 할당을 보완한다. NVML 샘플 최대는 정확한 순간 peak가 아니다.
- CPU: 프로세스 lifetime RSS high-water와 단계별 10Hz RSS 표본 최대를 구분한다.
  모델 로드의 과거 peak를 재생 단계 peak로 오해하지 않는다. pinned CPU 버퍼도 포함 여부를 적는다.
- 캐시: 같은 chunk/call의 post-refresh 상태와 전 구간 최대 resident tensor 바이트를 기록한다.
  같은 storage를 공유하는 view는 중복 합산하지 않는다. 논리적 payload와 backing allocation 크기를 구분한다.
  코드/padding/scale/pending/residual과 진단 shadow·복원 버퍼를 구분하고 패딩을 중복 합산하지 않는다.
- `resident_tensor_bytes`, metadata를 포함한 `hypothetical_packed_bytes`, 모델 포함 전체 peak는 서로 다른 열이다.
  현재 바이트 컨테이너+BF16 attention의 실행 시간은 실제 packed INT4 커널의 실행 시간이 아니다.
- encode/restore 호출 수, 처리한 원소/논리적 바이트, 명시적 H2D/D2H 복사량을 기록한다.
  같은 encode 횟수라도 전체 버퍼/청크별 루프의 처리량·커널 launch·concat 비용은 다르다.
  논리적 복사 바이트를 GPU DRAM 실측 대역폭으로 부르지 않는다.

이 수치를 이용해 품질/속도/실제 메모리를 같은 표에 놓되, 같은 명목 비트폭과 같은 메모리 예산의 비교를 혼동하지 않는다.
사후에 창 크기나 비트폭을 바꿔 equal-memory 결과를 추가하지 않는다. 그 질문은 별도 계획이다.

## 5. 결과를 설명하는 데 더 필요한 진단

**기본 오차와 누적 추가 변화 분리:** 첫 encode 직후 오차, 다음 encode 전후 변화량,
attention 직전의 누적 삽입 BF16 대비 오차를 따로 저장한다. 원본 norm도 함께 기록한다.
g=24가 원래 첫 encode부터 오차가 작았는지, 반복 추가 변화가 줄었는지 구분하기 위해서다.

**나이/시점:** 현재 쓰기와 과거가 섞인 그룹·첫 부분 그룹·내부 그룹을 나누고,
과거 토큰의 청크 나이 1–6과 생존 중 누적 encode 횟수를 기록한다.
첫 eviction 전후와 denoise/context-refresh에서 급변하는지 본다. 결과를 보고 가장 좋은 층만 고르지 않는다.

**크기와 방향:** KV L2 감소만으로 attention이나 영상 품질 개선을 보장하지 않는다.
개발 query 진단은 층 0/7/14/22/29, context-refresh, 현재 query 위치 0/1170/2340/3510/4679를 고정한다.
teacher와 같은 query를 써서 과거 K 오차의 q·δK/√128 RMS를 측정하고, 가능하면
같은 읽기 창에서 BF16/양자화의 과거 대 현재 attention mass도 비교한다.
full attention matrix를 덤프하지 않는다. 이 경로가 출력 동일성·비용 게이트를 통과하지 못하면 미검사로 명시한다.
query 결과로 v0.3의 GO 조건이나 주 검정 팔을 바꾸지 않는다.

**분포의 극단:** 전역 평균만으로 심한 그룹 손상을 숨기지 않도록 층/호출별 scale의
min/median/p95/max, zero scale·비유한 값·0코드 비율을 스트리밍 요약한다.
전체 코드를 대량 덤프하지 않고 고정한 개발 원시 예시 예산 512MiB를 유지한다.
분포 요약에서 빈 버퍼/패딩을 유효 토큰과 혼합하지 않는다. 0코드 비율 자체를 손상의 증거로 보지 않고,
zero scale은 원래 all-zero 입력 그룹과 nonzero 입력 그룹을 구분해 기록한다.
이 값으로 clip 범위·scale dtype·rounding을 튜닝해 기존 RTN을 바꾸지 않는다.

## 6. 출처와 완료 조건

GPU 모델/VRAM/드라이버, PyTorch·torchvision·CUDA runtime·flash-attn 버전,
모델/코드/config 해시, dtype·TF32·attention backend·compile 여부, 실행 모드를 결과에 기록한다.
10Hz의 온도/클록/전력/이용률 표본과 다른 GPU 프로세스 존재 여부를 기록해 반복 속도 차이의 맥락을 남긴다.
시스템 CUDA toolkit 버전과 PyTorch가 사용한 CUDA runtime 버전을 혼동하지 않는다.

측정 누락은 미측정으로 남긴다. 기존의 진단 포함 시간이나 theoretical packed 바이트로 채우지 않는다.
이 성능 항목들은 개발 측정/실용 비용 정보이며 메커니즘 GO 조건이나 확증 품질 주지표를 추가하지 않는다.
메커니즘 구현 게이트가 실패하면 성능 비교도 시작하지 않는다. GO 미통과 시 그전에 얻은
개발 smoke의 시간은 실행 비용으로만 보존하고, 위 반복 성능 벤치마크나 새 FR은 실행하지 않는다.

근거: [PyTorch 2.6 CUDA 동기화와 메모리 설명](https://docs.pytorch.org/docs/2.6/notes/cuda.html).
비동기 CUDA 실행 시간과 allocated/reserved 메모리가 다른 개념이라는 점을 측정 설계에 반영한다.
