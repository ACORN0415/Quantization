# 압축 KV 복원값 재사용 — 코덱 실행 가능성 확인

2026-10-31. **판정: QVG를 사용한 제한된 비용 진단으로 진행할 수 있다.**
현재 확보한 것은 공식 packed 코덱의 합성 입력 실행·복원 정확성·호출 수 증거다.
전체 생성 가속, 최종 latent 동등성, H2의 예측 성능은 아직 측정하지 않았다.

## 1. 코덱 선택

공식 [QVG 저장소](https://github.com/svg-project/Quant-VideoGen)의 revision
`0601468f2dbba6a17ac7086faec6d41527cad188`을 별도 디렉터리에 체크아웃했다.
공식 소스는 수정하지 않았다. INT2/1-stage/256 centroids/group64/k-means2회,
symmetric residual/no clipping은 해당 Self-Forcing launcher와 같다.

현재 RTX4090(SM89), PyTorch2.6.0+cu124, Triton3.2.0, FlashAttention2.7.4.post1에서
커널 실행을 확인했다. 프로젝트의 torch2.8.0 의존성을 그대로 설치한 공식 환경
재현은 아니다. 누락된 termcolor2.5.0와 loguru0.7.3은 `~/sfq_data/qvg-reuse-deps`에
격리 설치했고 기존 환경을 업그레이드하지 않았다.

QuantWM의 [공식 프로젝트 페이지](https://quantwm-project.github.io/QuantWM/)와
[논문](https://arxiv.org/html/2609.26425v3)은 확인했으나 이번 확인 범위에서 실행 코드
링크를 확보하지 못했다. 동명 DINO-WM 양자화 저장소를 해당 논문의 코드로 혼동하지
않는다. 따라서 첫 실행 후보는 실제 코드를 확보한 QVG다.

[ZipFusion](https://arxiv.org/html/2610.33719v1)은 논문상 BF16 lossless 표현,
FlashInfer/SGLang, Llama3.1-8B/Qwen3-8B, H100 평가를 다룬다. 이번 확인에서
저자 공식 구현을 확보하지 못했으며 RTX4090·영상·QVG 표현의 지원을 검증하지
않았다. **실행 가능한 동일 코덱 fused 기준선으로 등록하지 않는다.**
QVG 코드에서 확인한 fused 연산은 unpack·scale·centroid 복원이며, Self-Forcing
경로는 별도로 BF16 attention을 호출한다.

## 2. 실행 전 계획과 실제 결과

[고정 계획](../KV_REUSE_PREFLIGHT.md)에 따라 GPU 유휴를 확인한 뒤 실행했다.
최대600초·2시도·allocator6GiB 상한 중 **1시도, 22.477초**에 통과했다.
JIT/검증을 포함한 시간이며 추론 성능 수치가 아니다. 새 영상 생성은0회다.

합성 BF16 K/V 한 쌍의 형상은 각각 `[1,12,4680,128]`이다. 모델에서 채취한
활성값이 아니다. 원본 값은 인코딩 후 버리고 압축 표현에서 복원한 값만 비교했다.

| 항목 | 관측 결과 |
|---|---:|
| 원래 BF16 K/V 한 쌍 바이트 | 28,753,920 |
| packed residual+scale+centroid+index의 실제 tensor storage | 5,504,064 |
| 해당 합성 형상의 저장 비율 | 약5.224배 |
| residual dtype/형상 | uint8 / `[1,12,4680,32]` |
| 독립 PyTorch unpack/복원과 공식 decoder | K와 V 모두 원소 바이트 일치 |
| 원본·단일 복원·유지 경로의 attention 비교 | 5 queries × 3경로 모두 일치 |
| 압축 상태의 검사 전후 지문 | 동일 |
| 검사 전체 allocator peak allocated / reserved | 329,617,408 / 589,299,712 bytes |

위 메모리는 검증 임시 텐서까지 포함한 **합성 검사 전체**의 값이다. 경로별
메모리 비교 또는 전체 모델의 peak가 아니다. 5.224배는 이 작은 형상의 실제 저장
비율이며 논문의 장문맥 압축률을 재현한 수치가 아니다.

## 3. 주 비교 전에 제거해야 할 중복 복원

공식 [kv_cache.py:172](https://github.com/svg-project/Quant-VideoGen/blob/0601468f2dbba6a17ac7086faec6d41527cad188/quant_videogen/kv_cache.py#L172)는
한 K 또는 V span에 `uncompress_kv_cache(state,state)`를 호출하고 첫 결과만
사용한다. [uncompress.py](https://github.com/svg-project/Quant-VideoGen/blob/0601468f2dbba6a17ac7086faec6d41527cad188/quant_videogen/uncompress.py#L58)는 두 입력을 각각 복원한다.

실제 `triton_prq_dequantize_tensor` 진입점에서 계수한 결과:

| 경로 | K/V 한 쌍을5회 읽을 때 decoder 호출 |
|---|---:|
| 공식 ChunkedKVCache.read 그대로 | 20 |
| 동일 decoder로 각 K/V를 한 번씩만 복원 | 10 |
| 최초 복원값을5회 읽기 동안 유지 | 2 |

독립 복원 검사의 추가2호출은 별도 `reference_validation` 항목이며 위 횟수에
섞지 않았다. 입력과 출력의 바이트 비교를 위해 계측 경로에서 복원을 추가하지
않았다. 세 경로 모두 같은 레이아웃 변환과 read의 concat을 적용했다.

따라서 후속 주 비교는 **중복 제거 후 매번 복원(B) vs 복원값 유지(R)**다.
공식 그대로(U)는 재현·구현 차이 설명용으로 남긴다. U→B 개선을 R의 연구
기여로 합산하지 않으며, 이번 관찰로 발표 논문 수치가 영향을 받았다고 추정하지 않는다.

## 4. 생성 연결 전에 확인할 조건

공식 Self-Forcing은 `QUANT_FACTOR=8`이며 여덟 생성 청크를 묶어 압축한다.
따라서 이번3잠재프레임 합성 span 검사는 전체24잠재프레임 native span의 실행,
부분 읽기, 새 쓰기, 퇴거까지 검증한 것이 아니다.

공식 경로는 pre-RoPE K를 저장하고 read 뒤 전체 K에 위치 처리를 적용한다.
첫 구현은 **pre-RoPE 복원 span만 유지**하며 RoPE·attention·현재 청크 쓰기·압축
주기는 그대로 둔다. QVG의8청크 span을 임의의3프레임 독립 압축으로 바꾸지 않는다.
압축 상태 교체, span 제거, 부분 덮어쓰기 시 보관값의 무효화 검사가 필요하다.

첫 검사에서는 같은 `flash_attn_func`에 합성 입력을 전달했다. 모델의 native
attention wrapper·RoPE·전체 생성 latent의 동등성을 확인한 것으로 표현하지 않는다.

## 5. 결정과 남은 작업

- QVG packed decoder의 현재 GPU 호환성: **통과**.
- 단순 유지의 출력 동등성: **합성 read/attention 범위에서 통과**.
- 성능 baseline: 중복 복원을 제거한 B로 확정. U도 보조 대조로 유지.
- 전체 생성의 메모리·시간 이득: **미측정**.
- H2 비용 모델 및 미사용 조건의 regret: **미측정**.
- 동일 코덱 fused attention과의 우열: **기준선 미확보, 주장 제외**.

다음은 [측정 설계](MEASUREMENT_PLAN_KO.md)의 생성 연결·정확성 게이트다.
복잡한 선택기는 아직 필요성이 확인되지 않았다.

## 재현 자료

- [실행 코드](../../../tools/kv_reuse_preflight.py)
- [원시 검사 결과·소스 지문](attempt_01.json)
- [예산 기록](budget.json)
- 실행: `~/sfq_data/py tools/kv_reuse_preflight.py`

실행기는 기존 예산 파일을 덮어 초기화하지 않는다. 완료된 검사 이후 재실행은
별도 등록과 출력 위치를 사용해야 하며, 본 검사 결과를 새 실험으로 덮어쓰지 않는다.
외부 코드는 checkout revision과 파일별 SHA256으로 식별한다. 기존 종료 연구의
실행 파일·결과는 수정하지 않았다.
