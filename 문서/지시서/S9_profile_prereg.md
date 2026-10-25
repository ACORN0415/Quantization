# S9 — P3 fused 경로 제한 진단(프로파일): 사전 등록 (작성 2026-10-25 20:3x KST, 실행 전)

근거: [S9 §8](../결과서/S9_cand1_diag.md) (창 12/21/36f 에서 fused 종단 오버헤드 +36/+52/+74%, 원인 미확정) · 리뷰 합의: "2번(프로파일)을 제한된 진단으로 먼저, 커널 수정은 그 뒤". **프로파일은 새 방법 기여가 아니다.** 산출물은 셋: (a) 시간이 많이 드는 위치, (b) 원인 증거, (c) 가능한 개선의 크기(상한).

## 0. 측정 원칙 (리뷰 반영)

- CUDA 이벤트는 **호출 전후** 시간에만 쓴다. 하나의 fused 커널 내부를 복원·bias·softmax·메모리 접근으로 나누지 않는다. 연산은 겹쳐 실행되므로 부분 비용을 합산하지 않는다.
- bias OFF/ON 차이는 "보정 추가의 **순증가 비용**" 으로 보고하며 "bias 실행 시간" 이라 부르지 않는다.
- 커널 내부 지표(메모리·연산 처리량, 레지스터/spill, occupancy)는 Nsight Compute 가 필요하다. **이 머신에서는 `ERR_NVGPUCTRPERM`(GPU 성능 카운터 권한 없음)으로 불가** — 관리자가 `NVreg_RestrictProfilingToAdminUsers=0` 을 설정하거나 root 로 실행해야 한다. 3부는 **사용자 조치 전까지 보류**.

## 1. 1부 — 실제 생성에서의 시간 비중 (torch.profiler, 호출 단위)

| 항목 | 고정값 |
|---|---|
| 팔 | BF16 · fused INT2 λ0 · fused INT2 λ1 · fused K4V2 λ1 (창 21f, A1) |
| 입력 | `prompts_eval_c1_3.txt` 첫 1개 · **36 프레임**(12 chunk; 창이 차고 퇴출이 5회 일어나는 구간 포함) · seed 0 · 디코드 포함 |
| 측정 | `pipeline.inference` 전체를 `torch.profiler`(CPU+CUDA)로 감싸 커널별 self device time 집계. 분류: `flash`(FA2) / `fused_lowbit`(fused attention 커널) / `quant_pack`(패킹·양자화 커널) / `gemm` / `vae`(conv·upsample) / `elementwise·norm·softmax` / `other`. 분류 규칙은 커널 이름 문자열이며 결과 JSON 에 원 이름을 남긴다 |
| 보고 | 팔별 총 device ms 와 분류별 ms·%. **fused − BF16 의 증분**을 분류별로 분해: attention 커널 증분 / 패킹 증분 / 그 외 증분. 종단 wall 과 device 합의 차(CPU·launch·동기화)도 적는다 |
| 상한 산정 (c) | attention 커널이 fused 팔 device 시간의 p% 이고 FA2 대비 증분이 Δ ms 면, **완벽한 커널이 얻을 수 있는 종단 이득의 상한 = Δ / wall**. 이 수치가 작으면 커널 투자는 접는다 |

## 2. 2부 — 대표 shape 커널 비교 (S7 B4 하네스 확장, 한 프로세스·한 타이밍)

| 항목 | 고정값 |
|---|---|
| shape | B=1, H=12, D=128, chunk 4,680 · 창 ∈ {12, 21, 36, 63}f = {18,720 / 32,760 / 56,160 / 98,280} 토큰 (self 는 BF16) |
| 팔 | FA2 bf16 · fused INT2 λ0 · fused INT2 λ1 · fused K4V2 λ0 · fused K4V2 λ1 · **복원 → FA2**(packed → bf16 복원 시간과 FA2 시간을 **따로** 잰 뒤 합: 복원은 하네스의 `unpack_reference`(torch) — 순수 torch 복원이므로 복원 비용의 **상한**) · 패킹(`triton_quantize_kv`, chunk 1개, 쓰기 비용) |
| 방법 | warm-up 10, 반복 30, CUDA 이벤트 중앙값(B4 와 동일). 동일 난수 입력. 출력 정확도: fused vs 복원→FA2 상대오차 기록(B4 J2 와 같은 성격, 상한 아님) |
| 보고 | 창별 ms 표, FA2 대비 %, bias 순증가, 창 길이에 대한 기울기(토큰당 ms). **창이 길수록 FA2 대비 비율이 어떻게 변하는지** — §8 의 종단 추세와 대조 |

## 3. 판정 규칙 (결과 전 고정)

- (a) 위치: 1부에서 fused − BF16 증분의 **과반**이 어느 분류에 있는가(attention 커널 / 패킹 / 그 외). 과반이 attention 커널이면 2부가 원인 후보를 좁히고, 그 외면 커널 밖 문제(하네스·동기화·복원 버퍼)다.
- (b) 증거: 2부에서 (i) fused λ0 vs 복원→FA2 — fused 가 복원→FA2 보다 느리면 "fused 설계 자체가 복원 후 FA2 보다 손해" (ii) bias 순증가 (iii) 창 기울기.
- (c) 크기: 1부의 상한 공식. **상한이 종단 10% 미만이면 커널 투자 종료**를 권고한다(투자 기준은 §3.6 과 같은 10%).
- 이 진단으로 "원인은 X 다" 를 확정하지 않는다. 커널 내부 지표(3부)가 없으므로 (b) 는 호출 단위 증거다.

## 4. 상한·산출물

- 1부 4 팔 × 1 프롬프트 36f ≈ 10 분, 2부 ≈ 10 분. 상한 **2 시간**.
- 도구: `inference.py` 에 `TORCH_PROFILE_OUT` 환경변수 래퍼(코드 경로 불변), `kernels/bench/b9_kernel_sweep.py`, 분석 `Quantization/tools/analyse_profile.py`. 결과 `results/session9/profile/`, 결과서 S9 §9.
