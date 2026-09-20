# 커널 트랙 지시서 (세션 7~9) — GPU 메모리·CUDA 단계에서 저비트 KV 캐시를 실제로 빠르고 작게

작성: 2026-09-18 · 선행: `RESULTS_session5_gates.md` §4(J), `HANDOFF_session6.md` §9(P), `STATUS_2026-09-18.md`
환경: **개발·정확성 = 로컬 RTX 3080 10GB(sm_86)**, **성능 측정·모델 통합 = RTX 4090 24GB(sm_89, 로컬 서버)**. 세션 6 배치가 4090을 쓰는 동안 3080에서 병행. 기준 커밋: Self-Forcing `f2e831e`, `kernels/triton_kv_quant.py`, `kernels/triton_fused_lowbit_attn.py`(v2).

세션 1~6은 "무엇을 몇 비트로 어떤 구조로"를 정했다(하이 레벨). 이 트랙은 그 결정이 GPU 안에서 바이트를 실제로 덜 움직이고 덜 계산하게 만드는 일이다. 논문 B(MLSys)의 몸통.

**출발 수치(세션 5).** fused v2 커널 8.17ms vs FA2 6.9ms(+13%, 목표 ≤5%). quantize 커널 0.31ms(9.2배). 종단 51.3s vs 기존 INT4 경로 70.1s(−27%), BF16 원본 대비는 P0에서. 기존 경로의 낭비: quantize 1,261 + dequantize 343 ms/3chunk, copy_ 354 + cat 200 ms(퇴출 복사). 현행 int8 저장 3.40GB → 패킹 INT4 1.89GB, INT2 1.13GB. 격자 어긋남(4,680 = 292.5블록) → 퇴출마다 재양자화, 비멱등.

**규칙.** 모든 커널은 3080에서 (1) 참조 구현 대비 bit-exact 또는 allclose(1e-2) 관문, (2) 합성 형상 마이크로벤치 후 4090으로. 4090 숫자만 논문에. 시간은 CUDA event 중앙값(워밍업 제외), 메모리 트래픽은 **ncu**(`dram__bytes_read/write.sum`, `lts__t_sectors_hit_rate`, `sm__warps_active`, `smsp__sass_average_data_bytes_per_sector_mem_global`). 매 단계 roofline 점 하나.

---

## T0. 측정 기반 (먼저, 4090 2h + 3080)
- ncu로 현재 네 경로(BF16 FA2 / 기존 dequant→FA2 / fused v2 INT4 / fused v2 INT2)의 커널별 DRAM 바이트·L2 hit·occupancy·달성 FLOP/s. 각 커널을 4090 roofline(165 TFLOPS, 1.0 TB/s) 위에 점으로.
- fused v2의 병목 규명: 언팩 정수 연산(issue-bound)인가, 스케일 로드인가, 레지스터 압박→occupancy인가, bank conflict인가. **이 답이 T1의 방향을 정한다.**
- 산출: `kern/T0_roofline.csv/.pdf`, `kern/T0_bottleneck.md`.

## T1. fused attention ≤ +5% (핵심)
- T0 결과에 따라: 언팩을 shared memory 단계로 이동 + `ldmatrix`로 텐서코어 직결 / `cp.async` 2~3단 파이프라인 / 스케일을 세그먼트당 1회 로드 / 레지스터 타일 재배치. Triton으로 안 되는 단계는 CUTLASS 또는 raw CUDA(mma.sync m16n8k16 bf16).
- K3(3비트) 레이아웃: 8값=3바이트 정렬 문제 → **T4 비트 평면**으로 해결(2비트 평면 + 1비트 평면). T1에서는 INT4·INT2 먼저.
- 관문: 세션 5 J2 관문 1~3 재통과. 목표 FA2 대비 ≤ +5%(INT4·INT2·K4V2). 미달 시 어디서 새는지 ncu로 상위 3개.
- 산출: `kernels/fused_lowbit_attn_v3.*`, `kern/T1_bench.csv`.

## T2. 세그먼트 링 버퍼 (복사 0, 재양자화 0)
- 캐시를 chunk 세그먼트(4,680토큰) 단위 페이지로. 퇴출 = 포인터 이동, sink = 고정 페이지. `copy_`/`cat` 트래픽 제거(−554ms/3chunk 목표), 격자 어긋남은 설계상 소멸(세션 6 K와 정합).
- 관문: BF16에서 A1·A2·A4와 latent bit-exact(구조 변경이 결과를 바꾸면 안 됨). INT4에서 세션 6 K2(β) 결과와 일치.
- 산출: `kernels/segment_cache.py`, `kern/T2_traffic.csv`(ncu 바이트 전후).

## T3. 양자화를 K/V projection GEMM epilogue에 융합
- 현재: GEMM → BF16 K/V 쓰기 → 읽기 → 양자화 → 패킹 쓰기. 목표: GEMM 레지스터에서 바로 패킹 쓰기, BF16 K/V 미실체화. CUTLASS epilogue visitor 또는 Triton matmul + 커스텀 epilogue.
- 그룹 스케일(16토큰)이 GEMM 타일 경계와 맞는지 확인(타일 M을 16의 배수로).
- 관문: 출력 패킹이 `triton_kv_quant`와 bit-exact. 트래픽: K/V 관련 DRAM 바이트 약 1/3(ncu).
- 산출: `kernels/proj_quant_epilogue.*`, `kern/T3_traffic.csv`.

## T4. 비트 평면 저장 — 읽는 시점에 정밀도 선택
- key를 2비트 기본 평면 + 1비트 보정 평면으로 분리 저장. 세그먼트별 플래그로 "3비트로 읽기(앵커·최근)" / "2비트로 읽기(오래된)" 선택. K3V2 레이아웃 문제 해결 + 세션 5 H5 앵커 정책이 저장 형식 하나로 구현됨.
- 관문: 3비트로 읽은 결과가 K3 RTN과 bit-exact, 2비트로 읽은 결과가 K2 RTN과 bit-exact(평면 분해가 정확히 RTN 격자와 맞아야 함 — 맞지 않으면 어긋남 크기 기록).
- 실험(4090): 슬롯 나이별 정밀도 스케줄 {전부 3, 앵커 3+나머지 2, 전부 2} 화질 vs 바이트.
- 산출: `kernels/bitplane_kv.*`, `kern/T4_schedule.csv`.

## T5. 압축을 속도로 — INT8 텐서코어 QKᵀ
- attention은 compute-bound(AI 4,680)이므로 저비트 key를 BF16으로 풀어 계산하는 것은 낭비. query를 즉석 per-block INT8로, key는 INT4/INT2→INT8 확장(시프트만), **QKᵀ를 INT8 mma**로, softmax·PV는 BF16 유지(value는 민감도 낮음 → PV도 INT8 시험 가능).
- 우리만의 근거: 세션 5의 층·슬롯별 key 민감도 + 스칼라 보정을 INT8 스케일에 흡수.
- 관문: BF16 FA2 대비 출력 allclose(1e-2) + 3프롬프트 생성 MUSIQ/VBench ±0.5. 성능: FA2 대비 QK 구간 시간, 전체 attention 시간(목표 −30% 이상).
- 산출: `kernels/int8qk_attn.*`, `kern/T5_bench.csv`, 층별 민감도 표.

## T6. 용량을 실물로 — 다중 스트림
- 4090 한 장에서 동시 프롬프트 N개(각 63f) — BF16 원본 vs 레시피(A4+λ1+K3V2, T1~T3 커널). N을 늘리며 OOM 직전까지: 최대 N, 합산 FPS, 스트림당 latency. 목표 그림: x=N, y=합산 FPS, 두 곡선.
- 산출: `kern/T6_streams.csv/.pdf`.

## 우선순위·예산
T0 → T1 → T2 → T3 → T5 → T4 → T6. T0·T1·T2가 논문 B 최소본, T5가 되면 강한 논문. 3080 개발 시간이 대부분(주당 저녁 기준 3~4주), 4090 측정은 단계당 1~3h.
**논문 B 성공 기준**: BF16 원본 대비 종단 latency ≤ +5%(T1~T3), KV 트래픽·메모리 ncu 실측 그림, 격자 어긋남 소멸 증명, (T5 시) attention 자체 가속, (T6) 동시 스트림 3~4배.

## 산출물
```
kernels/  fused_lowbit_attn_v3, segment_cache, proj_quant_epilogue, bitplane_kv, int8qk_attn, tests/, bench/
kern/     T0_roofline, T0_bottleneck.md, T1~T6 csv/pdf, NOTES_kernel.md (단계별 관문 통과·ncu 수치·미결)
```
