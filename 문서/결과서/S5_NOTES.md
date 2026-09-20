# 세션 5 NOTES (진행 중 — 완료 시 정리)

## §0 시작 확인 (완료)
- 커밋 45836f4 / 226c405 = git log 일치. patches/README 재현: 33593df+01+02+20,
  b4c0936+21 → **작업 트리와 파일 단위 diff 0** (임시 worktree로 검증). 프롬프트 md5 일치.
- 회귀 관문: 플래그 OFF A1_int4 21f×3p vs 세션 3 latent **21/21 bit-exact**.
  legacy TUM(λ=1, self 포함) 21f×3p vs closeout_tum_int2 **21/21 bit-exact**.

## 발견 1 — "회복 +10.40"은 낙폭 지표다 (H의 전제 수정)
감사 B8의 지표는 `analyze_batch2.py`의 `delta()` = **MUSIQ(c20) − MUSIQ(c0)** (낙폭)이다.
같은 per-prompt 데이터로 재계산:
- tum_paper(λ=1) vs A1_int2: 낙폭 개선 +10.4 ✓ 재현. 그러나 **21-chunk 평균 MUSIQ는
  −11.08 ± 5.55 (t=−6.32, 부호 1/10)** — 평균 화질로는 유의하게 해롭다.
- 원인: 보정이 **chunk 0을 파괴**한다 (c0: 무보정 70.6 → λ=1 61.1 → λ=4 45.6).
  시작점이 낮아져 낙폭이 "개선"된 것.
- λ=4(구 +28.25)도 평균으로는 −11.5. **TUM 보정의 기존 서사("54~68% 회복")는 낙폭
  지표에서만 성립하며 평균 지표로는 성립하지 않는다** — 논문 기술 시 반드시 병기.

## 발견 2 — bias가 BF16-clean self 블록에도 적용되고 있었다
`delta_sq`는 캐시 전체를 덮지만 self(현재 chunk)는 항상 BF16으로 fresh 기록 후
읽힌다(C11). 즉 노이즈 없는 키에서 bias를 빼 왔다. chunk 0은 창 전체가 self라
λ에 비례해 파괴되는 관측과 정합. → 22_bias_modes의 기본값은 **self 제외**
(`--tum_include_self`로 legacy 재현).

## provenance 주의 (gateF2 재생성본)
`results/gateF2/*`(09-17 새벽 재생성)는 **수정 전 코드(λ=4 상당)**의 산물
(낙폭 +28.3 재현으로 확인; session5-start 커밋은 16:03에 생성됨).
λ=1은 `audit/rerun/tum_paper_int2`가 정본. 기존 무료 λ점: λ=0(tum_zero),
λ=1(tum_paper), λ=4(gateF2) — 전부 self-포함(legacy) 곡선의 점들.

## 22_bias_modes 구현 (완료, 관문 통과)
- `--tum_lambda` (bias 배수), `--tum_mode {taylor, exact_uniform, emp_channel_chunk,
  emp_group_token, emp_full}`, self 제외 기본값, `--tum_mass_chunks`(보정 **후**
  logit의 slot별 mass 로깅 — 감사가 지적한 F2 측정 공백을 메움).
- exact_uniform은 log(sinh x/x)의 8항 급수(각 항이 bmm) + 16토큰 블록 중복 제거.
  급수 오차 x≤2.5에서 4.4e-4, 수렴 위반 상계는 EXACT_X_TAIL로 로깅.
- emp_*: 양자화 직후 ε=k−k̂의 2차 모멘트를 3개 입도로 quant_state에 저장.

## J1 (완료, 관문 통과)
`kernels/triton_kv_quant.py` — 16토큰 블록 RTN sym + 진짜 패킹(INT4 2개/byte,
INT2 4개/byte). **UCSD 경로와 bit-exact**: codes/scales/dequant 모두, 실제 캐시
형상(32760, pad 8) 포함 6/6 케이스. bf16 연산 순서와 round-half-even까지 재현.

## J2 커널 (관문 1~3 통과)
`kernels/triton_fused_lowbit_attn.py` — FA2 구조 online softmax, packed 캐시를
레지스터에서 역양자화 + BF16 self 2원 소스, 선택적 in-kernel Taylor bias(패킹
영역만 = self 제외와 동치). K/V 비트 분리(K4V2 지원), chunk 단위 세그먼트
블록격자(퇴출과 정합, 재양자화 불필요 — RTN 멱등이라 의미 동일).
- gate1 BF16 passthrough vs FA2: max|diff| 9.8e-4 PASS
- gate2 INT4/INT2 vs dequant→FA2: 2.4e-4 / 4.9e-4 PASS
- gate3 bias ON vs 명시 fp32 경로: 2.4e-4 PASS
- 세그먼트 모드 vs per-chunk dequant→FA2: 2.4e-4 PASS

## J3 통합 (구현 완료, 관문 4·측정 대기)
`--attn_backend fused` + `--fused_bias_lambda`. 패킹 세그먼트 스트리밍:
chunk 확정 시(다음 chunk 진입) 1회 양자화, 오래된 비-sink 세그먼트 퇴출,
BF16 캐시 버퍼는 첫 호출에서 해제(메모리 주장 측정용).

## 지시서 개정 (HANDOFF_session5_1) 반영
- 관문 H 전면 교체: 4배 격차의 (a)비균일/(b)전파오염/(c)평균 배분 + 원인 겨냥 보정 + H5 앵커 관리.
- 사용자 지시: 기존 FR 결과 재사용, TF sweep부터 이어감. 파일명은 새 목록에 맞춰 정리.
- **사전 보고 사항(중요)**: 지시서 전제(λ*_FR≈4)는 낙폭 지표 기준. 평균 MUSIQ로는
  λ=1·λ=4 모두 무보정보다 −11 나쁨. self-포함 bias에서 λ↑ → chunk0 파괴 → 낙폭
  기계적 개선. 따라서 "4배 격차"의 일부/전부가 지표×구현 아티팩트일 가능성.
  sweep은 self-제외로, 판정은 평균 1차·낙폭 병기. self-제외 λ1 vs λ4 파일럿이
  이 가설을 직접 판정 → 그 결과로 (a)(b)(c) 배분 실험 규모 결정.
- 시간: 지시서 추정(보정 12~15분/런)의 실측치는 56분/런(63f×10p). 축소 규칙 즉시 적용.
- TF 지원: teacher_forced.py에 --tum_* 플래그 추가(ref 패스는 quantizer 없음 → 자동 무영향).
  TF λ=0도 explicit 경로로 통일(경로 효과 페어링).
- H3 ε통계: causal_model에 EPS_STATS(σ²_emp/(Δ²/12)·평균·커토시스, 슬롯별) +
  inference --kv_eps_stats. (a)의 직접 증거 = var_ratio, (c) = |mean|/σ.

## 발견 3 — self 제외가 TUM의 부호를 뒤집는다 (파일럿, 판정: 분기 A)
self-제외 λ sweep 파일럿 (A1 INT2, 63f×10p, vs 무보정 A1_int2, paired):

| | 평균 MUSIQ | 낙폭 | c20 post-bias self mass |
|---|---|---|---|
| λ=1 self제외 | **+10.04 (t=+3.27, 8/10)** | +5.27 (t=2.14) | 37.1% |
| λ=4 self제외 | **+12.12 (t=+8.14, 10/10)** | +33.94 (t=23.7) | **73.7%** |
| λ=1 legacy(self포함) | −11.08 (t=−6.32) | +10.40 | — |
| λ=4 legacy | −11.94 (t=−4.91) | +28.25 | — |

- **self 오적용이 평균 기준 약 21~24pt를 파괴하고 있었다.** 제외하면 TUM 보정은
  평균 지표에서도 진짜 회복을 준다(λ=1 +10.0). 판정 규칙 → H 진행.
- λ=4 > λ=1은 self-제외 후에도 유지(+12.1 vs +10.0) → (a)(b)(c) 배분 실험 유효.
- **주의**: λ=4의 self mass 73~87%는 BF16(~48%)을 크게 초과 — 분포를 "되돌리는"
  게 아니라 과보정으로 오래된 슬롯을 억제하는 영역. λ=8 결과와 H6로 판별 예정.
- INT2 피해(−29.2) 대비 회복률: λ=1 34%, λ=4 41%.

## 남은 실행 큐
1. (진행 중) H2/H3 스모크 → H1 파일럿 self제외 λ∈{1,4} 63f×10p
2. 파일럿 분석 → H1 격자 확정(legacy 3점 재사용) → 잔여 sweep
3. J0 마이크로벤치(`kernels/bench/j0_microbench.py`) — GPU 빈 틈에
4. J3 관문 4(3프롬프트 63f fused vs dequant, MUSIQ ±1) + 통합 측정
5. H2/H3 본 런 → I 격자(A4+보정 우선)

## H1 λ sweep 완성 — λ*_TF=1, λ*_FR=1~4 플래토, "4배 우위"는 아티팩트로 판정
FR(self-제외, 평균 MUSIQ Δ vs A1_int2, n=10): λ0.5 +7.09(t=2.75) · **λ1 +10.04(t=3.27)**
· λ2 +9.33(t=5.41) · λ4 +12.12(t=8.14) · λ8 +11.16(t=6.81).
직접 대비: **λ4−λ1 = +2.08 (t=0.69, n.s.)**, λ8−λ4 = −0.96 (t=−4.04, 유의 하락).
TF(오염 0, vs TF-λ0): **λ1 +4.69 (t=2.82, 9/10)** · λ4 +0.22 (n.s.) ·
**λ4−λ1 = −4.46 (t=−3.02, λ4가 유의하게 나쁨)**.

판정:
1. **λ*_TF = 1** — 순수 양자화 노이즈에는 논문/테일러 스케일이 정확. (a) 비균일
   양자화 오차의 "λ 부풀림" 기여 ≈ 0 (ε통계 var_ratio로 교차확인 예정).
2. **λ*_FR = 1~4 구별 불가(플래토)**, 8부터 하락. 감사 B8b의 "4배가 논문값보다
   유의하게 낫다(t=−3.46)"는 self-포함 × 낙폭 지표의 아티팩트였음이 확정.
3. FR 회복(+10.0)이 TF 회복(+4.7)의 2배 — 보정이 (b) 전파 오염의 누적 되먹임도
   끊어준다는 해석과 정합. FR이 과보정(λ2~4)에 관대한 것도 (b)와 정합(어차피
   오염된 옛 슬롯 억제가 덜 해로움).
4. **λ* = 1 확정** (규칙: 평균 1차). 근거: TF 최적, FR 수치 피크(λ4)와 통계적
   동률, 분포 복원 관점에서 λ1만 정당(c20 self mass: 무보정 19.2% → λ1 37.1% →
   BF16 47.8% 방향 복원; λ4는 73.7%로 과억제). I·J에서 λ=1 고정.
산출물: gateH_lambda_sweep.csv/.png

## J0 마이크로벤치 (batch2, CUDA event 중앙값, 레이어콜 기준)
(a) FA2 BF16 6.91ms · (b) dequant→FA2 7.44ms(+7.6%) · (c) explicit fp32 39.6ms ·
(e) quantize python 2.81ms → **Triton 0.31ms (9.2×)**, chunk 단위 0.057ms.
메모리(30레이어 KV): BF16 6.04GB → int8(현행) 3.40GB → **packed INT4 1.89GB → INT2 1.13GB**
— "5.6→1.4GB" 주장의 첫 실측(6.04→1.89/1.13).

## J2 커널 v1→v2
v1(gather 언팩) 43.9ms — 실패. 원인 상위 3: ① 바이트 중복 gather(채널당 PACK회),
② 토큰별 scale gather, ③ autotune 부재. v2 = coalesced 바이트 로드 + 산술 확장
(reshape가 채널 순서와 일치하도록 패킹 설계), 세그먼트-정렬 타일로 scale을
[BN/16, D] 구조 로드. 스윕 결과 **8.17ms (BM128/BN32/w4/s2) = FA2 +13.1%**.
> ⛔ 비율 대체: 단일 하네스 재측정에서 **+26.6%**(세션 7 B4). 절대 시간은 불변.
- J3 성공 기준(≤+5%)은 미달. 단 종단 기준으로는 양자화 경로끼리 비교 시
  현행(FA2+dequant+quantize ≈ 10.2ms/콜)보다 **빠름**. 실용 대안도 성립:
  "packed 저장 + Triton quantize + 호출시 dequant→FA2"는 지연 +8%에 상주 KV 1.89GB.
- v1 J3 생성 런(fused_int4/int2): 145~166s/프롬프트(dequant 경로 70s) — v2로 재측정 예정.
- fused 백엔드는 v2(BM128/BN32/w4/s2)로 전환함.

## 시간 기록
- TUM explicit 63f×10p 1런 ≈ 48분(taylor). H1 원안 17런은 불가 → legacy 재사용
  + self제외 신규런으로 축소.


## 세션 종료 기록 (2026-09-18 14:00)

배치 9개 완료(계획 3 + 결과가 요구한 6). 실행한 GPU 런 약 45개, 총 ~26시간.

### 관문 판정 최종
| 관문 | 판정 |
|---|---|
| 회귀(플래그 OFF / legacy / chunk0 self제외 / BF16 역학) | 전부 PASS (21/21, 21/21, 3/3, 63/63) |
| H1 λ sweep (FR 5점 + TF 3점) | 완료, λ*=1 |
| H3 ε 통계 + emp 모드 | 완료 — emp 무효, (a)(c) 기각 |
| H4 flatten 컨트롤 | 완료 — Jensen 구조 기여 0 |
| H6 분포 복원 | 완료 — 감사 F2 공백 종결 |
| H5-a 오라클 | 완료(정책/계보 분리), H5-b/c/d/e 미실행 |
| I 1·2단계 | 완료 |
| J1 bit-exact | PASS 6/6 |
| J2 등가 1~3 + 세그먼트 | PASS |
| **J3 관문 4(±1)** | **FAIL (+1.50/+1.52)** — 원인 규명, fused가 더 정확한 쪽 |
| J0 마이크로벤치·메모리 | 완료 |

### 이 세션에서 잡은 버그 4건
1. (감사 코드) bias의 self 슬롯 오적용 → 평균 −21pt 파괴 [버그 E]
2. (감사 방법론) 낙폭 지표가 chunk0 파괴를 보상으로 읽음 [버그 E]
3. (내 커널) 세그먼트 경계 스케일 행 언마스크 → 0×inf = NaN
4. (내 통합부) fused 상태가 프롬프트 간 누출 — 감사 버그 C와 동형

### 미실행(의도적, 축소 규칙)
H2 본런, H5-b/c/d/e, I 126f·LongLive, J4(5090). 근거와 우선순위는
RESULTS_session5_gates.md §8.

---

# 다음 세션을 위한 재사용 안내 (세션 5 종료 시점)

## 커밋 해시 (정본)
| 저장소 | 해시 | 내용 |
|---|---|---|
| Self-Forcing | **`1647122`** | 세션 5 최종 (45836f4 + 커밋 9개) |
| kv-quant-longhorizon | **`650ce90`** | 226c405 + KIVIK_RTNV 혼합 양자화기 |

세션 5 커밋 순서 (45836f4 위):
```
6e501dd 22_bias_modes        λ·모드·self제외·mass/ε 로깅·TF 플래그
cfe5f64 24_attn_backend      Triton pack + fused v1 + --attn_backend
e646d63 25_oracle_anchor     oracle_anchor.py
c101daf J2 kernel v2         gather 제거, 8.17ms
f2e831e oracle BF16 모드     + batch4/5 스크립트
75607ae H4 flatten 컨트롤    + J3 진단 배치
d3541d6 J2 v2 NaN 수정       세그먼트 경계 스케일 마스크
a4cf404 J3 fix 2            fused 상태 프롬프트별 리셋
1647122 batch6 스크립트
```
패치본: `~/gpu/patches/{22,24,25,26}_*.patch`(Self-Forcing), `23_kv_grid.patch`(kv-quant).
**worktree에 순서대로 적용해 HEAD와 파일 단위 일치 검증 완료.**

## 재사용할 코드 경로 (파일:줄)

### 격자 정렬 / 세그먼트 (미결 2순위 "격자 계보 효과"에 그대로 필요)
- **세그먼트 단위 양자화·퇴출**: `wan/modules/causal_model.py:36` `_fused_quantized_attention()`
  - chunk 확정 시 1회 양자화: `:68-71` (`pending` → `triton_quantize_kv`)
  - 오래된 비-sink 세그먼트 퇴출: `:78-82` (`segs.pop(n_sink)`)
  - 커널 호출(세그먼트 격자 전달): `:95-99` (`seg_len=chunk_tokens, seg_nb=seg_nb`)
- **비교 대상인 모델의 이동-재양자화**: `wan/modules/causal_model.py:396-400`
  (`num_evicted_tokens` 만큼 좌측 시프트 → 이후 `quantize_kv`가 8토큰 어긋난 격자에서
  재양자화. 4,680 mod 16 = 8이 원인.)
- **패킹·스케일 레이아웃**: `kernels/triton_kv_quant.py:88` `triton_quantize_kv()`,
  블록 상수 `:30` `BLOCK_TOKENS=16`, 참조 역패킹 `:110` `unpack_reference()`
- **커널의 세그먼트 스케일 로드(마스크 필수)**: `kernels/triton_fused_lowbit_attn.py:255-261`
  (`row_ok = srows < n_segs*SEG_NB` — 빼면 0×inf=NaN)

격자 정렬 효과만 따로 재기 원한다면 오라클 드라이버의 `oldest` 정책이 가장 싸다
(모델 rolling과 동일 내용, 재양자화 계보만 다름). 세션 5 측정: INT4 +1.29, INT2 +2.81.

### TF(teacher-forced) 드라이버
`teacher_forced.py` — lockstep ref/work, 매 chunk 재앵커.
- 재앵커·양자화 지점: `:63` `anchor_work_cache()`
- 보정 플래그(세션 5 추가): `--tum_correct --tum_mode --tum_lambda --tum_include_self`
  (ref 패스는 quantizer가 없어 자동으로 무영향)
- 실행 래퍼: `run_gateH1_tf.sh <tag> <bits> <lambda>` (10프롬프트, latent 덤프 →
  `decode_latents.py`로 비디오 → `metrics_all.py`로 MUSIQ)
- 용도: 오염 0 조건에서의 보정 최적 λ, (a)와 (b)의 분리.

### 오라클 앵커 드라이버
`oracle_anchor.py` — lockstep BF16 기준 + 양자화 free-running, 오염 기반 앵커 재선택.
- 오염 점수: `:216` (`work chunk K/V` vs `ref chunk K/V` 상대 L2)
- 정책 분기: `:225` (`oracle` = 오염 최소 N개 + 최신순, `oldest` = 순수 rolling 컨트롤)
- 캐시 재구성·재양자화: `:89` `set_cache_contents()`
- 플래그: `--policy {oracle,oldest} --kv_quant {RTN,BF16} --kv_bits --n_anchors --tag`
- **역학 관문**: `--policy oldest --kv_quant BF16`이 A1_bf16과 63/63 bit-exact여야 함
  (드라이버 자체가 모델 rolling과 동일함을 보증). 반드시 먼저 돌릴 것.

### 보정 관련 플래그 (inference.py)
```
--tum_correct                 보정 on (명시적 fp32 attention 경로, FA2 대비 ~6배 느림)
--tum_mode {taylor|exact_uniform|emp_channel_chunk|emp_group_token|emp_full}
--tum_lambda <float>          bias 배수 (λ*=1 확정)
--tum_include_self            legacy(세션4/감사) 재현용 — 기본은 self 제외
--tum_flatten_slots           구조 제거 컨트롤 (캐시 평균 down-weight만)
--tum_mass_chunks 1 6 20      보정 후 slot별 mass 로깅 → <stats_dir>/tum_mass.json
--kv_eps_stats <path>         ε 실측 통계(분산비·평균·커토시스) → JSON
--kv_eps_chunks 1 6 20
--attn_backend {auto,fused}   fused = 패킹 스트리밍 캐시
--fused_bias_lambda <float>   in-kernel bias (패킹 영역만 = self 제외와 동치)
--kv_quant KIVIK_RTNV         키 KIVI asym + 값 RTN sym (kv-quant 23 패치)
```

### 분석 도구
- `~/gpu/Quantization/tools/metrics_all.py` — MUSIQ/LPIPS 등. `--num_latent_frames`로
  63/126 구분, `--skip lpips clip flow`로 MUSIQ만.
- `~/gpu/Quantization/tools/analyze_gateH.py` — 평균·낙폭 두 지표 짝지은 t + mass 요약.
- `~/gpu/audit/compare_replication.py` — latent bit-exact 대조.
- `kernels/tests/test_quant_pack.py`, `kernels/tests/test_fused_attn.py` — 커널 관문.
- `kernels/bench/j0_microbench.py` — 경로별 지연 + 메모리 표.

## 평가 지표 환경 (중요)
- **VBench: 미설치.** `pyiqa 0.1.16`(MUSIQ)와 `openai-clip`만 있음. `lpips`·`decord`
  패키지는 없음(`metrics_all.py`의 lpips는 torchvision 경로로 동작, flow는 RAFT 필요).
- 따라서 현재 품질 주장은 전부 **MUSIQ 단일 지표**다. 세션 1 NOTES의 미결
  ("품질 주장에는 다른 지표가 필요")은 **여전히 미해결**이며, VBench 설치가
  다음 세션의 선행 과제 후보다(설치 시 디스크·의존성 확인 필요 — 현재 여유 54GB).
- MUSIQ의 한계가 이번에 실제로 드러남: self mass 37~74% 구간에서 둔감(λ1~4 동률).

## 미결 (우선순위, SESSION6_START.md와 동일)
1. **J3 관문 4 재설계** — "dequant 경로와 등가"라는 전제가 틀렸다(fused가 +1.5로 더
   정확). 새 기준은 BF16 대비 품질 또는 격자 계보를 통제한 대조군.
2. **격자 계보 효과 단독 논문화** — 재양자화 격자 정렬만으로 INT2 +2.81. 구현 완비.
3. H5-c 배포 가능 프록시(슬롯 나이/키 통계) + H5-e 결합(보정+앵커).
4. I 확장: 상위 2설정 126프레임, LongLive v1.0 재현.
5. H2 exact_uniform 본런 (급수 구현·정확도 검증 완료, 품질 런만 남음).
6. J2 커널 +13.1% → +5%: BM/BN autotune, num_stages, self 페이즈 병합.
7. J4 5090/sm_120 컴파일 점검.
8. (지표) VBench 설치 및 상위 설정 재평가.

## 함정 기록 (같은 실수 반복 방지)
- 낙폭 지표(c20−c0) 단독 사용 금지 — chunk 0 파괴를 보상으로 읽는다.
- `results/gateF2/*`는 수정 전 코드(λ=4 상당) 산물. λ=1 legacy 정본은
  `audit/rerun/tum_paper_int2`.
- Triton에서 세그먼트 경계를 넘는 로드는 반드시 마스크(0×inf = NaN).
- 캐시에 새 상태를 달면 `pipeline/causal_inference.py`의 프롬프트별 리셋에 추가할 것.
- 새 quantizer/드라이버는 "무양자화에서 기존 경로와 bit-exact" 관문을 먼저 통과시킬 것.
