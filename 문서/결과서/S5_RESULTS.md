# 세션 5 결과 — §4 후속 완결 (KIVI residual window · 126프레임)

> ⚠ **이 문서는 세션 5 시점 스냅샷이다.** 세션 6·7에서 아래 주장들이 철회·축소됐다:
> C-I(실측 ε 균일·ε=0) **철회** · C-D(구조는 INT4에서만) **철회** · C-R(오라클 정책) **분해** ·
> C-M(0.75GB 8×) **철회, 실측 4.00×** · C-J/C-B/C-H/C-L **범위 축소** · C-A **n=3 예비**.
> **먼저 `CORRECTIONS_2026-09-19.md`를 읽고 해당 항목을 대체해서 읽을 것.**


작성: 2026-09-17 · 시작 문서: `SESSION5_START.md` · 선행: `RESULTS_session4.md`, `~/gpu/audit/AUDIT_REPORT.md`

## 0. 시작 상태 검증

- Self-Forcing `45836f4`, kv-quant-longhorizon `226c405` — SESSION5_START.md 기준 커밋과 일치.
- long126 `a4_int4`는 세션 4에서 **OOM으로 실패**했던 건(외부 프로세스가 GPU 9.9GB 점유 중 VAE decode에서 사망). GPU가 빈 상태에서 재실행했고, 재실행 prompt000이 실패 전 부분 latent와 **42/42 bit-exact** — 결정성 유지 확인.

## 1. §4-1 KIVI residual window (63f, 10프롬프트, MUSIQ paired)

설정: `kivi{4,2}_rw{0,1ch}` — rw0은 세션 1~3이 측정한 형태(residual 없음), rw1ch는 논문 사양(최신 chunk 4,680 토큰을 BF16 유지). 산출물 `results/kivi_rw/`, `results/kivirw_metrics*.{csv,json}`.

### 수정된 KIVI는 같은 비트에서 RTN을 이깁니다 (10/10)

vs BF16, 21 chunk 평균, paired over 10 prompts:

| 설정 | Δ MUSIQ | t | 부호 |
|---|---:|---:|---:|
| RTN INT4 | −4.61 ± 4.22 | −3.46 | 0/10 |
| **KIVI4 rw0** | **−2.15 ± 1.96** | −3.48 | 0/10 |
| KIVI4 rw1ch | −2.01 ± 2.08 | −3.05 | 0/10 |
| RTN INT2 | −29.23 ± 6.76 | −13.68 | 0/10 |
| **KIVI2 rw0** | **−16.39 ± 4.51** | −11.49 | 0/10 |
| KIVI2 rw1ch | −15.91 ± 4.41 | −11.40 | 0/10 |

직접 비교(paired): KIVI4 vs RTN INT4 **+2.60 (t=+3.82, 10/10)**, KIVI2 vs RTN INT2 **+13.32 (t=+7.17, 10/10)**. chunk 20 끝점 기준으로는 INT2에서 18.0 → 41.4로, 붕괴 깊이가 절반 이상 줄어듭니다.

세션 1의 "zp-clamp 수정 후 KIVI key 오차 < RTN"(`issue/ISSUE_zp_clamp.md`)이 **화질 단위로 확인**된 것입니다. 세션 4의 "키가 단독 원인" 결론과도 정합 — KIVI의 key per-channel 그룹화가 키 오차를 줄이는 만큼 그대로 화질로 돌아옵니다.

### residual window 자체는 사실상 무효입니다

paired rw1ch − rw0:

| | 21 chunk 평균 | 마지막 5 chunk |
|---|---:|---:|
| KIVI4 | +0.14 ± 0.52 (t=0.84) | +0.54 ± 1.85 (t=0.92) |
| KIVI2 | +0.47 ± 0.65 (t=2.32, 8/10) | +1.19 ± 2.45 (t=1.53) |

INT4에서는 무의미, INT2에서 경계선(+0.47, 전체 결손 −15.9의 3%). **이유는 구조적입니다** — Self-Forcing은 self(현재 chunk)를 항상 BF16으로 읽으므로(기록→attention→재양자화, 감사 C11), 논문이 residual window로 보호하려는 "최신 토큰"의 대부분이 이미 보호되고 있습니다. rw1ch가 추가로 지키는 것은 직전 chunk 하나뿐이고, 세션 4 관문 G에서 최신 chunk의 mass 비중은 크지만 그 하나의 양자화 오차가 전체 붕괴에서 차지하는 몫은 작습니다. chunk 0~3 구간은 rw1ch==rw0 bit-exact(캐시가 residual 범위를 넘기 전).

**업스트림 구현이 residual window를 생략한 것은 이 셋업에서는 결과에 거의 영향이 없었다**가 결론입니다. 단, 이는 self-BF16 구조 덕이므로 일반 LLM 셋업으로 외삽하면 안 됩니다.

## 2. §4-2 126프레임 (42 chunk, 5프롬프트)

질문(세션 4 원안): rolling 캐시를 그대로 두고 rollout만 2배로 늘리면 INT4 피해가 chunk 20 이후 **계속 자라는가, 평탄해지는가**. 설정: A1(기본)·A4(캐시 구조 변경) × BF16/INT4, `prompts5`, seed 0.

### BF16은 42 chunk 내내 평탄합니다

a1_bf16 chunk 0~20 평균 68.6 → 21~41 평균 67.9 (a4_bf16도 동일 양상). 모델 자체는 126프레임에서 무너지지 않으므로, 아래의 하락은 전부 양자화 몫입니다.

### INT4 피해는 평탄해지지 않고 계속 자랍니다

per-chunk MUSIQ (5프롬프트 평균):

| chunk | a1_bf16 | a1_int4 | a4_bf16 | a4_int4 |
|---:|---:|---:|---:|---:|
| 0 | 67.2 | 67.2 | 67.2 | 67.2 |
| 10 | 69.7 | 61.3 | 68.9 | 67.3 |
| 20 | 67.6 | 55.3 | 66.8 | 62.7 |
| 30 | 67.1 | 49.7 | 69.1 | 57.4 |
| 41 | 67.4 | **43.2** | 68.3 | **56.0** |

a1_int4의 피해(vs 자기 bf16, paired over 5 prompts)는 chunk 0~20 구간 −5.97(t=−2.31)에서 21~41 구간 **−18.13(t=−11.85)**로 3배가 됩니다. 기울기는 chunk 10 이후 거의 일정(약 −0.6/chunk) — **무릎도 포화도 없이 선형으로 계속 내려갑니다**. 세션 1 미결 5번("INT2 곡선이 chunk 20에서도 오르는 중")의 INT4 버전 답: INT4도 horizon만 늘리면 같은 운명이고, 63f에서 "INT4는 바닥선 위 소폭"으로 보인 것은 horizon이 짧아서였습니다.

### 캐시 구조(A4)의 보호는 장기에서 더 커집니다

A4의 피해는 같은 구간에서 −2.46 → −8.15로, A1의 절반 이하 속도로 자랍니다. 피해 차(Δ of Δ, paired):

| 구간 | A4가 A1보다 덜 잃는 양 | t | 부호 |
|---|---:|---:|---:|
| chunk 0~20 | +3.51 ± 4.37 | +1.80 | 5/5 |
| chunk 21~41 | **+9.98 ± 5.40** | +4.13 | 5/5 |
| 전체 | +6.75 ± 1.23 | **+12.24** | 5/5 |

세션 3 C7("캐시 구조만으로 INT4 피해 70% 회복", 63f)의 회복률은 126f 전체 기준 약 56%(late 구간 55%)로 다소 줄지만, **절대 이득은 2배 이상**(chunk 41에서 43.2 vs 56.0, +12.8pt)이 됩니다. 구조 이득은 장기에서 소멸하지 않고 커집니다 — 다만 A4도 하락 중이므로 "구조만으로 충분"은 아니고 하락 속도를 늦출 뿐입니다.

### latent 곡선의 비대칭 (해석 주의)

rel_err(vs 자기 BF16, 5프롬프트 평균): a1_int4는 chunk 41에서 **1.04**로 63f 카오스 바닥(~0.74)을 넘어 계속 오르는 반면, a4_int4는 0.67~0.75에서 **멈춥니다**. 바닥을 넘는 rel_err는 "다른 궤적" 이상 — norm 자체가 어긋나며 분포를 이탈한다는 신호로, a1_int4의 MUSIQ 붕괴와 정합합니다. a4_int4는 궤적은 다르지만 분포 안에 머뭅니다. (rel_err는 바닥 위에서 화질을 말하지 못하므로 순위 주장은 MUSIQ 기준.)

그림: `results/long126_musiq.png`

## 3. 세션 4 결론에 붙는 것

| 세션 3~4 | 세션 5 |
|---|---|
| C7: 캐시 구조만으로 INT4 피해 70% 회복 (63f) | 유지·강화 — 126f에서 회복률 ~56%로 줄지만 **절대 이득은 +12.8pt로 2배 이상**, 5/5 |
| 63f에서 "INT4는 바닥선 위 소폭" (C1) | **horizon 한정** — 126f에서 INT4도 선형 하락 지속 (chunk 41에서 −24pt), 포화 없음 |
| C4: zp-clamp 수정 후 KIVI key 오차 < RTN | **화질로 확인** — KIVI4 vs RTN4 +2.60(10/10), KIVI2 vs RTN2 +13.32(10/10) |
| 논문 사양 residual window의 기여 (미측정이었음) | **사실상 0** — self-BF16 구조가 역할을 이미 흡수. 업스트림의 생략은 이 셋업에선 무해 |

## 4. 산출물

```
results/long126/a4_int4/            재실행 (videos, stats.json, run.log)
results/latents_long126/a4_int4/    5프롬프트 × 42 chunk + full.pt
results/long126_curves_5p.csv       5프롬프트 전체로 재계산한 rel_err 곡선
results/long126_curves_5p_per_prompt.json
results/long126_musiq.csv           per-chunk MUSIQ (metrics_all.py, 4설정)
results/long126_musiq_per_prompt.json
results/long126_musiq.png           MUSIQ·rel_err 곡선 그림
```

기존 `long126_curves.csv`는 a4_int4가 prompt000 단독 값이었으므로(실패 잔재) `_5p` 파일을 정본으로 볼 것. `latents_long126/a4_int4_partial_old/`는 bit-exact 대조를 마쳤으므로 삭제해도 됨(49MB).

## 5. 절차 기록

- a4_int4의 세션 4 실패 원인은 코드가 아니라 **외부 프로세스의 GPU 9.9GB 점유**로 인한 VAE decode OOM이었음(run.log의 `Process 1433487`). 빈 GPU에서 같은 명령으로 재실행 성공, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 사용.
- 재실행 결정성: 실패 전 부분 latent(prompt000, 42 chunk)와 42/42 bit-exact.
- `long126_musiq`는 `metrics_all.py --num_latent_frames 126 --skip flow clip lpips`로 산출(lpips 생략은 ref 정합성 때문이 아니라 A1/A4 계열별 ref 분리가 필요해서 — 필요 시 계열별로 두 번 돌릴 것).
