# 세션 3 작업 지시서 — sink 효과의 원인 분리, 지표 삼각측량, 메커니즘 증명, 병목 프로파일링

작성: 2026-09-16 · 선행: `HANDOFF_session2.md` → `RESULTS_session2.md`(Step F·G·H·I·J 완료)
환경: 세션 1·2와 동일 4090 24GB. `~/gpu/Self-Forcing`, `~/gpu/LongLive`(v1.0), `~/gpu/patches/`, `results/`를 이어 쓴다. Step K(14B)는 별도 H100 세션 — 이 문서 §7.

이 세션의 목적은 **세션 2의 헤드라인("attention sink가 양자화 오차 누적을 차단한다")을 리뷰어 공격에 견디게 만드는 것**이다. 새 발견보다 confound 제거·지표 보강·메커니즘 증명·현실적 동기 수치가 우선이다. 새 양자화 기법·커널 구현은 여전히 금지.

---

## 0. 세션 2가 남긴 사실과 구멍

**사실**
- TF/FR lockstep 인프라 완성. 주입(TF)은 상수, FR−TF는 증가하나 카오스 몫 미분리.
- Self Forcing(SF) chunk 7 민감도 계단 = 첫 chunk(암묵적 sink) 퇴출. `sink_size=3`으로 TF 계단 67~94% 감소. **sink3의 FR은 미실행.**
- LongLive(LL, sink3+window9, 같은 Wan2.1-1.3B): INT4 KV MUSIQ −0.3(SF −11.0), 캐시 오차는 2.3× 큼. INT2 퇴출 계단 소멸. sink만 BF16은 효과 없음.
- key 오차가 value 오차보다 단위당 약 9배 비쌈(K4V2 vs K2V4). UCSD asym 버그 2개 수정, KIVI INT4 > RTN INT4.
- 1.3B 활성화 outlier 없음.

**구멍(리뷰어 공격 포인트)**
1. LL의 강건함이 sink 때문인지, 짧은 window 때문인지, sink 구조로 fine-tune했기 때문인지 미분리.
2. sink 주장이 수치 완화만 있고 attention map 증거가 없다.
3. MUSIQ는 단일 프레임 지표 — 시간 축 결함(깜빡임, 모션 불연속, 프롬프트 이탈)을 못 본다. 프롬프트 3개, 오차 막대 없음.
4. 1.3B에서 KV cache가 실제 병목인지 수치가 없다(INT4 KV가 peak −19%인데 latency +24%).
5. 1.3B 결론의 14B 일반화 미확인.

---

## 1. 관문 A — sink 효과의 원인 분리 (최우선)

같은 SF 모델(학습 변경 없음)에서 캐시 설정만 바꿔 FR을 돌린다. 63프레임, 프롬프트는 §3에서 확정한 10개, seed 동일.

| 설정 | local_attn_size | sink_size | 목적 |
|---|---|---|---|
| A1 (기존) | 21 | 0 | 기준 |
| **A2** | 21 | 3 | sink만 추가 — **핵심** |
| **A3** | 12 | 0 | window만 LL과 맞춤 |
| A4 | 12 | 3 | LL 캐시 구조 그대로 (학습만 다름) |
| A5 | 21 | 6 | sink 크기 효과 |

각 설정 × {BF16, 대조군 1e-3, RTN INT4, RTN INT2}. 지표는 §3의 전체 세트.

**판정**
- A2 INT4가 LL 수준(MUSIQ ≈ 0)으로 가면 → "sink 구조 자체가 충분, 재학습 불필요". 논문의 실무 레시피가 가장 강해짐.
- A2는 여전히 나쁘고 A4가 LL 수준이면 → window도 필요.
- A4도 나쁘고 LL만 좋으면 → 학습 정합이 원인. 서사를 "sink+window로 학습된 모델은 저비트에 강건"으로 낮춤.
- A5 vs A2: sink를 키우면 더 좋아지는가, 6이면 잔여 계단(+0.024)이 사라지는가.

이 표가 논문의 핵심 표다. 관문 A 완료 전에 §4·§5로 넘어가지 말 것.

---

## 2. 관문 B — attention mass로 sink 메커니즘 증명

"첫 chunk가 sink"라는 주장을 attention 분포로 직접 보인다.

1. SF attention forward에 훅을 걸어 **chunk 6 생성 중**(퇴출 직전, 4 step 각각) 레이어 30 × head별로 softmax 후 attention mass가 chunk 0 토큰들에 가는 비율을 기록. BF16, 프롬프트 3개, 첫 chunk 토큰 비율(1/7 ≈ 14.3%)을 기준선으로.
2. 같은 것을 **chunk 7 생성 중**(첫 chunk 퇴출 후)에 기록 — mass가 어디로 재분배되는가(가장 오래 남은 chunk 1로 옮겨가면 "위치 기반 sink", 골고루 퍼지면 다름).
3. `sink_size=3` 설정에서 chunk 7·14·20 — sink에 가는 mass가 유지되는가.
4. LL에서 같은 측정 — 명시적 sink에 가는 mass.
5. 부수: INT4 KV 양자화 상태에서 chunk 0으로 가는 mass 변화(TUM Jensen bias 방향과 일치하는지 — 양자화된 key가 mass를 더 받는가).

**출력**: `results/attn_mass/` — 레이어×head 히트맵(chunk 6 vs 7, SF vs SF+sink3 vs LL) PNG, 레이어 평균 CSV. 논문 Fig. 2 후보.

**판정**: chunk 0 mass가 토큰 비율(14%)을 크게 넘으면(예: 30% 이상, 특정 레이어에서 50% 이상) sink 증명. 넘지 않으면 계단의 원인을 재검토하고 NOTES에 기록.

---

## 3. 관문 C — 지표 삼각측량 + 통계

**프롬프트 확대**: MovieGenBench 앞 10개로 고정(`prompts10.txt`, md5 기록). 세션 1·2의 3개는 이 10개의 부분집합이어야 함. 이후 모든 실행은 10개.

**지표 세트(모든 FR 실행에 후처리 적용, 재생성 불필요)**
1. latent rel_err (기존)
2. MUSIQ (기존) — 이제 평균 ± 표준편차 필수, 대조군 대비 차이의 신뢰구간
3. **시간 축 지표 1개 이상**: (a) 인접 프레임 optical-flow warping error(RAFT 또는 `torchvision` flow), (b) VBench의 `temporal_flickering`·`motion_smoothness` 두 차원만 설치(전체 VBench 불필요). (a)가 어려우면 (b).
4. **TF-LPIPS**: TF 모드 출력을 같은 chunk의 BF16 출력과 LPIPS로 비교. BF16 컨텍스트 하 chunk 대 chunk라 카오스 없음 — reference 기반 지표를 쓸 수 있는 유일한 위치. TF 영상은 이미 있음(`tf_*/`), 없는 설정은 디코딩만.
5. **CLIP score**(프롬프트-프레임 정렬): chunk 중앙 프레임 vs 프롬프트 텍스트. 이탈이 "다른 장면"인지 "프롬프트 벗어남"인지 구분.

**TF 영상 MUSIQ**: 세션 2의 열린 질문. MUSIQ(TF) ≈ BF16이고 MUSIQ(FR) < BF16이면 그 차이가 카오스 무관한 전파 피해. INT4·INT2·W4A4·W8A8 네 설정.

**출력**: `results/metrics_all.csv` (설정 × chunk × 5지표 × 평균/표준편차), 지표 상관 산점도(L2 vs MUSIQ vs temporal vs CLIP), 세션 2의 "두 지표가 반대로 실패한" 두 사례 쌍을 5지표로 재기술.

---

## 4. 관문 D — 병목 프로파일링 (본인 전문 영역, 논문의 동기 절)

리뷰어 질문 "1.3B에서 KV cache가 왜 문제인가"에 수치로 답한다. 4090에서 측정.

1. **chunk당 latency 분해**: `torch.profiler`로 DiT 4 step의 GEMM(QKV/O/FFN) vs attention vs VAE decode vs 기타. BF16 기준, 21프레임과 63프레임(rolling on).
2. **메모리 분해**: 가중치 / KV cache / activation / VAE / 텍스트 임베딩. allocated와 reserved 둘 다. 캐시 길이 12(LL) / 21(SF) / 32(LL-2.0 설정)에서 KV 비중.
3. **Roofline**: 4090 스펙(FP16 tensor ~165 TFLOPS dense, 1 TB/s)으로 chunk당 GEMM FLOP와 attention의 KV 읽기 바이트를 계산해 어느 연산이 compute-bound / memory-bound인지. attention의 arithmetic intensity를 캐시 길이 함수로.
4. **INT4 KV의 현재 비용**: 세션 1의 +24% latency가 어디서 나는지(역양자화 커널? BF16 재구성 버퍼? 파이썬 오버헤드?) 프로파일. 이것이 RQ3(fused 저비트 KV attention 커널)의 동기 수치.
5. **스케일 외삽 표**: 캐시 길이 × 모델 크기(1.3B/5B/14B, 가중치 크기와 head 수는 공개 config에서) × 동시 스트림 수(1/4/8)에 대해 KV 메모리를 계산해, 어느 조건에서 KV가 가중치를 넘고 12GB/24GB 카드 한도를 넘는지.

**출력**: `results/profile/` — 분해 막대 그래프 2장, roofline 1장, 외삽 표 CSV. 논문 §2(motivation) 그대로.

---

## 5. 후속 실험 (관문 A~D 후)

1. **k-step delayed TF**: BF16 컨텍스트를 chunk t−k까지 강제, k chunk만 자유 rollout. k ∈ {1,2,4,8}. 대조군도 같은 k로. 전파 horizon 그림.
2. **TUM Jensen-bias 보정 항**: `b_i ≈ (1/24d) Σ_c q_c² Δ_{i,c}²`를 cached logits에서 빼는 한 줄. SF INT2·INT4에서 MUSIQ·temporal 회복량. key 민감도 지배 결과와 직접 연결.
3. **KIVI residual window**: 최근 1 chunk를 FP16 유지. "KIVI-style" → 진짜 KIVI 비교.
4. **더 긴 rollout**: 126프레임(42 chunk) SF INT4·INT2, LL 동일. propagation 평탄화 지점.
5. **Causal Forcing 이식**(시간 남으면): 같은 코드베이스, 3번째 모델. A1 설정만.

---

## 6. 규칙

- 관문 A의 BF16 실행이 세션 1 BF16과 bit-exact(A1)인지, A2~A5는 chunk 0~(퇴출 전)까지 A1과 동일한지 먼저 확인.
- 프롬프트 10개로 통일 후 세션 1·2 핵심 설정(A1 INT4/INT2, LL INT4/INT2, 대조군)을 재실행해 표를 한 프롬프트 세트로 맞출 것. 3개짜리 수치는 논문에 안 씀.
- 새 기법·커널 없음. TUM 보정은 논문 수식 그대로만.
- 결정 못 할 사항은 NOTES에 질문으로. 단계별 `results/` 압축·다운로드.
- 절차 실수(세션 2의 config 덮어쓰기류)는 실행 전 `--output_folder`가 설정별로 다른지 dry-run으로 확인.

## 7. Step K — 별도 H100 세션 (3시간 상한)

- Wan2.1-T2V-14B BF16 로드, 프롬프트 3개, 4~8 step만. `diag_act.py` 적용: 모듈별 행 max/median, 상위 1% 채널 에너지, per-tensor vs per-channel FP8/INT4 오차, timestep 간 outlier 채널 Jaccard.
- 가능하면 LongLive-2.0-5B(BF16 체크포인트) 동일 진단 + INT4 KV FR 1개(A100 80GB면 가능).
- **key vs value 민감도 방향 확인**: 14B에서 K4V2 vs K2V4 한 번(KV만, fake 불필요). 방향이 유지되는지.
- 판정: outlier 채널이 지속적으로 존재하면 RQ2(sparse 분기) 유지·대상 상향; 없으면 RQ2 폐기하고 RQ3(fused 저비트 KV attention 커널)로 시스템 기여 집중.

## 8. 산출물

```
results/
  gateA_sink_isolation.csv/.png     A1~A5 × 4설정 × 5지표 (핵심 표)
  attn_mass/                        관문 B 히트맵·CSV
  metrics_all.csv, metric_corr.png  관문 C
  tf_musiq.csv                      TF 영상 MUSIQ
  profile/                          관문 D
  delayed_tf.csv/.png, tum_fix/, kivi_rw/, long126/   §5
  NOTES.md   (1) 관문 A 판정 문장 (2) chunk 0 attention mass 수치 (3) 지표 간 불일치 사례
             (4) INT4 KV latency +24%의 원인 (5) 프롬프트 10개 md5 (6) 각 단계 GPU 시간
patches/  09_attn_mass_hook.patch, 10_metrics.py, 11_profile.py, 12_delayed_tf.patch, 13_tum_correction.patch
```

## 9. 우선순위 (시간 부족 시)
A → C(프롬프트 확대·지표) → B → D → 5-1 → 5-2 → 나머지. A와 C는 반드시. B는 히트맵 한 장이라도.
