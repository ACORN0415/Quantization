# 세션 2 작업 지시서 — 진짜 누적 측정(teacher-forced 분리), KIVI 점검, sink 모델, 임계점 탐색

작성: 2026-09-16 · 선행: `HANDOFF_4090_session.md` → `RESULTS_4090_session.md`
환경: 세션 1과 동일 4090(24GB). `~/gpu/patches/` 패치와 `results/` 를 그대로 이어 쓴다. 단, 6절(Wan 14B outlier 진단)만 H100 80GB 인스턴스가 별도로 필요하며, 이 세션의 마지막에 또는 별도 짧은 세션으로 수행한다.

이 세션의 목적은 세션 1이 남긴 세 가지 미해결을 닫는 것이다: (1) 세션 1의 "누적 없음"은 누적을 잴 수 없는 지표로 낸 결론이므로 **teacher-forced 분리로 진짜 누적을 잰다**, (2) KIVI 이상치의 원인을 구현 수준에서 확정한다, (3) sink 유무와 비트폭에 따라 붕괴 임계점이 어디로 움직이는지 잰다. 새 양자화 기법·커널 구현은 여전히 금지.

---

## 0. 세션 1 결과 요약 (이 세션이 딛고 서는 사실)

- seed 고정 + chunk별 latent 덤프 + UCSD quantizer 통합 파이프라인 완성. `local_attn_size: 21` 명시 필요(기본 −1은 퇴출 꺼짐). 캐시 = 7 chunk, 퇴출은 chunk 7부터.
- latent L2 `rel_err`는 카오스 이탈로 약 8 chunk 만에 포화. 노이즈 섭동 대조군(eps 1e-3)이 바닥선.
- RTN INT4 KV: 바닥선과 구분 불가. RTN INT2 KV: chunk 7에서 꺾인 뒤 선형 상승, 영상 붕괴. KIVI INT4: key 오차가 RTN INT4의 6.6배(이상).
- W4A4 fake-quant: 바닥선 1.8×이지만 영상은 멀쩡. FP8: 1.27×.
- Wan2.1-1.3B 활성화에 outlier 채널 없음(행 max/median 1.2~4.6×).
- in-situ KV 오차(저장 직전 vs 직후)는 상수. **이것은 양자화 오차 크기이며 전파를 재는 지표가 아니다.**

---

## 1. Step F — teacher-forced vs free-running 분리 (최우선, 이 세션의 핵심)

### 목적
"컨텍스트를 통해 전파된 오차"를 카오스 혼동 없이 chunk별로 정의한다.

- **TF(teacher-forced)**: chunk t를 생성할 때 KV cache에 **BF16 run의 K/V**(같은 seed, 같은 프롬프트)를 강제로 채우고, 현재 chunk의 연산만 양자화한다. 결과 = 매 chunk **주입**되는 순수 오차 e_inj[t].
- **FR(free-running)**: 세션 1과 같은 자유 rollout. 결과 = 주입 + 전파 e_total[t].
- **전파량** = e_total[t] − e_inj[t]. 이것이 chunk 인덱스에 따라 증가하면 누적, 평평하면 비누적. 카오스 바닥선 문제가 없다: TF는 매 chunk BF16 궤적에 재고정되므로 이탈이 쌓이지 않는다.

### 구현
1. BF16 run에서 chunk마다 **캐시 저장 직전 K/V 텐서**(RoPE 후, 레이어별)를 덤프하는 훅 추가. 63프레임 × 30레이어 × 3프롬프트 — 디스크 수 GB. `--kv_dump_root`.
2. `--kv_inject_root` 옵션: chunk t 생성 시작 전에 캐시 슬롯(0..t−1)을 덤프된 BF16 K/V로 덮어쓴다(rolling 퇴출 반영해 최근 7 chunk만). 현재 chunk의 K/V는 정상 계산·양자화 후 저장하되 다음 chunk 시작 시 다시 BF16으로 덮인다.
3. 검증: BF16 설정으로 TF를 돌리면 FR과 **bit-exact**여야 한다. 이 관문을 먼저 통과.
4. TF 모드에서 latent 오차는 chunk마다 독립이므로 `rel_err_TF[t]`가 chunk에 따라 평평한지 확인 — 평평하지 않으면 주입 훅에 버그.

### 실행 격자 (프롬프트 3개, 63프레임, 세션 1과 같은 seed)
| 설정 | TF | FR(세션 1 재사용 가능) |
|---|---|---|
| RTN INT4 KV | 실행 | 있음 |
| RTN INT2 KV | 실행 | 있음 |
| W4A4 fake | 실행 | 있음 |
| W8A8 FP8 fake | 실행 | 있음 |
| K/V proj만 W4A4 | 실행 | 있음 |

### 출력
`results/propagation.csv`: `config, chunk_idx, err_TF_mean, err_FR_mean, propagation = err_FR − err_TF` + PNG(설정별 propagation 곡선, chunk 7 세로선). **이 그림이 논문 Fig. 1 후보다.**

판정: propagation이 chunk 7 이후에 증가하는가(퇴출이 앵커를 없애 전파를 키우는가), INT2에서만 증가하는가, W4A4는 어떤가.

---

## 2. Step G — KIVI 구현 점검

1. `kv_quant/kivi.py`를 읽고 key 양자화가 **RoPE 적용 후** 텐서에 per-channel로 걸리는지 확인. Self Forcing 캐시는 RoPE 후 key를 저장한다.
2. KIVI 원 논문(Liu et al., ICML 2024) 설정과 대조: group size, 채널 축 방향(head_dim 축인지 token 축인지), residual window(최근 토큰 FP16 유지) 존재 여부.
3. 확인되면 두 가지 실험: (a) 채널 축 방향만 고쳐 재실행, (b) 가능하면 RoPE 전 key에 양자화하고 읽을 때 RoPE 적용(구조 변경 크면 (b)는 생략하고 NOTES에 기록).
4. 결과를 `results/kivi_fix/`에. RTN INT4 대비 key 오차(0.0278)가 같은 자리로 오면 구현 버그 확정.

---

## 3. Step H — sink 모델: LongLive

```bash
git clone https://github.com/NVlabs/LongLive.git
```
- Wan2.1-1.3B 기반, frame sink 3 + local window 9 latent frame(effective 12). 체크포인트 다운로드.
- 세션 1 패치(seed 고정, latent 덤프, UCSD quantizer)를 LongLive 파이프라인에 이식. 캐시 구조가 다르므로 `causal_model.py` 대응 파일에서 sink 슬롯이 어디인지 파악해 NOTES에 도식으로 남길 것.
- 실행: BF16 / 노이즈 대조군 1e-3 / RTN INT4 KV / RTN INT2 KV / W4A4 fake. 63프레임, 프롬프트 3개.
- 추가 1개: **sink 슬롯만 BF16 유지 + 나머지 INT2** — sink 오차가 상수 편향으로 작용한다는 가설의 직접 검증.
- 비교 대상 질문: Self Forcing에서 chunk 7에 있던 INT2 꺾임이 LongLive에서는 (a) 사라지는가, (b) window 경계로 옮겨가는가, (c) 없어지고 대신 오프셋만 커지는가.

---

## 4. Step I — 붕괴 임계점 탐색 (Self Forcing, FR 모드)

세션 1: 캐시 오차 2.8%(INT4)는 안전, 13%(INT2)는 붕괴. 그 사이를 채운다.
- RTN INT3 (group 128)
- RTN INT4, group 512 / 1024 / per-tensor (그룹을 키워 오차를 인위적으로 늘림)
- 비대칭: K4V8, K8V4, K2V8, K8V2, K3V3
- 각 설정에서 in-situ 캐시 오차(K, V 분리)와 chunk 20의 latent rel_err를 기록.
- 출력 `results/threshold.csv` + 산점도(x = 캐시 오차 크기, y = chunk 20 rel_err, K/V 색 구분). 붕괴 무릎(rel_err 약 1.3 초과)이 어느 캐시 오차에서 나오는지, K와 V의 무릎이 다른지.

---

## 5. Step J — reference-free 지표 1개 붙이기

latent L2와 나란히 볼 품질 지표. VBench 전체는 무거우니 하나만.
- 1순위: VBench의 **Imaging Quality** 차원만(MUSIQ 기반, 단일 모델). 설치 실패 시 `pyiqa`의 MUSIQ 또는 NIQE.
- chunk 단위로 디코딩된 프레임(chunk 중앙 프레임 1장)에 적용해 `iq[c]` 곡선.
- 세션 1의 FR 결과 전부(12개 설정 + 대조군)에 후처리로 적용 — 재생성 불필요, 저장된 영상 사용.
- 출력 `results/iq_curves.csv/png`. 질문: 대조군은 평평한가(카오스 이탈은 품질을 안 떨어뜨림), INT2는 chunk 7 이후 떨어지는가, W4A4는 어디인가.

---

## 6. Step K — Wan 14B 활성화 outlier 진단 (H100 80GB 필요, 2~3시간)

RQ2(sparse outlier 분기)의 존속 여부를 결정한다.
- Wan2.1-T2V-14B(bidirectional이어도 무관, 활성화 통계만 본다) BF16 로드. 프롬프트 3개, 4~8 step만 실행하며 `diag_act.py`를 그대로 적용.
- 기록: 레이어별·모듈별(q/k/v/o/ffn1/ffn2) 행 max/median 비율, 상위 1% 채널이 차지하는 에너지 비율, per-tensor vs per-channel FP8/INT4 오차, timestep별 outlier 채널 인덱스의 Jaccard 유사도.
- 가능하면 LongLive-2.0-5B BF16도 같은 진단(우리 계열의 유일한 5B).
- 판정 기준: 행 max/median이 20× 이상인 채널이 일부 레이어에 지속적으로 존재하면 RQ2 유지(대상을 14B/5B로), 아니면 RQ2 폐기 또는 재정의.

---

## 7. 산출물

```
results/
  propagation.csv/.png        Step F (핵심)
  kivi_fix/                   Step G + NOTES에 원인 서술
  longlive/                   Step H (BF16, ctrl, int4, int2, w4a4, sink_bf16_rest_int2)
  threshold.csv/.png          Step I
  iq_curves.csv/.png          Step J
  outlier_14b/ (별도 H100)    Step K
  NOTES.md                    (1) TF bit-exact 관문 통과 여부 (2) KIVI 원인 (3) LongLive 캐시 도식
                              (4) 임계점 수치 (5) 14B outlier 판정 (6) 각 단계 시간·VRAM
patches/  06_kv_dump_inject.patch, 07_longlive_port.patch, 08_kivi_fix.patch, iq_eval.py
```

---

## 8. 규칙

- Step F의 BF16 TF bit-exact 관문을 통과하기 전에는 F의 다른 설정을 돌리지 말 것.
- 세션 1 결과(FR 곡선)는 재생성하지 않고 재사용. 같은 seed·프롬프트·프레임 수 유지.
- 새 양자화 기법·커널 없음. KIVI "수정"은 원 논문 사양으로 맞추는 것까지만.
- 결정 못 할 사항은 추측 대신 NOTES에 질문으로. 단계별로 `results/`를 압축·다운로드.
- Step K는 별도 인스턴스이므로 4090 세션 종료 후 진행해도 된다. 비용 상한 3시간.

## 9. 우선순위 (시간 부족 시)
F → G → I → J → H → K. F와 G는 반드시. H(LongLive 이식)는 가장 오래 걸리므로 F/G/I/J가 끝난 뒤 남은 시간에 착수하고, 못 끝내면 이식 상태와 막힌 지점을 NOTES에 남긴다.
