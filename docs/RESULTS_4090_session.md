# 4090 세션 결과 보고 — Self Forcing 재현 + 프레임 축 양자화 오차 측정 1차

작성: 2026-09-16 · 대상 지시서: `HANDOFF_4090_session.md`
실행 환경: RTX 4090 24GB 1장, Ubuntu 22.04, driver 560.28.03, CUDA 12.6

**전체 완료.** 지정된 항목은 하나도 빠지거나 대체되지 않았다. 진행 중 지시서 가정과 어긋나는 블로커 2건을 고쳐서 통과했고, §6 지표에 방법론적 문제가 있어 대조군을 추가했다.

---

## 0. 세 줄 요약

1. **누적 가설은 이번 측정에서 성립하지 않는다.** KV 캐시에 주입되는 양자화 오차는 chunk 축을 따라 평평하다(오히려 미세 감소). 비트폭은 매 chunk 주입되는 *상수* 오차 크기를 정할 뿐이고, 붕괴는 그 상수가 모델의 자기 보정 범위를 넘을 때 일어난다.
2. **§6이 지정한 latent L2 지표는 품질이 아니라 카오스적 궤적 이탈을 잰다.** 대조군(BF16 + 미세 섭동) 없이는 "누적"과 "그냥 다른 샘플로 감"을 구분할 수 없다. 대조군을 추가해 판정 기준을 세웠다.
3. **RTN INT4는 안전(대조군과 구분 불가), RTN INT2만 chunk 7 퇴출 시점에 꺾인 뒤 선형 누적하며 영상이 실제로 붕괴한다.**

---

## 1. 환경 (§1)

| 항목 | 값 |
|---|---|
| torch | 2.6.0+cu124 |
| torchvision | 0.21.0+cu124 |
| flash-attn | **2.7.4.post1 (설치 성공, 사전 빌드 wheel — 소스 빌드 불필요)** |
| Python | 3.10.12, venv (`~/gpu/sf-venv`) |
| numpy / transformers / diffusers | 1.24.4 / 5.17.0 / 0.31.0 |
| Self-Forcing | `33593df3e81fa3ec10239271dd2c100facac6de1` |
| kv-quant-longhorizon | `b4c09363735c89947222781db899862bc0dc55c5` |
| self_forcing_dmd.pt | md5 `ef41ae42bfb87abe43d5f640fb634d0c`, 5,676,252,553 B |
| Wan2.1-T2V-1.3B | 17GB (umT5 인코더 11GB, VAE 508MB, DiT 5.7GB) |

### 지시서와 달랐던 점

- **conda 없음** → `venv` 사용. `python3.10-venv`를 apt로 설치해야 했다.
- **flash-attn 빌드 불필요.** ABI가 맞는 사전 빌드 wheel(`cu12torch2.6cxx11abiFALSE-cp310`)이 그대로 동작한다. `torch._C._GLIBCXX_USE_CXX11_ABI`가 False인 것만 확인하면 된다. 30분 제한을 쓸 일이 없었다. (참고: `wan/modules/attention.py`에 SDPA fallback이 이미 있어 실패해도 자동 우회된다.)
- **`requirements.txt`를 그대로 쓰면 안 된다.** `nvidia-tensorrt`, `pycuda`, `onnx*`, `dashscope`, CLIP, `wandb`, `flask*` 등은 CLI inference 경로에서 import되지 않는다(데모·학습 전용). 추려낸 `requirements-infer.txt`를 썼다.
- **단, `lmdb`는 반드시 필요하다.** `utils/dataset.py`가 무조건 import한다. 처음 추려낼 때 빠뜨려 첫 실행이 실패했다.
- `huggingface_hub` 1.x는 `--local-dir-use-symlinks`를 제거했고 CLI 이름이 `hf`로 바뀌었다. README의 `huggingface-cli`도 경고와 함께 동작은 한다.
- `transformers`가 5.17.0으로 잡혔다(리포 요구 `>=4.49.0`). `AutoTokenizer` 한 곳에서만 쓰이고 T5는 리포 자체 구현이라 문제없었다.

---

## 2. 재현성 — 관문 통과 (§2)

지시서가 "이게 안 되면 이후 모든 비교가 무의미"라고 한 항목이다. **단순 재실행 bit-exact만으로는 부족하다**고 판단해 검증을 3종으로 늘렸다.

| 검증 | 결과 |
|---|---|
| 같은 seed 두 번 실행, chunk latent `torch.equal` | **PASS** (21/21, 3 프롬프트 × 7 chunk) |
| **chunk마다 전역 CUDA RNG를 7777개씩 강제 소비시킨 뒤 비교** | **PASS** (21/21) |
| UCSD `causal_model.py` 패치 적용 전후 비교 | **PASS** (63/63) |

두 번째가 핵심이다. 지시서가 실제로 걱정한 실패 모드는 "양자화 코드가 RNG를 소비해 궤적이 어긋나는 것"인데, 단순 재실행으로는 이걸 못 잡는다.

### 찾은 문제와 수정

노이즈는 `inference.py`에서 한 번에 미리 샘플링되므로 그 자체는 결정적이었다. **그런데 `pipeline/causal_inference.py`가 denoising step마다 `torch.randn_like()`로 전역 CUDA RNG를 추가 소비하고 있었다.** 다른 RNG 소비자가 하나라도 끼면 그 시점부터 전부 어긋난다.

수정: 실행 이력에 의존하지 않는 chunk별 `torch.Generator` **두 줄기**로 교체.

- 초기 chunk 노이즈: `seed * 1000003 + chunk_idx`
- chunk 내 re-noising: `seed * 1000003 + 7919 + chunk_idx`

같은 chunk의 초기 노이즈와 re-noise가 상관되지 않도록 스트림을 분리했다.

### latent 저장

디코딩 전 chunk별 `denoised_pred`를 `.pt`로 덤프하는 훅을 추가했다(`--latent_dump_root`). 오차 측정은 전부 이 latent로 했다.

---

## 3. Self-Forcing 자체의 블로커 — 63프레임 실행 불가 (§3)

**지시서 Step B는 원래 코드로 실행되지 않는다.**

`WanDiffusionWrapper`의 `local_attn_size` 기본값이 `-1`이고 동봉된 config 어느 것도 이를 덮어쓰지 않는다. 그래서 `wan/modules/causal_model.py`의 rolling 퇴출 분기가 **꺼진 채로** 동작한다. 그런데 KV 캐시 버퍼는 `32760` 토큰 = **정확히 21 latent frame**으로 고정이다. 21프레임에서는 딱 맞지만, 그 이상이면 쓰기 슬라이스와 소스 텐서의 shape이 어긋나 죽는다.

수정: `configs/self_forcing_dmd_long.yaml`에 `model_kwargs.local_attn_size: 21`을 넣어 퇴출 경로를 활성화했다.

### 곡선 해석에 직결되는 귀결

chunk당 3 latent frame이므로 캐시는 정확히 7 chunk를 담고, **chunk 6이 마지막 토큰까지 채운다. 퇴출은 chunk 7부터 시작된다.** 이 값이 `error_curves.png`의 세로선 위치다.

21프레임 실행은 이 설정의 영향을 받지 않는다(21프레임 내에서는 퇴출 분기가 발동하지 않음을 확인). 따라서 Step A 기준값은 그대로 유효하다.

---

## 4. 실측 성능 (§2, §3, §4)

| 실행 | 프레임 | 영상당 | peak alloc | peak reserved |
|---|---:|---:|---:|---:|
| BF16 (Step A) | 21 | 13.9s | **12.75 GB** | 15.68 GB |
| BF16 (Step B) | 63 | 46.6s | 13.16 GB | 19.40 GB |
| fakequant 전반 | 63 | 46.6~51.2s | 13.16 GB | 19.4~19.5 GB |
| RTN INT4 (KV) | 63 | 58.0s | **10.70 GB** | 15.94 GB |
| RTN INT2 (KV) | 63 | 58.0s | 10.70 GB | 15.94 GB |
| KIVI INT4 (KV) | 63 | 60.3s | 10.70 GB | 15.75 GB |

### 지시서 기대치와 다른 점

- **기대 ~19GB, 실측 12.75GB (allocated).** reserved는 15.68GB까지 간다. 어느 설정 기준의 추정치였는지 확인이 필요하다.
- **프레임 3배(21→63)에도 peak가 거의 안 움직인다**(12.75→13.16GB). rolling 캐시가 제 역할을 한다는 직접 증거 — 영상 길이와 무관하게 KV 상주량이 21프레임으로 상한이 걸린다.
- **§4가 확인하라고 한 "양자화했는데 peak VRAM이 BF16보다 큰" 현상은 재현되지 않았다.** RTN INT4는 BF16보다 **19% 낮고**(10.70 vs 13.16GB), 대신 24% 느리다. 상주하던 BF16 캐시 약 6GB가 사라지는 이득이 레이어별 임시 역양자화 버퍼 비용을 넘어선다. 이 설정(`local_attn_size=21`, 캐시가 작음)에서는 논문이 지적한 역전이 일어나지 않는다.

---

## 5. UCSD 코드 통합 (§4)

### 통합 방식

그들의 `scripts/01_generate.py`는 Self-Forcing을 `third_party/`에 두고 자체 하네스로 구동한다. 그 경로로 가면 **우리 seed 고정과 chunk별 latent 덤프가 빠져서 오차 곡선을 뽑을 수 없다.** 그래서 반대로 **그들의 quantizer를 우리 파이프라인에 붙였다.**

1. `docs/patches/self_forcing_kv_quant.patch`를 그대로 적용. `wan/modules/causal_model.py`만 건드리는데 우리가 손대지 않은 파일이라 충돌 없음. 적용 후 BF16이 여전히 bit-exact임을 확인(quantizer가 없으면 no-op).
2. `_initialize_kv_cache`가 레이어 dict에 `"quantizer"`, `"quant_state"`를 심도록 수정.
3. `inference.py --kv_quant RTN --kv_bits 4`가 `kv_quant.factory.create_quantizer`로 생성해 `pipeline.kv_quantizer`에 설정.

### 그들 코드에서 발견한 버그 (§4 진행 불가 요인)

`kv_quant/factory.py`가 `RTNQuantizer`/`KIVIQuantizer`에 `key_bits`, `value_bits`, `name`을 넘기는데 **함수 안에서 바인딩하지 않는다.** `**kwargs`로 들어오지만 꺼내 쓰질 않아 자유 변수로 남아 있다. 두 경로 모두 `NameError: name 'key_bits' is not defined`로 죽는다.

**하필 §4가 지정한 RTN INT4 / RTN INT2 / KIVI INT4 세 설정이 전부 이 경로다.** 고치지 않으면 Step C에서 실행 가능한 것이 하나도 없었다. `kwargs`에서 pop하도록 최소 수정했다(`patches/05_ucsd_factory_kwargs.patch`).

### 프롬프트 간 캐시 재사용

그들 패치는 quantizer가 붙으면 `kv_cache["k"]`를 빈 텐서로 교체하는데, 파이프라인의 프롬프트 간 리셋은 인덱스 카운터만 0으로 만든다. `quant_state`를 비우고 k/v 버퍼를 재할당하도록 추가했다.

---

## 6. §6 지표의 함정 — 이번 세션의 방법론적 핵심

지시서가 지정한 `rel_err[c] = ||x_q[c] − x_bf16[c]|| / ||x_bf16[c]||`는 양자화 하에서 **약 8 chunk 만에 1.0 부근 이상으로 포화된다.** 1.0은 양자화 latent가 기준과 사실상 무상관이라는 뜻이다.

**그런데 영상은 멀쩡하다.** 프롬프트 0의 frame 40:

| 설정 | rel_err @ chunk 20 | 실제 영상 |
|---|---:|---|
| BF16 | — | 프롬프트와 정확히 일치 |
| W8A8 (FP8) | 0.94 | 같은 장면·인물, 일관적. 행인 배치만 다름 |
| W4A16 | 1.24 | 완전히 일관적, 구도만 다름 |
| W4A4 | 1.34 | 일관적, 약간 뭉개짐. 붕괴 아님 |

즉 이 지표는 **카오스적 샘플러의 궤적 이탈**을 재는 것이지 품질 저하가 아니다. 4-step 증류 모델은 초기에 살짝 밀리면 똑같이 타당한 다른 샘플로 간다. latent L2 단독으로는 "KV 캐시를 통한 누적"과 "샘플러가 다른 데로 감"을 구분할 수 없다.

### 양자화기가 조악해서 생긴 문제는 아니다 (측정으로 확인)

모델 탓으로 돌리기 전에 먼저 확인했다.

- **실제 체크포인트 가중치 180개**: FP8 per-tensor 0.0265 vs per-channel 0.0264 — 동일. outlier 채널이 없다는 뜻. INT4 group-128은 0.1225(보정 없는 RTN의 정상값), INT8은 0.0061.
- **실제 forward 중 캡처한 활성화**: per-tensor 0.0263~0.0290 vs per-token 0.0257~0.0265, 행 최대/중앙 비율 1.2~4.6배. 병적 활성화 outlier는 50~1000배다. **이 모델엔 없다.**

따라서 단순한 per-tensor FP8 스케일이 손해를 끼치지 않으므로 **양자화기는 그대로 뒀다.** 추측으로 고쳤다면 무의미한 변경이었다.

### 대조군 추가 (지시서에 없는 항목)

BF16 그대로, 초기 노이즈만 `(1 + eps·N(0,1))`로 곱한 실행. "이 모델이 *어떤* 섭동에서든 갈라지는 속도"를 잰다.

**섭동 크기와 무관함을 확인했다**: `eps=1e-2`와 `1e-3`의 곡선 차이가 chunk 5 이후 평균 **+0.024**뿐이다. 카오스적 포화의 전형적 특징이며, 대조군이 섭동이 아니라 샘플러의 성질임을 뜻한다. 이 곡선이 **카오스 바닥선**이고, 다른 모든 곡선은 이것의 배수로 읽어야 한다.

> `eps=1e-5` 대조군은 모든 chunk에서 정확히 0.0000이 나왔다. 이건 이탈 측정이 아니다 — bf16의 상대 정밀도가 약 0.4%라 1e-5 섭동이 반올림으로 사라져 노이즈 텐서가 그대로다. 우연히 얻은 네 번째 bit-exact 확인이며, 그림에서는 제외했다.

---

## 7. 오차 곡선 결과 (§6)

`results/error_curves.png` (latent L2, 검은 점선이 대조군) · `results/error_curves.csv`

| 설정 | c0 | c7 | c14 | c20 | 바닥선 대비 | 판정 |
|---|---:|---:|---:|---:|---:|---|
| 대조군 1e-3 | 0.073 | 0.584 | 0.655 | 0.738 | 1.00× | **카오스 바닥선** |
| 대조군 1e-2 | 0.115 | 0.600 | 0.718 | 0.751 | 1.02× | 바닥선(크기 무관 확인) |
| **RTN INT4 (KV)** | 0.000 | 0.569 | 0.677 | **0.808** | **1.09×** | **바닥선 위 — 누적 없음** |
| W8A16 | 0.149 | 0.640 | 0.765 | 0.836 | 1.13× | 바닥선 수준 |
| W8A8 (FP8) | 0.295 | 0.763 | 0.833 | 0.937 | 1.27× | 약간 위 |
| K/V projection만 W4A4 | 0.475 | 0.772 | 0.946 | 1.087 | 1.47× | 위 |
| W4A16 | 0.626 | 0.929 | 1.154 | 1.239 | 1.68× | 위 |
| W4A8 | 0.623 | 0.936 | 1.147 | 1.244 | 1.69× | 위 |
| KIVI INT4 (KV) | 0.000 | 1.009 | 1.153 | 1.277 | 1.73× | 위 |
| K/V 제외 W4A4 | 0.609 | 0.927 | 1.143 | 1.290 | 1.75× | 위 |
| W4A4 | 0.632 | 1.006 | 1.175 | 1.337 | 1.81× | 위 |
| **RTN INT2 (KV)** | 0.000 | 0.909 | 1.387 | **1.850** | **2.51×** | **실제 누적 + 영상 붕괴** |

KV 양자화 3종의 c0이 **정확히 0.000**인 것은 계측 검증이다 — chunk 0에는 참조할 이전 KV가 없으니 KV 양자화가 영향을 줄 수 없다.

### 곡선 모양 판정 (§6의 질문에 대한 답)

- **RTN INT4 = 평평.** chunk 8 이후 대조군과 구분 불가(chunk 9에서 0.624 대 0.621). 앞서 "상승 후 포화"로 보였던 것은 흡수가 아니라 **이탈이 천장에 닿은 것**이다. 대조군이 없었다면 "누적 후 흡수"로 오독했을 것이다.
- **RTN INT2 = 계단 후 선형.** 퇴출 경계까지는 다른 곡선들과 같이 가다가 **chunk 7 — rolling 캐시가 가장 오래된 chunk를 버리기 시작하는 바로 그 지점** — 에서 위로 꺾이고, 이후 1.85까지 거의 선형으로 오른다. √2(무상관 선)를 크게 넘는다. **지시서가 예상한 "퇴출 시점의 계단형 변화"가 실제로 관측된 유일한 사례다.**
- 나머지는 모두 대조군과 **같은 기울기**로 오프셋만 높다. 기울기가 같다는 것은 오차가 chunk마다 주입될 뿐 축적되지 않는다는 뜻이다.

---

## 8. KV 캐시 자체의 양자화 오차 — 메커니즘 규명 (§6 조건부 항목)

캐시에 넣기 직전과 다시 꺼낸 직후의 상대 오차. 30 레이어 × 3 프롬프트 평균.
`results/kv_cache_error.png` · `results/kv_cache_error.csv`

| 방법 | chunk 8 (K) | chunk 20 (K) | 변화 | chunk 20 (V) |
|---|---:|---:|---:|---:|
| RTN INT4 | 0.0280 | 0.0278 | **−0.5%** | 0.0371 |
| RTN INT2 | 0.1383 | 0.1298 | **−6.1%** | 0.1728 |
| KIVI INT4 | 0.1907 | 0.1841 | **−3.5%** | 0.0569 |

**세 방법 모두 캐시 오차가 누적되지 않는다.** 캐시가 찬 뒤로는 평평하고 오히려 미세하게 줄어든다.

UCSD 패치는 매 chunk마다 **캐시 전체**를 역양자화·재양자화하므로 복리 누적이 당연히 예상되는 구조인데, 일어나지 않는다. RTN 계열 반올림이 **거의 멱등**이기 때문이다(이미 양자화 격자 위에 있는 값은 다시 양자화해도 자기 자신으로 간다).

### latent 곡선과 합치면

> **KV 캐시 안에서 오차가 복리로 쌓이는 일은 일어나지 않는다.** 비트폭이 정하는 것은 매 chunk 주입되는 **상수** 오차의 크기(INT4 2.8%, INT2 13%)이고, rollout 품질은 그 상수가 모델 동역학이 흡수할 수 있는 범위를 넘을 때 붕괴한다. 시간이 지나며 캐시가 나빠지는 것이 아니다.

지시서 §0이 세운 누적 가설에 대해, 최소한 RTN/KIVI 계열 캐시 양자화에서는 **부정적 결과**다.

### 육안 확인 (frame 75, rollout 후반)

- **RTN INT4** (rel_err 0.81): 선명하고 완전히 일관적. 열화 없음.
- **RTN INT2** (rel_err 1.85): 배경이 녹아내리듯 번지고 구조가 무너짐. **실제 붕괴.**

rel_err 약 1.2~1.4가 "다른 샘플로 갔을 뿐"과 "붕괴"를 가르는 경계로 보인다.

---

## 9. 예상 밖 결과 — KIVI INT4가 RTN INT4보다 훨씬 나쁘다

KIVI는 더 정교한 방법(키 per-channel, 값 per-token)이라 같은 비트에서 RTN을 이겨야 한다. 그런데 바닥선 대비 거리가 거의 두 배다(1.28 대 0.81).

캐시 계측이 원인을 짚어준다: **KIVI의 key 오차가 0.184로 RTN의 0.0278 대비 6.6배**이고, 같은 4비트인데 **RTN INT2의 key(0.130)보다도 나쁘다.** value 오차(0.057)는 더 좋은 쪽이지만 그 손해를 못 갚는다.

부수적 관찰 하나: RTN INT2(k=0.130, v=0.173)가 KIVI(k=0.184, v=0.057)보다 key 오차는 작은데 latent 결과는 더 나쁘다. **value 오차가 key 오차보다 비싸다**는 힌트다.

---

## 10. 지시서 대비 추가·변경 사항

추가한 것 (지정 항목은 하나도 빼거나 대체하지 않음):

| 항목 | 이유 |
|---|---|
| 대조군 2종 (`noise_perturb` 1e-3, 1e-2) | §6 지표만으로는 누적과 궤적 이탈을 구분 불가 |
| `w8a16` (INT8 가중치) | §5 지정 설정이 전부 8 chunk 내에 이탈 천장을 침. 약 20배 작은 섭동이라 초반 곡선이 읽힘 |
| `--kv_err_log` (in-situ KV 캐시 오차) | §6이 조건부로 요청한 항목. 이탈 혼동에 면역인 직접 지표 |
| RNG 교란 강건성 검증 | 단순 재실행 bit-exact로는 실제 위험을 못 잡음 |
| 활성화 outlier 진단 (`diag_act.py`) | 양자화기를 고칠지 추측 대신 측정으로 결정 |

§8 금지 사항 준수: 새 양자화 기법·커널 구현 없음(RTN/FP8 표준 방식만), VBench 미설치, torch.compile 미사용, flash-attn은 빌드 없이 wheel로 해결, 프롬프트는 MovieGenBench 앞 3개 고정(`prompts3.txt`, md5 `aaef428afc8da6c3b8cc192db4328c07`).

---

## 11. 산출물

```
~/gpu/Self-Forcing/results/
  env.txt                     GPU/CUDA/패키지/체크포인트 md5/config 값
  SETUP.md                    환경 재현 절차 (venv 기반)
  NOTES.md                    상세 기록 (영문, 291줄)
  error_curves.csv / .png     latent L2 곡선 (12개 설정 + 대조군, 퇴출 경계선 포함)
  kv_cache_error.csv / .png   KV 캐시 자체 오차 (keys/values 분리)
  requirements-frozen.txt
  bf16_ref/ long_bf16/                      Step A, B
  kv_int4/ kv_int2/ kv_kivi4/               Step C
  fq_w8a8/ fq_w4a16/ fq_w4a8/ fq_w4a4/      Step D
  fq_kv_only_w4a4/ fq_no_kv_w4a4/ fq_w8a16/ Step D 추가
  ctrl_perturb1em2/ 1em3/ 1em5/             대조군
  kverr_int4/ kverr_int2/ kverr_kivi4/      KV 캐시 오차 계측
  latents21/ latents63/                     chunk별 latent (.pt)
  (각 실행 디렉터리: videos/, stats.json, run.log)

~/gpu/patches/
  01_inference_seed_dump_stats_quant.patch
  02_causal_inference_seed_latent_kvhooks.patch
  03_causal_model_ucsd_kvquant_plus_errlog.patch
  05_ucsd_factory_kwargs.patch              UCSD 리포용
  fakequant.py measure.py measure_kverr.py diag_act.py
  self_forcing_dmd_long.yaml requirements-infer.txt run_*.sh
  README.md                                 적용 순서와 설명

~/gpu/session_results.tar.gz    77MB  (latent 제외 — 영상·CSV·PNG·로그·패치)
~/gpu/session_latents.tar.gz    1.0GB (chunk별 latent 원본)
```

---

## 12. 다음 세션 과제

1. **KIVI의 key 양자화가 왜 이렇게 나쁜가.** 같은 비트폭에서 순정 RTN보다 6.6배 나쁜 건 기법이 아니라 구현을 의심할 크기다. KIVI를 기법으로 평가하기 전에 `kv_quant/kivi.py`를 논문과 대조할 것.
2. **value가 key보다 취약해 보인다.** 비대칭 할당(K4V8, K2V8, K8V2)을 돌려 latent 곡선이 value 오차를 따라가는지 확인.
3. **붕괴 임계점은 어디인가.** INT4(캐시 오차 2.8%)는 안전, INT2(13%)는 붕괴. 3비트 실행이나 block size를 일부러 키운 INT4로 무릎을 찾을 것.
4. **품질 주장을 하려면 지표를 바꿔야 한다.** latent L2는 카오스 바닥선에서 포화되어 그 위로는 품질에 대해 아무 말도 못 한다(W4A16은 1.24에 영상 멀쩡, RTN INT2는 1.85에 붕괴). 참조 불요 지표나 지각 기반 지표가 필요하다. (VBench는 이번 세션 범위 밖이었다.)
5. **더 긴 rollout.** 63프레임은 퇴출 경계 이후 14 chunk를 준다. INT2 곡선은 chunk 20에서도 여전히 오르는 중이라 평탄해지는지 계속 가는지 불명.
6. **peak VRAM 기대치 확인.** 지시서는 21프레임에서 ~19GB를 예상했으나 실측 12.75GB(allocated)다. 어느 설정 기준의 추정이었는지 확인 필요.
