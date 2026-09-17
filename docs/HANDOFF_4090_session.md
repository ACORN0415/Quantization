# 4090 세션 작업 지시서 — Self Forcing 재현 + 프레임 축 양자화 오차 측정 1차

작성: 2026-09-16 · 대상: 클라우드 RTX 4090(24GB) 1장, Ubuntu, CUDA 12.x
이 세션의 목표는 **파이프라인을 확보하고 첫 오차 곡선을 뽑는 것**. 새 기법 구현은 하지 않는다. 시간당 비용이 나가므로 아래 순서를 지키고, 각 단계 끝에 결과를 저장·다운로드한다.

---

## 0. 배경 (30초 요약)

연구 주제: Wan2.1 기반 causal(streaming) 비디오 diffusion 모델(Self Forcing)에서 저정밀 양자화 오차가 KV cache를 통해 **프레임(chunk) 축으로 누적되는지** 측정한다. 모델은 3 latent frame짜리 chunk를 4 step으로 순서대로 생성하고, 과거 chunk의 K/V를 캐시에 저장해 attend한다. 오차 주입점은 (a) 가중치/활성화 양자화(W4A4, FP8), (b) KV cache 저장 시 양자화(INT4/INT2). 이번 세션은 (b)를 UCSD 공개 코드로, (a)를 fake quantization으로 흉내내어 **chunk 인덱스별 오차 곡선**을 얻는다.

참고 논문
- Self Forcing: https://arxiv.org/abs/2506.08009 · 코드 https://github.com/guandeh17/Self-Forcing
- UCSD 33-method KV 양자화 연구: https://arxiv.org/abs/2603.27469 · 코드 https://github.com/suraj-ranganath/kv-quant-longhorizon

---

## 1. 환경 세팅 (목표 30분 이내)

```bash
# 시스템 확인
nvidia-smi                      # 4090, 24GB, driver ≥ 550 확인
python --version                # 3.10 권장

# Self Forcing
git clone https://github.com/guandeh17/Self-Forcing.git && cd Self-Forcing
conda create -n sf python=3.10 -y && conda activate sf
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install flash-attn --no-build-isolation   # 실패하면 건너뛰고 SDPA fallback 사용 (아래 3절 참고)
python setup.py develop

# 체크포인트 (HF: gdhe17/Self-Forcing) — README의 다운로드 명령 그대로
# 필요 파일: checkpoints/self_forcing_dmd.pt, Wan2.1-T2V-1.3B 베이스(VAE, umT5 인코더 포함)
```

flash-attn 빌드가 20분 넘게 걸리면 중단하고 `pip install flash-attn` 사전 빌드 wheel(cu12, torch 버전 일치)을 시도. 그래도 안 되면 attention을 `torch.nn.functional.scaled_dot_product_attention`으로 바꾸는 패치 사용(코드에 fallback 분기가 있는지 먼저 grep).

확인 사항을 기록: torch 버전, CUDA 버전, flash-attn 설치 여부, 체크포인트 SHA 또는 파일 크기.

---

## 2. BF16 기준 재현 (Step A)

```bash
python inference.py \
  --config_path configs/self_forcing_dmd.yaml \
  --output_folder videos/bf16_ref \
  --checkpoint_path checkpoints/self_forcing_dmd.pt \
  --data_path prompts/MovieGenVideoBench_extended.txt \
  --use_ema
```

- 먼저 프롬프트 **3개**만으로 돌려 동작 확인(data_path를 3줄짜리 파일로 교체).
- 기록: peak VRAM(`torch.cuda.max_memory_allocated()`를 inference.py 끝에 출력하도록 한 줄 추가), 영상 1개당 wall-clock, FPS.
- 기대값: 480×832, 21 latent frame(≈5초, 81 픽셀 프레임), peak ≈ 19GB 근처, 4090에서 영상당 수십 초.
- config 확인: `num_frame_per_block: 3`, `denoising_step_list: [1000, 750, 500, 250]`, `local_attn_size`(rolling cache 길이) 값을 그대로 기록.

**중요 — 재현성**: 노이즈 seed를 chunk 단위로 고정해야 이후 양자화 실험을 BF16과 같은 궤적에서 비교할 수 있다. `pipeline/` 아래 노이즈 샘플링 위치를 찾아 `torch.Generator`에 (base_seed + chunk_idx)로 seed를 주도록 패치하고, 같은 seed로 두 번 돌려 출력 latent가 bit-exact인지 확인(`torch.equal`). 이게 안 되면 이후 모든 비교가 무의미하므로 반드시 통과시킬 것.

**latent 저장**: 디코딩 전 latent 텐서(`[1, 21, 16, 60, 104]`)를 chunk별로 `.pt`로 저장하는 훅 추가. 이후 오차 측정은 latent 공간에서 한다(VAE 디코딩은 시각 확인용).

---

## 3. 긴 rollout (Step B)

- `--num_output_frames` 또는 config로 latent frame 수를 21 → **63**(≈15초, 21 chunk)로 늘려 rolling cache가 실제로 퇴출을 시작하는 구간을 포함시킨다. `local_attn_size`보다 길어야 의미 있음.
- 프롬프트 3개 × BF16으로 실행, chunk별 latent 저장, peak VRAM·시간 기록.
- OOM이 나면 프레임 수를 낮추되, `local_attn_size`의 2배 이상은 유지.

---

## 4. KV cache 양자화 — UCSD 코드 (Step C)

```bash
cd .. && git clone https://github.com/suraj-ranganath/kv-quant-longhorizon.git
```

- README를 읽고 Self Forcing 위에 붙이는 방식을 파악(자체 fork인지, 패치 스크립트인지).
- 최소 3개 설정만 실행: **RTN INT4**, **RTN INT2**, **KIVI-style INT4**(있으면). 각각 Step B와 같은 프롬프트·seed·프레임 수.
- 이 코드가 제공하는 지표(prefix-quality curve, drift-last, peak VRAM)를 그대로 출력.
- 확인 포인트: 논문이 지적한 "양자화했는데 peak VRAM이 BF16보다 큰" 현상이 재현되는지 기록.

---

## 5. 가중치/활성화 fake quantization (Step D)

새 파일 `fakequant.py`를 만들어 `torch.nn.Linear` forward에 hook으로 끼운다. 실제 저비트 커널은 쓰지 않고 BF16 연산 전에 값을 양자화-역양자화만 한다.

```python
def fq_int(x, bits, group=128, dim=-1):
    # per-group symmetric, dim 기준 group 단위 absmax scale
    shape = x.shape
    x = x.reshape(-1, group) if group else x.reshape(-1, shape[-1])
    s = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / (2**(bits-1)-1)
    return (torch.round(x/s).clamp(-(2**(bits-1)), 2**(bits-1)-1) * s).reshape(shape)

def fq_fp8_e4m3(x):
    return x.to(torch.float8_e4m3fn).to(x.dtype)   # 4090은 FP8 캐스트 지원
```

- 적용 대상: DiT 블록 내부 Linear만(텍스트 인코더·VAE 제외). 레이어 이름으로 필터(`q`, `k`, `v`, `o`, `ffn`).
- 설정 4개: **W8A8(FP8)**, **W4A16**(가중치만 INT4), **W4A8**, **W4A4**. 가중치는 1회 사전 양자화, 활성화는 forward마다.
- 각 설정으로 Step B와 같은 프롬프트·seed·프레임 수 실행, chunk별 latent 저장.
- 추가 2개(시간 되면): **K/V projection만 W4A4**(다른 Linear는 BF16) vs **K/V projection 제외 W4A4**. 이 둘의 차이가 "캐시로 들어가는 오차"의 기여도를 분리한다.

---

## 6. 오차 곡선 계산 (Step E)

저장된 latent로 BF16 기준 대비 chunk 인덱스별 오차를 계산하는 스크립트 `measure.py`:

- chunk c마다: `rel_err[c] = ||x_q[c] − x_bf16[c]||₂ / ||x_bf16[c]||₂`, 그리고 latent PSNR.
- 설정별로 프롬프트 3개 평균과 표준편차.
- 출력: CSV(`config, chunk_idx, rel_err_mean, rel_err_std, psnr_mean`) + matplotlib PNG(x=chunk index, y=rel_err, 설정별 선).
- 추가로 KV cache 텐서 자체의 양자화 오차(캐시 저장 직전 vs 직후)를 레이어 평균으로 chunk별 기록(Step C에서 훅 가능하면).

보고 싶은 것: 곡선이 **평평한지(흡수) / 선형 증가(누적) / 계단형(rolling 퇴출 시점에 변화)** 인지. `local_attn_size` 경계 chunk에 세로선을 그려 둘 것.

---

## 7. 산출물 (세션 종료 전 반드시 다운로드)

```
results/
  env.txt                 # torch/cuda/flash-attn/GPU 정보, config 값
  bf16_ref/               # latent .pt (chunk별), 영상 3개, vram/time 로그
  long_bf16/              # 63 latent frame 버전
  kv_int4/ kv_int2/ kv_kivi4/
  fq_w8a8/ fq_w4a16/ fq_w4a8/ fq_w4a4/ (fq_kv_only_w4a4/, fq_no_kv_w4a4/)
  error_curves.csv
  error_curves.png
  NOTES.md                # 막힌 곳, 패치한 파일과 diff, 관찰
patches/                  # seed 고정, latent 저장 훅, fakequant 훅의 git diff
Dockerfile 또는 environment.yml   # 재현용
```

`NOTES.md`에는 특히: (1) seed 고정이 bit-exact를 통과했는지, (2) UCSD 코드 통합 방식, (3) peak VRAM 실측표, (4) 곡선 형태에 대한 한 줄 관찰.

---

## 8. 하지 말 것 / 주의

- 새 양자화 기법·커널 구현 금지. 이번 세션은 측정 인프라만.
- VBench 설치·평가는 하지 않는다(무겁고 이번 목표 아님). latent 오차와 눈 확인만.
- torch.compile은 첫 실행에서 끄고(디버깅 편의), 재현 확정 후에만 켠다.
- flash-attn 빌드에 30분 이상 쓰지 말 것. SDPA로 간다.
- 실험 중간마다 `results/`를 압축해 다운로드(스팟 인스턴스 회수 대비).
- 프롬프트는 MovieGenBench 파일 앞 3개를 고정 사용(이후 비교 가능성).

---

## 9. 막히면

- OOM(BF16 기준에서): `local_attn_size` 축소보다 먼저 프레임 수를 줄인다. 그래도 나면 TAEHV VAE 옵션.
- 체크포인트 다운로드 실패: HF 토큰 필요 여부 확인, `huggingface-cli download gdhe17/Self-Forcing`.
- UCSD 코드가 특정 commit의 Self Forcing을 요구하면 그 commit으로 맞춘다(우리 패치는 그 위에 재적용).
- 결정 못 할 사항은 추측으로 진행하지 말고 NOTES.md에 질문으로 남기고 다음 단계로.
