# 인계: sf-kvquant 재구현 및 검증 (Codex용)

작성 2026-10-27 · 작성 Claude (이 문서까지만 작성하고 손을 뗌) · 대상 Codex

## 0. 목표와 사용자 요구

`~/Quantization`은 Self-Forcing KV 캐시 양자화 연구의 **패치·도구·문서 저장소**다.
연구를 돌리던 실제 코드 트리(`~/gpu/Self-Forcing`)는 이 머신에 **없다**. 이번
작업의 목표는 그 실행 코드를 **새로 만든 저장소 `~/sf-kvquant`에 깨끗하게 다시
세우는 것**이다.

사용자 요구 (그대로 지킬 것):

1. upstream을 git fork로 두지 않는다. **upstream git 연결이 없는 독립 저장소**로 만들고,
   나중에 사용자 본인의 GitHub에 올린다.
   - push는 사용자가 저장소를 만들고 지시한 뒤에만 한다. 이 머신에는 `gh`가 없다.
2. 출처는 **README와 `vendor/`에 명시**한다.
3. 클론한 저장소에서 **실험에 필요 없는 부분은 빼고**, 필요한 부분은 **다시 구현**한다.
4. 다시 구현한 코드가 **맞는지 확인까지** 한다. 아래 §4의 관문 G1~G5가 완료 기준이다.

## 1. 현재 상태 (Claude가 끝낸 것)

| 항목 | 상태 | 위치 |
|---|---|---|
| upstream 클론 | 완료 | `~/sfq_data/upstream/{Self-Forcing, kv-quant-longhorizon, LongLive}` |
| 라이선스 확인 | 완료 | Self-Forcing **Apache-2.0**, LongLive **Apache-2.0**, kv-quant-longhorizon **라이선스 없음**(→ 코드 재배포 금지, §3) |
| 새 저장소 | 커밋 1개 | `~/sf-kvquant` (`main`, remote 없음) |
| 커밋 `da5438c` | Self-Forcing `33593df`의 추론 경로 파일 20개를 **수정 없이** 복사함. `cmp`로 원본과 동일함을 확인 | `vendor/UPSTREAM.md`, `vendor/LICENSE.self-forcing` |
| Python 환경 | 완료 | `~/sfq_data/env` (conda, py3.10). **반드시 래퍼 `~/sfq_data/py`로 실행**: `PYTHONNOUSERSITE=1`이 없으면 `~/.local`의 패키지가 섞인다 |
| 패키지 | torch 2.6.0+cu124, flash-attn 2.7.4.post1, diffusers 0.31.0, transformers 5.20.2, pyiqa, lmdb, `Quantization/docs/requirements-infer.txt` 전체 | 옛 4090 환경(`Quantization/docs/env.txt`)과 torch·flash-attn 버전이 같다. transformers만 5.17 → 5.20 |
| 체크포인트 | **다운로드 중** (백그라운드, 로그 `~/sfq_data/dl.log`, 끝나면 `DL_DONE`) | `~/sfq_data/ckpt/wan_models/Wan2.1-T2V-1.3B/`, `~/sfq_data/ckpt/checkpoints/self_forcing_dmd.pt` (md5 `ef41ae42bfb87abe43d5f640fb634d0c`, 5,676,252,553 B) |
| GPU | RTX 4090 24GB, RAM 61GB, 디스크 여유 약 90GB → 다운로드 후 약 67GB | |

`da5438c`에 들어 있는 파일 (`inference.py`의 import 폐쇄를 AST로 추적해서 고름):
`LICENSE inference.py configs/{default_config,self_forcing_dmd}.yaml pipeline/{__init__,causal_inference}.py utils/{dataset,lmdb,misc,scheduler,wan_wrapper}.py demo_utils/memory.py wan/__init__.py wan/modules/{__init__,attention,causal_model,model,t5,tokenizers,vae}.py`

**아직 import가 깨져 있다.** `pipeline/__init__.py`, `wan/__init__.py`, `inference.py`가
복사하지 않은 모듈(training pipeline, image2video, distributed, `CausalDiffusionInferencePipeline`,
`TextImagePairDataset`)을 import한다. 이것을 정리하는 것이 T1이다.

## 2. 해야 할 일

커밋은 작업 단위로 나눈다. 첫 커밋(원본 그대로)은 **절대 수정하지 않는다.**
`git diff da5438c -- <file>`이 곧 "upstream 대비 변경 내역"이 되게 하기 위해서다.

### T1. 불필요한 부분 정리 (동작 변화 없음)
- `pipeline/__init__.py`: `CausalInferencePipeline`만 남긴다.
- `wan/__init__.py`: `from . import modules` 정도로 줄인다.
- `inference.py`: i2v, distributed, `CausalDiffusionInferencePipeline` 분기를 제거한다.
  - `low_memory = free VRAM < 40GB` 판정(4090에서는 True)과 DynamicSwap 동작은 **바꾸지 않는다**. bit-exact 비교의 전제다.
- 원본에서 고친 파일에는 Apache-2.0 §4(b)에 따라 파일 상단에 한 줄 표기를 단다
  (예: `# Modified from Self-Forcing 33593df; see vendor/UPSTREAM.md`).

### T2. `kvq/quant.py`: 양자화기 다시 구현
kv-quant-longhorizon 코드를 복사하지 말고 새로 작성한다. 다만 **수치는 아래 명세와
비트 단위로 같아야 한다.** 그래야 옛 실험과 비교할 수 있다.

- 입력 `[B, L, H, D]`. L 방향으로 **16토큰 그룹**(`group=16`)으로 나누고, 끝은 **0으로 패딩**한 뒤 `view(B, nb, 16, H, D)`.
- **RTN (대칭)**
  - 그룹 축(dim 2)으로 `amax(|x|)`
  - `qmax = 2^(b-1) - 1`
  - `scale = (amax / qmax).clamp_min(1e-8)`
  - `q = round(x / scale).clamp(-qmax-1, qmax).to(int8)`
  - scale은 **fp16으로 저장**
  - 복원: `q.to(bf16) * scale.to(bf16)`
  - 연산 dtype은 입력 그대로(bf16)다. **fp32로 올리면 비트가 달라진다.**
- **KIVI-style (비대칭, zero point 버그를 고친 형태)**
  - key: (그룹, head, channel)마다 min과 max
  - value: 토큰마다 dim (3,4)에 대해 min과 max
  - `scale = ((max - min) / (2^b - 1)).clamp_min(1e-8)`
  - `q = round((x - min) / scale).clamp(0, 2^b - 1)`
  - scale과 **min(= offset)** 을 fp16으로 저장
  - 복원: `q * scale + min` (bf16)
  - 배경: `Quantization/patches/kv-quant-longhorizon/08_kivi_asym_fix.patch`, `Quantization/issue/ISSUE_zp_clamp.md`
- key bits와 value bits는 따로 지정한다 (2~8).
- 비트 패킹은 하지 않는다. int8 코드를 그대로 들고 있는 시뮬레이션이다. 메모리 계산은 별도 함수로 낸다.

### T3. KV 저장소: `kvq/store.py` + `wan/modules/causal_model.py` 연결
- `kv_cache` dict에 `"store"` 키가 **없으면 upstream 코드 경로를 한 글자도 건드리지 않는다.** BF16 bit-exact(G2)가 이걸로 보장된다.
- 롤링·삽입 인덱스 계산은 upstream 코드(`causal_model.py` 약 206~236행)와 같아야 한다. 헬퍼로 빼더라도 연산 순서는 유지한다.
- 구현할 저장 정책 두 가지:
  1. **`rolling` (P1, 옛 실험과 호환)**: 매 호출마다 다음을 순서대로 한다.
     1. 상태를 전체 버퍼(`local_attn_size * 1560` 토큰)로 복원한다. 상태가 없으면 0 버퍼.
     2. upstream과 같은 방식으로 롤링·삽입한다.
     3. 그 버퍼로 attention을 계산한다(현재 청크는 방금 쓴 BF16).
     4. **버퍼 전체를 다시 양자화**해 상태로 저장한다.
     5. BF16 버퍼는 비운다.

     기준은 `Quantization/patches/self-forcing/03_causal_model_ucsd_kvquant_plus_errlog.patch`다. G4에서 이 경로가 옛 코드와 bit-exact여야 한다.
  2. **`chunk` (P3, 새 실험용)**: 청크가 확정될 때(다음 청크의 첫 쓰기가 들어올 때, 직전 쓰기는 context refresh) **그 청크만 한 번 양자화**해 세그먼트로 추가한다.
     - 현재 청크는 BF16 pending으로 둔다.
     - 퇴출은 세그먼트 단위로 하고, sink 세그먼트는 남긴다.
     - 세그먼트는 **다시 양자화하지 않는다.**
     - 참고 구현: `Quantization/self_forcing_patches/0003-s9-psac-reference-backend.patch`의 `_psac_attention`. rank 0, bias 없음으로 두면 구조가 같다.
- **residual window (KIVI)는 옛 버그를 고쳐서 구현한다.** 옛 패치 17은 버퍼의 **물리적 끝** `k[:, -resid:]`를 BF16으로 남겼다. 그래서 버퍼가 차기 전(A1 설정에서 청크 0~5)에는 0 패딩을 보호하고 최신 청크를 양자화했다. 올바른 동작은 `local_end_index` 직전의 `resid` 토큰을 보호하는 것이다.
- 설정은 모듈 전역변수나 환경변수로 하지 말고, **pipeline이 store 객체를 만들어 `kv_cache`에 넣는 방식**으로 한다. 옛 코드에서는 프롬프트 사이에 상태가 새는 문제가 있었다.
- 프롬프트가 바뀔 때 store를 초기화한다. upstream reset 블록(`causal_inference.py` 약 131~139행)과 같은 위치에서 한다.

### T4. 재현성 장치 (옛 패치 01/02와 같은 규약)
- `--rng chunk`
  - 초기 노이즈: 청크별 generator, 시드 `seed * 1000003 + chunk_idx`, 형상 `[num_samples, nfpb, 16, 60, 104]` bf16
  - 재노이징: 청크별 generator, 시드 `seed * 1000003 + 7919 + chunk_idx`, `torch.randn(shape, generator=g, dtype=flat_pred.dtype)`
  - `set_seed(seed)`는 그대로 호출한다.
- `--rng release`: upstream과 같이 전역 `torch.randn` 사용(기본값). 기준은 `Quantization/patches/self-forcing/01_*.patch`, `02_*.patch`.
- 청크별 latent 덤프: `{"latent": fp32 cpu, "chunk_idx", "start_frame", "num_frames"}`, 파일명 `chunk_{idx:04d}.pt`, 프롬프트별 디렉터리.
- KV 오차 로그: 호출마다 `(prompt, chunk, call_phase, layer, k_rel, v_rel, tokens)`를 원시 기록한다.
  - 옛 지표 정의: 재양자화 후 복원값과 양자화 직전 버퍼의 `[:, :local_end_index]` 상대 L2. 계산은 `.float()` 후 `norm`, 분모는 `clamp(min=1e-12)`.
  - 참고: 이 지표는 연구에서 이미 철회한 지표다(CORRECTIONS C-I: 오래된 슬롯은 Q(x̂)−x̂라 0). **G5 재현용으로만** 쓰고, 새 실험용으로는 BF16 원본 대비 오차를 별도로 기록한다.

### T5. README / NOTICE
- README에 넣을 내용:
  - 목적
  - 출처 표 (`vendor/UPSTREAM.md` 링크)
  - 설치 (래퍼, 체크포인트 받는 명령은 `~/sfq_data/dl.sh` 참고)
  - 실행 예시
  - 검증 결과 표 (§4)
- `NOTICE`: Self-Forcing과 Wan2.1 고지, 그리고 "이 저장소에서 수정한 파일 목록".

## 3. 함정 (이미 확인된 것)

1. **kv-quant-longhorizon은 라이선스가 없다.** 코드를 저장소에 복사하지 않는다.
   - 테스트는 `KVQ_REF_REPO` 환경변수로 로컬 체크아웃을 받았을 때만 비교한다.
   - 비교 기준은 **패치 17을 적용한 상태**다. 17은 05·08을 누적 포함한 diff라 단독으로 적용한다.
2. **`Quantization/patches/self-forcing/`의 03·09·15·16은 모두 원본(`98e398a`) 기준 누적 diff다.** 차례로 쌓으면 실패한다.
   - 옛 트리 재현 = 원본 + `01` + `02` + `16`. 셋 다 `git apply`로 깨끗하게 적용되는 것을 확인했다.
   - 16에는 TUM/sink 기능도 들어 있지만, 플래그를 켜지 않으면 동작하지 않는다.
3. **`self_forcing_patches/`(세션 7~9)는 적용되지 않는다.** 기준 커밋 `e107a36`이 어디에도 없고, fused Triton 커널(`kernels/triton_kv_quant.py`) 소스도 이 머신과 저장소 어디에도 없다. 이번 범위에서는 제외한다.
4. 기본 config(`self_forcing_dmd.yaml`)는 `local_attn_size = -1`이다. 이 상태에서는 21 latent 프레임을 넘으면 shape 오류가 난다. 63프레임 실험은 `Quantization/patches/self-forcing/configs/gateA_A1.yaml`(`local_attn_size: 21`, `sink_size: 0`, A2는 sink 3)을 쓴다. 이 config는 저장소에 새로 작성해 넣는다(설정 파일이므로 문제 없음).
5. `Quantization/tools/tum_correction.py`는 Δ = 2·scale을 쓰는 **감사 전 사본**이다. 올바른 값은 Δ = scale이다. 이번 범위는 아니지만 가져오지 않는다.
6. head 수 12와 head_dim 128이 upstream 곳곳에 하드코딩돼 있다. 1.3B 전용임을 README에 적는다.
7. 옛 `kv_err` 집계는 경계가 어긋나 있다. 청크 c의 묶음 = 직전 청크의 refresh 호출(30) + 청크 c의 디노이징 호출 4×30. 그래서 첫 프롬프트 청크 0은 120개, 이후 프롬프트의 청크 0은 150개다(이전 프롬프트 마지막 refresh가 섞임). G5 비교 스크립트가 원시 기록에서 이 규칙대로 묶어야 한다.

## 4. 검증 관문 (완료 기준)

모든 관문의 결과(명령, 수치, 통과 여부)를 `docs/VERIFY.md`에 표로 남긴다. 실패하면
원인을 찾기 전에 기준을 바꾸지 않는다.

| 관문 | 내용 | 통과 기준 |
|---|---|---|
| **G1 단위** | `tests/test_quant.py`: 무작위 `[1, L, 12, 128]` bf16을 넣고 RTN/KIVI × bits {2,3,4,8} × L {16의 배수, 4680, 32760}로 우리 구현과 참조(`KVQ_REF_REPO`, 패치 17)의 복원값 비교. 상수 그룹, 전부 양수 그룹, 0 그룹 같은 경계 사례 포함 | `torch.equal` (비트 동일) |
| **G1b 저장소 단위** | GPU 없이 합성 텐서로 시험: `rolling`·`chunk` store의 인덱스·퇴출·sink 유지. `chunk` store의 세그먼트가 이후 호출에서 비트 단위로 변하지 않음. residual window가 버퍼가 차기 전에도 최신 토큰을 보호함 | assert |
| **G2 릴리스 BF16** | 원본 `~/sfq_data/upstream/Self-Forcing` 사본과 우리 저장소를 `--rng release`, seed 0, 기본 config, 21프레임, 프롬프트 3개로 돌려 최종 latent 비교 | 3/3 `torch.equal` |
| **G3 RNG 관문** | `--rng chunk`에서 청크마다 전역 CUDA RNG를 7777회 소모하는 훅을 켜고/꺼서 비교 (옛 연구의 핵심 관문) | bit-exact |
| **G4 옛 코드 대비** | 옛 트리(원본 + 01+02+16, kv-quant + 17, 스크래치 위치에 구성)와 우리 `rolling` store를 같은 설정(A1 config, 63프레임, `--rng chunk`, seed 0, 프롬프트 2개)으로 BF16 / RTN INT4 / RTN INT2 / KIVI INT4(residual 0) 실행 | 청크별 latent 전부 `torch.equal` |
| **G5 저장 수치 재현** | `Quantization/results/kivi_rw/kivi4_rw0_kv_err.json`(옛 4090, `Quantization/docs/prompts10.txt` 10개, A1, KIVI INT4, residual 0)을 우리 코드로 재현. 첫 값은 prompt0 chunk0 `k_rel_mean = 0.037011147709563375`, `v_rel_mean = 0.14003970734775067`, `n_records = 120` | 모든 행이 상대오차 1e-6 이내. 비트 단위 일치는 기대하되 요구하지 않는다(머신이 다르다). 불일치하면 원인 조사 후 보고 |
| (선택) G6 MUSIQ | `Quantization/results/gateA_metrics_A1_per_prompt.json`의 `A1_bf16` 청크별 프롬프트 MUSIQ와 비교 | pyiqa 버전 차이를 고려해 보고만 한다 |

G4·G5를 통과하면 "다시 구현한 경로가 옛 실험 코드와 같다"고 쓸 수 있다. `chunk` store는
비교할 옛 기준이 없으므로 G1b와, 같은 실행에서 `rolling` store와의 차이가 0이 아님을 보이는
것까지만 한다.

## 5. 하지 말 것

- GitHub 생성이나 push. 사용자 지시를 기다린다.
- 첫 커밋 `da5438c` 수정이나 rebase.
- kv-quant-longhorizon 코드 복사.
- LongLive 포팅, TF/FR 실험, 새 측정. 이번 범위는 **재구현과 검증까지**다.
- 체크포인트 다운로드가 끝나기 전에 G2~G5 실행. `~/sfq_data/dl.log`에 `DL_DONE`이 찍혔는지 확인하고, md5도 확인한다.

## 6. 보고

끝나면 다음을 사용자에게 보고한다.
- 커밋 목록
- `docs/VERIFY.md`의 표
- 실패했거나 건너뛴 관문과 그 이유
- 남은 위험

수치는 실행 로그의 값을 그대로 옮긴다.

## 참고 경로

- 연구 문서 진입점: `~/Quantization/문서/INDEX.md`, `~/Quantization/문서/결과서/S6_FINDINGS.md`
- 옛 환경과 설치 메모: `~/Quantization/docs/{env.txt, SETUP.md}`
- 옛 패치: `~/Quantization/patches/`
