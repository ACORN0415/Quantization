# §2 인용 확인 — 2026-09-20

확인 방법: arXiv 초록·HTML 본문과 공개 저장소를 직접 조회. 상태는 셋이다.
`[확인]` 원문에서 직접 확인 · `[2차]` 2차 출처만 확인 · `[미확인]` 이번에 보지 않음.

> **먼저 읽을 것: §5의 조사 누락.** 이번 확인에서 **§2의 위치 설정 문장을 바꿀 수 있는 사실**이 나왔다. 서지 정리보다 그쪽이 중요하다.

---

## 1. Quant VideoGen (QVG) — `[확인]`

**서지.** *Quant VideoGen: Auto-Regressive Long Video Generation via 2-Bit KV-Cache Quantization*, arXiv **2602.02958**, ICML 2026. 교신저자 Kurt Keutzer, Chenfeng Xu. 코드 `github.com/svg-project/Quant-VideoGen`, 프로젝트 페이지 `svg-project.github.io/qvg/`.

**설정이 둘이다 — 하나로 뭉치지 않는다.** 원문 §5.1은 **QVG를 S=1, B=64**, **QVG-Pro를 S=4, B=16**으로 구분한다. **QVG-Pro의 블록 16은 우리와 같은 크기**이므로, 비교 서술에서 어느 설정을 말하는지 반드시 밝힌다.

**§2가 인용하는 문장 — 원문 확인됨.** §5.1(Setup):

> "We use streaming chunk-wise compression to quantize KV-cache once per chunk and avoid re-compression drift"

**우리 §2 서술의 정정.** 현재 "learned codebooks with a fused dequantizing attention kernel"이라 썼는데 더 정확히는 — **의미 기반 평활화 + k-means 군집**으로 **센트로이드 K=256(uint8 인덱스)**, **Progressive Residual Quantization**, 블록 크기는 **설정에 따라 64 또는 16**(위 참조). fused 커널은 §4.3에 있고 "dequantizes the tensor and adds back the assigned centroids for all stages"다. **스칼라 RTN이 아니다** — 우리와 양자화 방식이 근본적으로 다르다는 점을 §2에 명시해야 비교가 공정하다.

## 2. UCSD 33-method 연구 — `[확인]` **여기가 중요하다**

**서지.** *KV Cache Quantization for Self-Forcing Video Generation: A 33-Method Empirical Study*, **Suraj Ranganath, Vaishak Menon, Anish Patnaik** (University of California, San Diego), arXiv **2603.27469v1**, 2026-03-29. 코드 `github.com/suraj-ranganath/kv-quant-longhorizon`.

**§2가 인용하는 주장 — 원문 확인됨.** 초록: "the current integration reconstructs or retains large BF16 buffers during attention and refresh stages". §5.4: "some quantized methods compress the KV cache substantially and still exceed BF16 peak VRAM ... several methods still reconstruct dense BF16 tensors during attention reads". 현재 우리 문장("dominant practical cost")보다 원문이 더 구체적이므로 원문 표현에 맞춘다.

**그리고 이것이 §5의 문제다** — 아래.

## 3. Tuncer, Becker & Pfeil (내부 호칭 "TUM") — `[확인]` 일부

**서지.** *Quantized Keys Steal Attention: Bias Correction for KV-Cache Compression in Video Diffusion*, arXiv **2605.26266**.
저자·소속(원문 표기): **Tuna Tuncer**(¹Technical University of Munich, ²Tensordyne), **Felix Becker**(Tensordyne), **Thomas Pfeil**(Tensordyne).

**내부 호칭 "TUM"을 논문에서 쓰지 않는다** — 1저자만 TUM 소속이고 나머지는 Tensordyne이다. 본문·문서 전체에서 **Tuncer et al.**로 통일할 것.

`[2차]` ICML 2026 워크숍(SCALE, F2S) 채택 — 트위터 게시물에서만 확인. 원문에 게재 정보 없음.
`[확인]` 양자화 방식: **그룹별 per-token** — "divides each token's d channels into groups of size g, with an independent (Δ, z) per group j", 그림에서 **g=32**. 평가 모델 MAGI-1 · SkyReels-V2 · HY-WorldPlay, 비대칭 INT2(QuaRot+RTN).
`[확인]` **확인한 논문 원문에서 코드 링크를 찾지 못했다** — GitHub·프로젝트 페이지·코드 공개 언급이 본문에 없다. **공개 코드가 없다는 증명은 아니다**(저자 페이지·다른 경로에 있을 수 있고, 찾아보지 않았다).

> **§2에 넣을 대조 한 줄.** 그들은 **토큰 안에서 채널 축**으로 묶고(g=32), 우리는 **토큰 축**으로 묶는다(16토큰). 그룹이 토큰을 가로지르지 않으면 퇴출이 그룹 구성을 바꿀 수 없다 — 축의 선택이 이 논문의 조건과 직접 관련된다. **단, 이 함의는 논문 서술에서 끌어낸 우리 해석이며 그들 구현으로 확인한 것이 아니다**(코드를 찾지 못했다).

## 4. KIVI — `[확인]`

*KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache*, arXiv **2402.02750**, ICML 2024. 코드 `github.com/jy-yuan/KIVI`(커널 포크 `max410011/KIVI-kernel`).
README: "quantizing the key cache per-channel and the value cache per-token to 2bit", 설정 인자 `residual_length`("the number of recent fp16 tokens"). 구조: `quant/`(CUDA), `models/llama_kivi.py`, `config/`.

## 5. **조사 누락 — §2를 확정하기 전에 반드시 해결할 것**

UCSD 33-method 연구는 우리 조건과 정면으로 겹친다. `[확인]` **Appendix A Table 3**:

> "Shared defaults across the harness include blockwise KV quantization at the causal cache boundary with **block size 16**"

그리고 RTN 변형은 "symmetric blockwise quantization of both K and V". **즉 같은 모델 계열(Self-Forcing), 같은 블록 크기 16, 같은 대칭 blockwise RTN이고, 코드가 공개돼 있다.**

**둘 중 하나로 갈리지 않는다.** 공개 코드에 그 조건이 있다는 것만으로 **논문의 모든 평가가 그 조건에서 수행됐다는 결론은 나오지 않는다** — 논문이 쓴 **커밋·설정·실행 경로**와의 연결까지 확인해야 하고, **일부 경로에만 존재하는 경우**를 따로 구분해야 한다(조사 명세의 "경로 단위" 규칙이 여기에도 적용된다).

**조사 기준 커밋.** `suraj-ranganath/kv-quant-longhorizon` **`b4c09363735c89947222781db899862bc0dc55c5`** (`main`, 2026-09-20 조회). 아래 인용은 모두 이 커밋의 파일이며, 고정 커밋에서 다시 받아 대조했다 — `ls-remote` 해시를 앞선 조회에 사후로 붙인 것이 아니다.

| 확인해야 할 것 | §2가 말할 수 있게 되는 것 | 상태 |
|---|---|---|
| 공개 코드의 **어느 경로**에 구조가 있는가 | 구조를 가진 경로의 범위 | **답함 (5.1)** |
| 우리 패치가 그 저장소 대비 **무엇을 바꿨는가** | 구조가 그들 것인지 우리 것인지 | **답함 — 동일하지 않다 (5.3)** |
| 공개 설정에서 **실제로 격자가 밀리는가** | 구조에서 현상으로 넘어갈 수 있는지 | **답함 — 밀리지 않는다 (5.2)** |
| 쓰기 시 **경계 그룹 재구성**은 남는가 | 퇴출과 별개의 저장 이력 효과 | **미확인 (5.2)** |
| 논문이 보고한 실행이 **그 커밋·설정·경로를 썼는가** | 출판된 평가와의 연결 여부 | **미확인 (5.4)** |

다 채워지기 전에는 "출판된 평가가 영향받는다"도 "우리 패치가 유일하다"도 쓰지 않는다.

> **가장 중요한 한 줄.** 양자화 패치에 이동 코드가 있다는 사실과, 우리 **유한 rolling 캐시**에서 그 분기가 실행된다는 사실을 연결하면 안 된다. 비교의 핵심은 패치가 아니라 **드라이버의 캐시 용량 정책**이다.

---

### 5.1 공개 코드의 경로 — **확인함** `[확인: 고정 커밋 패치 직접 읽음]`

`scripts/10_clone_deps.sh` → `guandeh17/Self-Forcing`을 `third_party/Self-Forcing`으로 clone.
`scripts/11_apply_self_forcing_patch.sh` → `docs/patches/self_forcing_kv_quant.patch`를 `git apply`.
그 패치는 `wan/modules/causal_model.py`의 `CausalWanSelfAttention` 한 곳(@@ -197,11 +197,33 @@, blob `98e398a..5c0f839`)만 고친다. 로컬 pristine **`33593df`**의 `causal_model.py`에 **문맥까지 일치하며 적용된다**(메모리 안에서만 적용해 확인; 파일은 건드리지 않았다).

패치가 하는 일을 호출 순서대로 적으면:

```
cache_k, cache_v = quantizer.dequantize_kv(kv_cache["quant_state"], ...)   # 캐시 전체 복원
cache_k[:, sink_tokens : sink_tokens+num_rolled_tokens] = \
    cache_k[:, sink_tokens+num_evicted_tokens : ...]                      # 퇴출량만큼 이동
cache_k[:, local_start_index:local_end_index] = roped_key                  # 새 토큰 기록
kv_cache["quant_state"] = quantizer.quantize_kv(cache_k, cache_v, ...)     # 캐시 전체 재양자화
```

**확정할 수 있는 것은 여기까지다 — 구조다.**

> 공개 패치는 우리 P1과 같은 **복원·이동·전체 재양자화 구조**를 갖는다.

(i) 재양자화 입력이 복원값이고, (ii) 이동량 `num_evicted_tokens`를 그룹 크기로 반올림하는 코드가 **패치 어디에도 없으며**, (iii) 재양자화가 새로 쓴 구간이 아니라 **캐시 전체**에 걸린다. `kv_quant/rtn.py`의 `RTNQuantizer`는 `block_size: int = 16`을 기본값으로 **토큰 축**에 그룹을 잡고 그룹마다 스케일을 둔다 — 우리와 같은 축, 같은 크기다.

**"같은 조건이다"라고 쓰지 않는다.** 구조를 갖는다는 사실과 특정 설정에서 실제로 격자가 이동한다는 사실은 다른 진술이고, **§5.2에서 후자는 성립하지 않는 것으로 추적됐다.**

**"우리 조사 범위에서 P1형 경로는 우리 패치뿐이다"도 여전히 쓸 수 없다.** 구조는 공개 패치에 있다. 동시에 **"공개 코드도 우리와 같은 문제를 겪는다"도 쓸 수 없다** — 그 드라이버에서는 분기가 실행되지 않는다(§5.2). 두 방향 모두 막혀 있고, 쓸 수 있는 것은 **구조의 존재와 실행 조건의 부재를 함께 적는 것**뿐이다.

**용어 충돌 주의.** 그들의 `EXPERIMENTS.md`에서 "block"은 **3 latent frames**(`num_frame_per_block = 3`)이고, `rtn.py`의 `block_size=16`은 **양자화 그룹**이다. 논문 Table 3의 "block size 16"은 후자다. 우리 본문에서 두 뜻을 같은 단어로 쓰지 않는다.

### 5.2 구조에서 현상으로 — **연결되지 않는다** `[확인: 정적 추적]`

**드라이버가 캐시를 미리 키운다.** 고정 커밋의 `scripts/01_generate.py` 순서:

| 단계 | 코드 동작 |
|---|---|
| 캐시 초기화 | `_initialize_kv_cache()` 호출 |
| **용량 확장** | `ensure_kv_cache_capacity()`가 K/V 버퍼를 **`num_output_frames × frame_seq_length` 이상**으로 확장 |
| 용량 기록 | 확장된 길이를 각 레이어의 `kv_cache_size`에 저장 |
| 프롬프트 시작 | 인덱스·양자화 상태를 초기화하되 **확장한 용량을 줄이지 않음** |
| 생성 | 지정한 `num_output_frames`만큼 생성 |

수치로 확인하면, 42 latent frames · 프레임당 1,560토큰이면 필요한 용량은 42 × 1,560 = **65,520토큰**이다. 3프레임 chunk는 4,680토큰이고 총 **14개**이므로 마지막 chunk를 써도 누적 길이는 정확히 65,520 — 미리 확보한 용량을 **초과하지 않는다**. 공개 패치의 퇴출 조건은 `num_new_tokens + local_end_index > kv_cache_size`이므로 **성립하지 않는다**.

대조한 pristine `33593df`의 기본 모델 설정은 **`local_attn_size = -1`**이고, 퇴출 분기는 `local_attn_size != -1`도 요구한다. 다만 의존 Self-Forcing 커밋을 공개 harness가 고정하지 않으므로 **이것은 "대조한 의존 코드의 조건"으로만 쓴다.**

> **§5.2의 결론 문장 (본문에 그대로 쓸 것).** 고정한 공개 harness는 생성 전에 캐시를 전체 출력 길이 이상으로 확장한다. 대조한 Self-Forcing 의존 코드와 이 드라이버를 결합한 RTN 경로에서는, 지정된 생성 길이 내 **용량 초과에 의한 퇴출이 발생하지 않는 것으로 정적 추적된다**. 따라서 chunk 크기의 나머지가 8이라는 사실만으로 이 경로에 **퇴출 유발 격자 이동이 있다고 할 수 없다.**

**여기서 추론하면 안 되는 것.** "퇴출이 없으니 저장 이력 효과가 전부 없다"는 따라 나오지 않는다. **쓰기 시 경계 그룹 재구성(경로 A)은 퇴출과 별개로 남을 수 있다** — 캐시 전체를 매 쓰기마다 재양자화하고 꼬리의 그룹이 부분적으로만 채워져 있다면, 다음 쓰기에서 그 그룹은 다른 구성원으로 다시 형성된다. **이건 아직 확인하지 않았다** `[미확인]`.

**4,680 mod 16 = 8은 그대로 참이지만 여기서는 쓰이지 않는다.** 이동이 일어날 때만 의미가 있는 수이고, 이 경로에서는 이동이 일어나지 않는다. (덧붙여 65,520 mod 16 = 0이다.)

### 5.3 우리 패치와의 차이 — **확인함** `[확인]`

공개 패치를 적용한 코드와 현재 우리 `causal_model.py`의 차이는 **498행 추가·7행 삭제**다.

**따라서 "공개 패치를 그대로 쓰고 있다"고 쓰면 안 된다.** 동시에 복원 → 이동 → 기록 → 전체 재양자화라는 **기본 경로는 공개 패치에 이미 존재한다** — 우리가 도입한 것이 아니다. 두 진술을 함께 써야 정확하다.

**구현 위치도 정정한다.** `kv_quant/base.py`는 **추상 인터페이스**이고 실제 양자화 연산은 `rtn.py`와 `utils.py`에 있다. 우리 로컬 `base.py`·`utils.py`는 공개본과 다르므로, **원본 구현을 인용할 때는 고정 커밋 `b4c0936`의 파일을 기준으로 읽는다.**

### 5.4 논문 실행과의 연결 — **미확인** `[미확인]`

`scripts/10_clone_deps.sh`는 세 저장소를 clone하면서 `--branch`도 `git checkout <sha>`도 쓰지 않고, `EXPERIMENTS.md`에도 커밋 해시가 없다. 정확한 진술은 이것이다.

> **확인한 자료에서는 논문 실행에 사용된 Self-Forcing 커밋을 특정하지 못했다.**

로그·산출물·커밋 이력까지 조사하지는 않았으므로, **"조사를 더 해도 닫히지 않는다"고 단정하지 않는다.** 현재 미확인이라는 사실만 기록한다. §2는 "공개 코드의 해당 경로가 이 구조를 갖는다"까지만 쓰고, "출판된 평가가 영향받는다"는 쓰지 않는다.

### 5.5 남은 작업

1. **쓰기 시 경계 그룹 재구성**이 공개 경로에 남는지 (§5.2 마지막 문단). 퇴출이 없어도 성립할 수 있으므로 별도 확인이 필요하다. GPU 불필요.
2. 논문 실행에 대한 귀속은 **위가 다 확인돼도 별도로 유지**한다 (§5.4).

### 5.6 이 조사가 바꾼 비교의 초점

처음에 우리는 **양자화 패치**를 비교 대상으로 잡았다. 그게 틀렸다. 패치에 이동 코드가 있는지는 조건의 절반일 뿐이고, 나머지 절반은 **드라이버가 캐시를 어떻게 잡는가**다. 공개 harness는 전체 출력 길이를 미리 확보하고, 우리는 유한 rolling 창을 쓴다 — 같은 패치를 놓고도 한쪽에서는 퇴출 분기가 실행되지 않는다.

**따라서 앞으로 다른 구현과 비교할 때는 양자화 방식과 함께 캐시 용량 정책을 같은 비중으로 기록한다.** 이 구분은 §2뿐 아니라 §10(재현 조건)에도 들어가야 한다.

---

## 6. 아직 확인하지 않은 것 — `[미확인]`

KV-AdaQuant(arXiv 2502.15075) · Deep Forcing(2512.05081) · Self-Forcing · LongLive · SkyReels-V2 · MAGI-1 · VBench · MUSIQ · FlashAttention-2.
로컬 노트에 arXiv 번호는 있으나 **이번에 원문을 열지 않았다.** §2가 이들에 대해 하는 진술(예: "LongLive retrains for a short cache", "Deep Forcing adjusts sink and window without retraining")은 아직 근거 위치가 없다.

## 7. 본문에 반영할 정정 요약

| 위치 | 현재 | 교체 |
|---|---|---|
| §2 KV 양자화 | QVG "learned codebooks" | k-means 센트로이드(K=256) + Progressive Residual Quantization, 블록은 **QVG B=64 / QVG-Pro B=16**. **스칼라 RTN이 아님**과 **설정 구분**을 함께 명시 |
| §2 KV 양자화 | "A recent study surveys 33 methods and reports the transient BF16 reconstruction buffer as their dominant practical cost" | 원문 표현으로: 여러 방법이 캐시를 크게 압축하고도 **BF16 peak VRAM을 초과**하며, attention 읽기에서 dense BF16 텐서를 재구성한다 |
| 전체 | "TUM" | **Tuncer et al.** |
| §2 새 문장 | — | 그룹 축 대조(그들 채널 축 g=32 / 우리 토큰 축 16). **우리 해석이며 그들 구현으로 확인하지 않음**을 함께 |
| §2 위치 설정 | "우리 조사 범위에서 유일한 경로는 우리 패치" | **삭제.** 공개 저장소의 `self_forcing_kv_quant.patch`가 같은 구조를 갖는다(§5.1) |
| §2 위치 설정 (대체) | — | "공개 harness(`b4c0936`)의 해당 경로는 캐시 전체를 복원 → 이동 → 재양자화하며 그룹 정렬 코드가 없다 — **구조에 대한 진술이다.** 다만 그 드라이버는 생성 전에 캐시를 전체 출력 길이 이상으로 확장하므로, 대조한 의존 코드와 결합한 RTN 경로에서는 지정 생성 길이 내 **용량 초과 퇴출이 발생하지 않는 것으로 정적 추적된다.** 논문이 보고한 실행이 이 커밋·설정·경로를 썼는지는 확인한 자료로 특정하지 못했다." |
| §2·§10 | 비교 항목이 양자화 방식뿐 | **캐시 용량 정책을 같은 비중으로 기록**한다(§5.6) |
| §2 | "우리는 공개 패치를 그대로 쓴다" 류의 서술 | **쓰지 않는다.** 우리 코드는 공개 패치 적용본 대비 498행 추가·7행 삭제(§5.3) |
| `sec4_5` 집필 메모 | "조사에서 P1형 경로는 우리 패치 하나였다" | **틀렸다 — 삭제** |
