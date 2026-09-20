# §2 인용 확인 — 2026-09-20

확인 방법: arXiv 초록·HTML 본문과 공개 저장소를 직접 조회. 상태는 셋이다.
`[확인]` 원문에서 직접 확인 · `[2차]` 2차 출처만 확인 · `[미확인]` 이번에 보지 않음.

> **먼저 읽을 것: §5의 조사 누락.** 이번 확인에서 **§2의 위치 설정 문장을 바꿀 수 있는 사실**이 나왔다. 서지 정리보다 그쪽이 중요하다.

---

## 1. Quant VideoGen (QVG) — `[확인]`

**서지.** *Quant VideoGen: Auto-Regressive Long Video Generation via 2-Bit KV-Cache Quantization*, arXiv **2602.02958**, ICML 2026. 교신저자 Kurt Keutzer, Chenfeng Xu. 코드 `github.com/svg-project/Quant-VideoGen`, 프로젝트 페이지 `svg-project.github.io/qvg/`.

**§2가 인용하는 문장 — 원문 확인됨.** §5.1(Setup):

> "We use streaming chunk-wise compression to quantize KV-cache once per chunk and avoid re-compression drift"

**우리 §2 서술의 정정.** 현재 "learned codebooks with a fused dequantizing attention kernel"이라 썼는데 더 정확히는 — **의미 기반 평활화 + k-means 군집**으로 **블록 64, 센트로이드 K=256(uint8 인덱스)**, 그리고 **Progressive Residual Quantization**. fused 커널은 §4.3에 있고 "dequantizes the tensor and adds back the assigned centroids for all stages"다. **스칼라 RTN이 아니다** — 우리와 양자화 방식이 근본적으로 다르다는 점을 §2에 명시해야 비교가 공정하다.

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
`[확인]` **공개 코드 없음** — 논문에 GitHub·프로젝트 페이지·코드 공개 언급이 전혀 없다.

> **§2에 넣을 대조 한 줄.** 그들은 **토큰 안에서 채널 축**으로 묶고(g=32), 우리는 **토큰 축**으로 묶는다(16토큰). 그룹이 토큰을 가로지르지 않으면 퇴출이 그룹 구성을 바꿀 수 없다 — 축의 선택이 이 논문의 조건과 직접 관련된다. **단, 이 함의는 우리 해석이며 그들 구현으로 확인한 것이 아니다**(코드 없음).

## 4. KIVI — `[확인]`

*KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache*, arXiv **2402.02750**, ICML 2024. 코드 `github.com/jy-yuan/KIVI`(커널 포크 `max410011/KIVI-kernel`).
README: "quantizing the key cache per-channel and the value cache per-token to 2bit", 설정 인자 `residual_length`("the number of recent fp16 tokens"). 구조: `quant/`(CUDA), `models/llama_kivi.py`, `config/`.

## 5. **조사 누락 — §2를 확정하기 전에 반드시 해결할 것**

UCSD 33-method 연구는 우리 조건과 정면으로 겹친다. `[확인]` **Appendix A Table 3**:

> "Shared defaults across the harness include blockwise KV quantization at the causal cache boundary with **block size 16**"

그리고 RTN 변형은 "symmetric blockwise quantization of both K and V". **즉 같은 모델 계열(Self-Forcing), 같은 블록 크기 16, 같은 대칭 blockwise RTN이고, 코드가 공개돼 있다.**

**그런데 G1 조사표에 이 경로가 없다.** 조사한 것은 KIVI 원본·QVG 원본·"우리가 쓴 Self-Forcing용 패치" 셋이다. 우리 문서가 그 패치를 계속 **"UCSD 패치"**라고 부르는데, 그렇다면 다음 둘 중 하나다.

| 경우 | §2가 말할 수 있는 것 |
|---|---|
| **재양자화 조건이 그들의 공개 코드에 있다** | 우리 조건 위에서 수행된 **출판된 33-method 평가가 존재한다.** §2의 "우리 조사 범위에서 이 조건을 보인 유일한 경로는 우리가 쓴 패치뿐"은 **틀린 문장이 된다** |
| **우리 패치가 그 조건을 도입했다** | 현재 §2가 맞다. 다만 **"우리 패치가 UCSD 공개 코드에서 무엇을 바꿨는가"를 명시**해야 하고, 그 차이가 우리 결과의 원인이라는 점을 분명히 써야 한다 |

**둘 중 무엇인지 지금은 모른다.** 저장소에 `kv_quant/` 모듈이 있으나 README는 퇴출·재양자화 로직을 기술하지 않는다 — **소스를 직접 읽어야 한다.**

**할 것(GPU 불필요).** G1 조사표에 다음 두 행을 추가한다.
1. `suraj-ranganath/kv-quant-longhorizon @ <commit>` · RTN 설정 · 기본 backend · 캐시 쓰기/퇴출 호출 경로
2. 우리 패치 — **그 저장소 대비 diff**를 함께 기록(무엇을 바꿨는가)

조사 명세의 Q1~Q5를 그대로 적용하고, Q2-a(codes·scales·그룹 대응 보존)와 Q2-c(그룹 구성·스케일 변화)를 **파일:라인 @commit**으로 답한다. **이 두 행이 채워지기 전에는 §2의 위치 설정 문장을 확정하지 않는다.**

---

## 6. 아직 확인하지 않은 것 — `[미확인]`

KV-AdaQuant(arXiv 2502.15075) · Deep Forcing(2512.05081) · Self-Forcing · LongLive · SkyReels-V2 · MAGI-1 · VBench · MUSIQ · FlashAttention-2.
로컬 노트에 arXiv 번호는 있으나 **이번에 원문을 열지 않았다.** §2가 이들에 대해 하는 진술(예: "LongLive retrains for a short cache", "Deep Forcing adjusts sink and window without retraining")은 아직 근거 위치가 없다.

## 7. 본문에 반영할 정정 요약

| 위치 | 현재 | 교체 |
|---|---|---|
| §2 KV 양자화 | QVG "learned codebooks" | k-means 센트로이드(블록 64, K=256) + Progressive Residual Quantization. **스칼라 RTN이 아님**을 명시 |
| §2 KV 양자화 | "A recent study surveys 33 methods and reports the transient BF16 reconstruction buffer as their dominant practical cost" | 원문 표현으로: 여러 방법이 캐시를 크게 압축하고도 **BF16 peak VRAM을 초과**하며, attention 읽기에서 dense BF16 텐서를 재구성한다 |
| 전체 | "TUM" | **Tuncer et al.** |
| §2 새 문장 | — | 그룹 축 대조(그들 채널 축 g=32 / 우리 토큰 축 16). **우리 해석이며 그들 구현으로 확인하지 않음**을 함께 |
| §2 위치 설정 | "우리 조사 범위에서 유일한 경로는 우리 패치" | **§5가 해결될 때까지 보류** |
