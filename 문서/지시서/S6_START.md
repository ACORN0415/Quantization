# 세션 6 시작 상태 (세션 5가 작성, 2026-09-18)

## 커밋 (이 두 해시가 세션 6의 기준)
- Self-Forcing **`7f07533`** = 45836f4 + patches/{22,24,25,26}  (커밋 11개)
- kv-quant-longhorizon `650ce90` = 226c405 + patches/23
- 재현 절차: `~/gpu/patches/README.md` (22 → 24 → 25 → 26, 23은 kv-quant 쪽)

## 관문 통과 기록 (재검증 불필요)
- 플래그 OFF A1_int4 21f×3p vs 세션 3 latent: **21/21 bit-exact**
- legacy TUM(`--tum_include_self`) vs 감사 closeout: **21/21 bit-exact**
- self-제외 ON에서 chunk 0이 λ=0과 **3/3 bit-exact**(chunk 1은 3/3 상이)
- 오라클 드라이버 `oldest`가 BF16에서 A1_bf16과 **63/63 bit-exact**
- Triton pack 커널이 UCSD 경로와 **bit-exact 6/6**(실제 캐시 형상 포함)
- fused 커널 등가 관문 1~3 + 세그먼트 모드 PASS (max|diff| ≤ 9.8e-4)

## 세션 5가 확정한 것 (요약, 상세는 RESULTS_session5_gates.md)
1. **감사 버그 E**: TUM의 "회복"은 낙폭 지표 + self 슬롯 bias 오적용의 합작.
   self 제외 + 평균 MUSIQ에서 λ=1이 **+10.04 (t=3.27)**, 피해의 34% 회복.
   λ=4 우위(감사 B8b)는 **소멸**(λ4−λ1 = +2.08, t=0.69 n.s.).
2. **λ\* = 1 확정.** TF(오염 0)에서 λ*_TF=1이고 λ=4는 −4.46로 해롭다.
3. **배분**: (a) 비균일 양자화 오차 ≈ 0(ε 분산비 0.94~0.98), (c) 0 아닌 평균 ≈ 0
   (|mean|/σ ≤ 0.003), (b) 전파 오염이 실효 노이즈를 약 2배로(분포 복원점 λ≈2).
4. **Jensen 구조의 기여는 0** — flatten 컨트롤 +10.49 ≈ taylor +10.04.
   메커니즘은 "캐시를 self 대비 스칼라로 낮추기". 실측 ε 보정은 무효(캐시에
   실현 오차가 없음 — 저장이 멱등).
5. **관문 I**: 키 비트가 지배(값 비트는 무관). A4 구조 + K3V2/KIVI-K2V2 + λ=1 보정이
   **0.75GB(8× 압축)에서 −0.72~−1.11 MUSIQ** — INT4(1.89GB, −4.61) 대비 메모리 2.5배↓
   화질 3.9pt↑. 키 3비트가 하한선(K2V2는 −17.6).
6. **관문 J**: quantize 커널 9.2배, 패킹 메모리 6.04→1.89/1.13GB(첫 실측),
   fused v2 8.17ms(FA2 +13.1%), 종단 51.3s/프롬프트(dequant 70.1s, −27%).
7. **격자 계보 효과(신규)**: 모델 rolling은 퇴출마다 8토큰 어긋난 격자에서 재양자화하며
   이는 멱등이 아니다. 이를 피하면 INT4 +1.29, INT2 +2.81(10프롬프트).
   fused 백엔드는 세그먼트 고정 격자라 구조적으로 이 오차가 없다(관문 4의 +1.5).
8. **H5-a 앵커 관리**: INT2에서 정책 +11.20(t=7.27), 계보 +2.81. 단 오라클 선택이
   {0,1,2}로 퇴화 — 이득은 "무엇을 앵커로 두나"가 아니라 운영 방식에서 온다.

## 알아야 할 코드 사실 (세션 5에서 추가/변경된 것)
- `--tum_mode {taylor, exact_uniform, emp_*}`, `--tum_lambda`, `--tum_include_self`(legacy),
  `--tum_flatten_slots`(구조 제거 컨트롤), `--tum_mass_chunks`(보정 후 mass 로깅),
  `--kv_eps_stats`(ε 실측), `--attn_backend fused`, `--fused_bias_lambda`
- 기본값이 **self 제외**다. 세션 4 이전 수치를 재현하려면 `--tum_include_self` 필요.
- fused 백엔드는 `kv_cache["fused"]`에 패킹 세그먼트를 들고 있고, 프롬프트마다
  `causal_inference.py`가 이를 비운다(빼먹으면 누출 — 세션 5에서 겪음).
- Triton 커널의 스케일 로드는 반드시 마스크할 것(세그먼트가 292.5블록이라 경계에서
  넘어감; 언마스크면 0×inf = NaN).
- `KIVIK_RTNV` 양자화기(키 KIVI asym + 값 RTN sym)가 kv-quant에 추가됨.

## 미결 (우선순위 순)
1. **J3 관문 4 재설계**: "dequant 경로와 등가"라는 전제가 틀렸다(fused가 더 정확).
   새 기준은 "BF16 대비 품질"이나 "격자 계보를 통제한 대조군"이어야 한다.
2. **격자 계보 효과의 단독 논문화**: 재양자화 격자 정렬만으로 INT2 +2.81.
   싸고(구현이 이미 있음) 독립적인 발견.
3. H5-c 배포 가능 프록시(슬롯 나이/키 통계)와 H5-e 결합(보정+앵커).
4. I 확장: 상위 2설정 126프레임, LongLive v1.0 재현.
5. H2 exact_uniform 본런(급수 구현·정확도 검증은 완료, 품질 런만 남음).
6. J2 커널 +13.1%를 +5%로: BM/BN autotune, num_stages, self 페이즈 병합.
7. J4 5090/sm_120 컴파일 점검.

## 재사용 안내
도구 경로·플래그 전체 목록, 격자 정렬 코드 위치(파일:줄), TF/오라클 드라이버 사용법,
평가 환경(**VBench 미설치**, MUSIQ 단일 지표), 함정 기록은
`~/gpu/Self-Forcing/results/NOTES_session5.md` 하단 "다음 세션을 위한 재사용 안내" 참조.

## 주의
- 품질 지표는 **평균 MUSIQ 1차, 낙폭 병기**. 낙폭 단독은 chunk 0 파괴를 보상으로 읽는다.
- `results/gateF2/*`는 수정 전 코드(λ=4 상당) 산물이다. λ=1 legacy 정본은
  `audit/rerun/tum_paper_int2`.
- MUSIQ는 self mass 37~74% 구간에 둔감하다 — 분포 지표와 품질 지표는 다른 것을 본다.
