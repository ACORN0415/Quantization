# 압축 KV 복원값 재사용: 실행 가능성 확인 계획

작성: 2026-10-31. 이전 단회·반복 분석 연구와 별도 진단이다.

## 현재 승인 범위

공식 packed 코덱의 실행 가능성, 정확성 검사, 비용 측정 설계를 확인한다.
새 선택 정책, fused attention 개발, 새 영상 품질 실험은 수행하지 않는다.
첫 비교의 주장은 해당 코덱의 복원 후 attention 경로에 한정한다.

## 실행 전에 고정한 단계 0

- 공식 QVG: `https://github.com/svg-project/Quant-VideoGen`
- 소스 revision: `0601468f2dbba6a17ac7086faec6d41527cad188`.
- 체크아웃: `~/sfq_data/qvg-reuse-reference`. 소스 파일을 수정하지 않는다.
- INT2, 1 PRQ stage, 256 centroids, group64, k-means 최대2회,
  symmetric residual, clipping 없음. 공식 Self-Forcing launcher의 코덱 설정이다.
- 현재 환경 PyTorch2.6.0+cu124/Triton3.2.0/FlashAttention2.7.4,
  RTX4090 SM89. 공식 프로젝트 의존성 torch2.8.0과 다르므로 이 환경의
  실제 실행 결과로만 호환성을 판단한다. 기존 환경을 업그레이드하지 않는다.
- 누락된 termcolor2.5.0, loguru0.7.3만 별도 target 디렉터리에 설치한다.
- 합성 BF16 K/V `[1,12,4680,128]` 한 쌍, seed20261031.
  실제 모델에서 나온 KV나 생성 품질 표본이라고 부르지 않는다.
- 패킹 실물 확인: INT2 residual uint8의 마지막 축32, scale·centroid·index
  포함 실제 storage bytes를 계산한다. 원본 KV는 코덱을 거친 후 버린다.
- unpack/scale/centroid를 적용하는 별도 PyTorch 복원 기준과 공식 Triton
  출력의 형상·dtype·원소 바이트를 비교한다. 원본 BF16과 일치하라는 검사가 아니다.
- 공식 ChunkedKVCache.read, 단일 복원 경로, 5회 읽기 중 복원값 유지 경로의
  출력 바이트를 비교하고 실제 decoder 호출 지점에서 횟수를 기록한다.
- 같은 FlashAttention 호출의 결과를 5개 query에 대해 직접 바이트 비교한다.
  이는 attention 입력·출력 smoke test이며 최종 생성 latent 검증은 아니다.
- 계측을 위한 추가 복원 호출을 실제 경로 횟수에 섞지 않는다. 기준 복원,
  비교 실행별 횟수를 각각 초기화해 보고한다.
- GPU 시작 전 외부 compute PID가 없어야 한다. 단계0 wall budget600초,
  allocator 한도6GiB, 프로세스 실행 시도 최대2회. timeout/OOM/불일치는
  실패 기록으로 남긴다. JIT·검증·실패 재실행을 예산에 포함한다.
- 성능 결론은 내리지 않는다. 단계0은 timed generation도 H2 검증도 아니다.

## 코드 검토에서 발견한 비교 조건

공식 `ChunkedKVCache.read`는 단일 K 또는 V span을 읽을 때
`uncompress_kv_cache(state, state)`의 첫 반환값을 사용한다.
해당 helper는 두 입력 각각을 복원하므로 같은 tensor를 두 번 복원한다.
이 현상은 실제 호출 수로 확인하며, 성능 본 비교에는 중복 복원만 없앤
단일 복원 경로를 반드시 포함한다. 해당 정리는 decoder·코드·scale을
바꾸지 않는다. 원본 경로 대비 이득과 중복 제거 이후 재사용 이득을 분리한다.

## 단계 0 결과 이후 확정할 측정 설계

자세한 설계·코덱 선택 이유·fused 비교 제한은 `kv_reuse_preflight/REPORT_KO.md`
와 `kv_reuse_preflight/MEASUREMENT_PLAN_KO.md`에 기록한다.
단계1 이후 수치 측정은 해당 계획의 실행 조건을 충족한 뒤 시작한다.
기존 연구의 26회·4시간 예산을 재사용하지 않는다.
