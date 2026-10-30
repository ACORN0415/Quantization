# QVG 재사용 단계1 실행 등록

2026-10-31. 상위 계획은 `../kv_reuse_preflight/MEASUREMENT_PLAN_KO.md`.
이 등록은 성능 결과를 보기 전에 구체화한 실행 조건이다.

- 공식 QVG revision `0601468f2dbba6a17ac7086faec6d41527cad188`, INT2,
  group64, 1-stage, centroids256, k-means2회,8생성청크 압축을 유지한다.
- 기존 `policy_study_prompts.txt`의0번, seed303,36잠재프레임,
  native4 denoising+clean refresh, native pre-RoPE/attention 사용.
- 공식 checkpoint의 generator_ema를 로드한다. 입력·소스는 manifest로 봉인한다.
- local_attn_size180을 사용하되 이번 생성은36프레임이라 퇴거하지 않는다.
- 캐시 전체를 CPU로 옮겨 메모리를 출력하는 `_print_memory_usage`는 **모든 팔에서**
  끈다. 이는 값의 변환이 아닌 진단용 전송이며 주 측정에 포함하지 않는다.
- 텍스트는 같은 official encoder로 한 번 구하고 고정한다. 모델 로딩·text·VAE·
  파일 저장은 생성 timing 바깥이다. VAE 대신 종료 계측 stub을 사용한다.
- 총 GPU 프로세스 상한20GiB, allocator cap18GiB로 context/외부 할당의 여유를 둔다.
  NVML0.5초 표본,83도 이상/다른 compute PID/상한 초과/모니터 실패는 무효·중단.
  NVML 표본으로 모든 순간 peak를 측정했다고 표현하지 않는다.
- R은 timing 안에서1GiB BF16 pool을 할당하고 완전한 K/V span 쌍 순서로 배정한다.
  현재 native span은 K/V 한 쌍230,031,360바이트이므로 처음4층만 유지한다.
  slot을 재활용하는 동적 정책은 구현하지 않는다. 부족하면 B와 같은 복원으로 우회한다.
- 보관값은 pre-RoPE 복원값이다. Python 객체 identity와 tensor version으로 변경을
  감지하고 write/state 교체/완전 퇴거 시 무효화한다. native partial quantized-span
  overwrite는 원래 코드에서 dangling 상태를 만들 수 있어 B/R에서 명시적으로 거부한다.
  실제36프레임 실행은 해당 unsupported 경로를 사용하지 않는다.
- 실제 native span `[1,12,37440,128]`에서 부분 읽기·새 쓰기·state 교체·in-place
  tensor 변경·완전/부분 퇴거·전체 덮어쓰기·공간 부족을 생성 전에 검사한다.
- U/B/R 검증에서 실제 self-attention Q/K/V를 전 호출 기록/직접 비교한다.
  U 입력은 바뀌지 않은 prefix를 재사용한 **무손실 zstd 원소 바이트 테이프**로 보존한다.
  B/R은 복원한 전체 바이트와 실제 입력 바이트를 비교한다. 해시 비교로 대체하지 않는다.
  attention 테이프 한도65GiB, 디스크 여유8GiB 미만이면 무효·중단.
- 검증 입력 테이프는 profiling/warmup/timing에서 완전히 끈다. 최종 latent 바이트는
  모든 실행 종료 뒤 U 결과와 직접 비교한다.
- 생성60forward×30층. U 실제 decoder2400, B1200, R1048 호출을 기대하며 실제
  entry에서 계수한다. read는 압축용60회를 포함3660. 예상값 출력으로 대체하지 않는다.
- native source의 압축 경계 sync는 유지한다. timed run에 별도 내부 sync를 추가하지 않는다.
- 프로파일은 B의 별도 실행. decode/read/attention의 GPU event 구간은 중첩될 수 있어
  합산하지 않는다. `(152/1200)×decode+layout interval / 전체 시간`은 균일 비용을
  가정한 낙관적 추정일 뿐 엄밀한 절약 상한이 아니다.
- 전체120분·최대12생성은 native-span gate 시작부터 적용한다. 시작 후 재설정하지 않는다.
  순서 U/B/R 검증3 → B profile1 → B/R warmup2 → BR/RB/BR timed6.
-3쌍 모두5%보다 크고 낙관적 추정도5%를 넘으면 H2의 별도 등록 검토로 진행한다.
 3쌍 모두5%미만이고 추정도5%미만이면 이 설정의 성능정책 연구를 종료한다.
 그 외는 판단 불가로 남긴다.5%부근 잡음을 유의한 성공으로 해석하지 않는다.
 한 조건의 성능 진단이며, fused 우위·저널 신규성·비용모델 예측력은 주장하지 않는다.
