# Self-Forcing 패치 — 세션 7

`~/gpu/Self-Forcing` 은 원저자 저장소(`guandeh17/Self-Forcing`)의 clone 이라
그쪽으로 push 할 수 없다. 우리가 얹은 변경은 여기에 패치로 보관한다.

| 파일 | 내용 |
|---|---|
| `0001-*.patch` | 세션 7 코드: P4 `--kv_storage master`, 덤프 키에 prompt 추가, 진단 스크립트, 그림 도구 |
| `0002-*.patch` | 실행 산출물을 추적에서 제외하는 `.gitignore` |
| `0001-c1-*.patch` | 브랜치 `c1-exact-uniform`: 수치 안정형 `exact_uniform`, head별 attention mass |
| `0001-l1-*.patch` | 브랜치 `l1-attr-diag`: P3 경로 귀속 진단 (캡처 훅 + 대조 스크립트) |
| `session7_code.diff` | 위 전체의 `.py` 변경만 모은 단일 diff (읽기용) |

적용: `git -C <Self-Forcing 사본> am self_forcing_patches/0001-*.patch`
기준 커밋은 `e107a36` 이다.

**주의.** L1 실험을 낸 소스는 `e107a36` + diff `1abfaf01ab4b0f91` 이고, 그 diff 는
`Self-Forcing/results/session6/l1/SOURCE_1abfaf01ab4b0f91.patch` 에 따로 보존돼
있다(그 실행을 재현하려면 이쪽을 쓴다).

원자료(`results/`)는 저장소에 넣지 않았다 — 141 MB 이고 대부분 반복 실행 로그다.
로컬 `~/gpu/Self-Forcing/results/` 와 `/vfsdata/gpu_data/` 에 있다.
