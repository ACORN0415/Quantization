# 양자화기 × 저장 정책 개발 비교

**개발 데이터이며 p값·새 보류 표본 검증이 아니다.**

| prompt | RTN L | RTN C-repeat | KIVI-style L | KIVI-style C-repeat | interaction |
|---|---:|---:|---:|---:|---:|
| 0 | 66.2296 | 71.2199 | 75.2064 | 76.3858 | -3.8110 |
| 4 | 56.7037 | 61.8279 | 64.0236 | 70.0692 | 0.9214 |

interaction = (KIVI_C-repeat−RTN_C-repeat)−(KIVI_L−RTN_L).
고정 개발 진행 기준: **NO-GO**. 효과 기준 False, 비용 기준 True.
GO는 후속 계획 준비 허용이며 새 방법·일반성·논문 기여의 입증이 아니다.
KIVI-style은 현재 재구현의 고유 그룹 축·offset을 유지하며 원본 KIVI 전체 방법의 재현이 아니다.
같은 명목 비트폭이며 같은 메모리 예산이 아니다. 실제 상주량·offset·pending을 cases.csv에 기록했다.
이 결과를 보고 양자화기·표본·INT2로 교체해 같은 개발 기준을 다시 통과시키지 않는다.

[고정 개발 계획](../QUANTIZER_POLICY_CROSS_DEVELOPMENT.md) · [전체 사례·비용·바이트](cases.csv) · [판단·재생·해시](decision.json)
