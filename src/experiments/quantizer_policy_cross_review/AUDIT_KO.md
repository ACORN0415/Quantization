# 양자화기 × 저장 정책 개발 비교 감사

검토일: 2026-10-29 (Asia/Seoul). 수치 측정은 20:23 KST에 8/8개를 마쳤다.
이 문서는 **같은 작성 주체에 속한 별도 에이전트의 코드 검토·재계산**이다.
외부 독립 검증, 새 GPU 재실행, 공식 KIVI 구현과의 수치 대조를 뜻하지 않는다.
동결한 측정 코드·개발 계획·원본 결과 수치는 수정하지 않았다.

## 결론

현재 첫 실행의 주지표 집계와 진행 판단을 무효로 만드는 오류는 발견하지 않았다.
두 개발 프롬프트의 interaction은 각각 **−3.810978465610077**, **+0.921385977003311**이었다.
부호가 다르고 두 번째 사례의 절댓값이 사전 기준 1.0보다 작으므로 **효과 기준 NO-GO**다.
자원 기준은 통과했다. 결과를 보고 기준이나 입력·양자화기를 바꿔 같은 개발 GO를 만들지 않는다.

다만 **재시작 시 평가 비용 기록을 덮어쓰는 결함**과 **자원 측정·중지 규칙의 구현 차이**가 있다.
이번 실행은 평가 재호출이 없고 모든 비용에 충분한 여유가 있어 현재 NO-GO 판단에 영향을 주지 않았다.
후속 실행 도구에서는 아래 항목을 수정해야 한다.

## 실제로 다시 확인한 것

실행 명령:

```bash
~/sfq_data/py tools/audit_quantizer_policy_cross.py
```

| 검사 | 확인 결과 |
|---|---|
| 실험 셀 | p0/p4 × seed101 × RTN/KIVI-style × L/C-repeat, 정확히 8개. 누락·중복 없음 |
| 평가 입력 | 8개 × 501 = 4,008개 finite MUSIQ. 후반 252프레임·경계·내부 평균을 per-frame 값에서 재계산 |
| interaction | 두 사례의 4셀 차이에서 별도 재계산. 저장된 값·effect_pass=False·advance=False와 일치 |
| 평가와 생성 연결 | quality의 latent SHA가 sampling의 latent SHA와 일치. sampling의 prompt/seed/method/arm/group/bits도 폴더·CSV와 일치 |
| 실제 저장량 | 코드·패딩·scale·offset·pending/residual 합계 확인. KIVI offset을 포함하며 실제 합계와 가상 패킹 추정치를 구분 |
| 전체 모델 게이트 | no-quant L/C-repeat 각각 42청크 일치 기록 확인. p0 RTN 두 팔의 보존된 실제 궤적 대응 플래그 확인 |
| BF16 재생 | 두 프롬프트 각각 5,040개 고유 행, teacher 42청크 일치. 6개 층·42청크·5호출·4팔과 history 토큰 마스크 확인 |
| 재생 요약 | 청크 7–41의 K/V 상대 L2를 원시 행에서 다시 평균해 8개 replay_summary 셀과 일치 확인 |
| 산출물 | 40개 예정 파일의 누락·중복, 바이트, SHA256 모두 확인 |
| 현재 소스 | 새로 source hash를 계산해 결과의 `b59e2f705ba548b4f2edd91db3ded296ede17010000bc68b711d9ffe7b6a3788`와 일치 확인 |

상기 GPU 게이트는 보존된 기록의 감사이며, 이번 검토에서 모델을 다시 실행한 것은 아니다.
RTN 대응 플래그와 별개로 40개 산출물의 현재 파일 해시는 실제로 다시 계산했다.

## 첫 실행의 평가 비용 기록은 유효한가

코디네이터 로그에서 각 셀의 `--task quality`가 **정확히 한 번** 호출됐다.
각 파일의 mtime은 `sampling.json < quality.json ≤ quality_resource.json < 완료 로그(+1.1초 허용)`였고,
최종 `RAW QUALITY ... 501/501` 이후에 평가 파일이 저장됐다. 평가 완료 캐시를 재호출한 흔적은 없다.

| 개발 셀 | 코디네이터 로그의 시작→완료 |
|---|---:|
| p0 RTN/L | 201초 |
| p0 KIVI-style/L | 205초 |
| p0 RTN/C-repeat | 193초 |
| p0 KIVI-style/C-repeat | 195초 |
| p4 RTN/C-repeat | 193초 |
| p4 KIVI-style/C-repeat | 195초 |
| p4 RTN/L | 201초 |
| p4 KIVI-style/L | 205초 |

CSV의 계측 합계 `setup + generation + quality 호출`은 176.70–189.90초다.
이 합계는 두 하위 프로세스의 인터프리터 시작·일부 검사 비용을 모두 포함한 종단 시간과 다르다.
로그 벽시계 193–205초도 사전 상한 600초보다 작으므로 이번 비용 통과 여부는 달라지지 않는다.
기록된 생성/평가 peak allocated는 6.612–6.821GiB, 프로세스 CPU high-water는 33.156–33.162GiB로 상한보다 작았다.

## 후속 도구에서 수정할 것

### 1. 평가 재사용 시 실제 자원 기록이 skip 비용으로 덮어써짐

[교차 worker](../../../tools/quantizer_policy_worker.py#L215)는 `common.quality(args)` 뒤에 항상
`quality_resource.json`을 새로 쓴다. 하지만 [공통 quality](../../../tools/mechanism_worker.py#L247)는
기존 `quality.json`이 있으면 바로 반환한다. 따라서 동일 코디네이터를 재시작하면 완료된 평가의
실제 시간·peak가 캐시 확인의 짧은 시간·낮은 peak로 바뀔 수 있다.

후속 버전에서는 기존 quality와 resource의 source/latent 연결을 확인해 **둘 다 그대로 보존**해야 한다.
quality만 있고 resource가 없으면 실제 평가 비용이 없다는 상태로 처리해야 하며, skip 비용으로 채우면 안 된다.
보강한 감사 도구는 `resource.seconds ≥ VAE decode + quality eval + MP4 encode`도 확인해 이 문제를 탐지한다.
이번 8개는 모두 첫 호출이었고 실제 평가 비용이 기록돼 있어 이 결함이 발생하지 않았다.

### 2. 모델 로딩·텍스트 처리의 allocated peak가 합계에서 빠짐

[공통 execute](../../../tools/mechanism_worker.py#L168)가 생성 시작에 CUDA peak를 reset한다.
[분석 코드](../../../tools/quantizer_policy_study.py#L47)는 그 생성 peak와 평가 peak의 최댓값만 사용한다.
따라서 이 값은 **생성/평가 단계의 peak**이며, 모델 로딩·텍스트 처리까지 포함한 전 과정 allocated peak는 아니다.
후속 버전에서는 setup peak를 reset 전에 별도로 남기고 결합하거나, 계획의 측정 범위를 정확히 제한해야 한다.
NVML telemetry는 유용한 보조 자료지만 100ms 표본이며 CUDA allocated의 정확한 최댓값과 동일하지 않다.

### 3. 자원 초과를 중지할지 NO-GO로 집계할지 규칙이 다름

[개발 계획](../QUANTIZER_POLICY_CROSS_DEVELOPMENT.md#L63)은 자원 초과를 가설의 NO-GO와 구분해 중지한다고 썼다.
실제 [코디네이터](../../../tools/quantizer_policy_study.py#L46)는 개별 생성/평가의 초과를 즉시 검사하지 않고,
8개 완료 후 `resource_pass=False`를 `advance=False`에 합친다. 재생 작업에는 별도 즉시 예산 검사가 있다.
후속 실행에서는 단계 종료 직후 비용을 검사하고 효과 NO-GO와 실행 중지 상태를 분리해야 한다.
이번에는 모든 케이스가 자원 기준을 통과했으므로 **현재 NO-GO의 원인은 효과 기준 미충족**이다.

## 해석 범위

이 비교의 KIVI-style value는 토큰별 head×channel 축을 함께 사용하고, scale 외에 min offset을 보존한다.
그룹 정의·residual 정책을 포함한 원본 KIVI 전체 방법을 재현한 실험이 아니다.
또 C-repeat는 실제로 과거를 반복 encode하므로, 이번 결과를 C-once의 실용적 순위로 대신할 수 없다.

두 개발 입력에서 interaction이 일관되지 않았다는 결과는 중요하지만,
저장 정책과 양자화기가 일반적으로 상호작용하지 않는다는 증명도 아니다.
현재 데이터는 미리 정한 후속 투자 기준을 통과하지 못했다는 결론으로 사용한다.

[고정 개발 계획](../QUANTIZER_POLICY_CROSS_DEVELOPMENT.md) · [원본 결과](../quantizer_policy_cross_development/REPORT_KO.md) · [판단·해시](../quantizer_policy_cross_development/decision.json)
