# TT → 양자화: 저장 예산 축소 개발 실험

2026-10-30. 기존 결과를 본 뒤 설계한 탐색 실험이며 확증 사전 등록이 아니다.

질문: 직접 V 양자화보다 실제 캐시를 적게 사용하면서 프레임 품질을 유지하는 TT rank가 있는가?

- 기존 p0/p4, seed101, Wan/Self-Forcing, 126 latent / 501 RGB frames, 42 chunks.
- K와 현재 V는 BF16 유지. 과거 V만 chunk-once 저장. 기존 TT 영상 실험과 동일.
- factor bits 2/3/4 × TT rank 64/128/192 × 두 프롬프트 = 새 영상 18개.
- rank299 TT→양자화 및 직접 양자화/BF16는 완료된 기존 실험 결과를 해시 확인 후 재사용한다.
- FP32 TT 분해, 실제 factor bit packing, 기존 symmetric RTN(2비트는 3수준) 유지.
- 결과에 따라 rank/표본/지표를 바꾸거나 사례를 제외하지 않는다. 오류는 기록하고 동일 조건으로 재개한다.

주 요약: 각 프롬프트의 직접 양자화 대비 후반 전체 프레임 MUSIQ 차이와 실제 K+V cache bytes. 개발 선별 기준으로 두 프롬프트 모두 MUSIQ 손실 1점 이내이고 캐시가 감소하면 후보로 표시한다. 이 1점은 검증된 지각적 동등성 한계가 아니며 n=2에서 품질 동등성/비열등성을 주장하지 않는다.

모든 조건의 DINO, 내부/경계 RAFT flow와 warp, 경계 MUSIQ, latent NMSE, 생성 시간, encode/restore 시간, GPU peak allocated/reserved를 같이 보고한다. 움직임 변화와 영상 육안 확인 없이 MUSIQ 후보를 최종 성공으로 부르지 않는다.

K와 pending 공간이 남으므로 V 저장량 감소율과 전체 캐시 감소율을 따로 보고한다. 전체 peak에는 모델 및 분해/복원 임시 버퍼가 포함된다. 기존 타이밍은 다른 실행 세션의 단일 측정이므로 정밀 속도 우위 검정은 하지 않는다.

측정 전 게이트: 모든 rank/bit payload 예측 및 실제 바이트 일치, 유한한 복원; fixed rank299 adapter가 기존 budget rank299 codec과 같은 복원을 만드는지 확인. rank299 각 비트 전체 모델 8청크가 기존 생성과 비트 동일함을 추가 확인한다. 전체 모델에서 p0 identity 8청크 원본과 비트 동일, rank64/192 각 bit에서 8청크 finite/첫청크 동일/30×7 encode 확인. 게이트를 통과한 뒤 18개 본 개발 영상을 생성한다.

기존 동결 소스는 수정하지 않는다. 새 소스/모델/체크포인트/기준 latent/평가 파일 해시를 manifest에 기록한다. 완성 latent와 preview를 보존한다. 생성→raw quality→한국어 비교 보고서를 systemd 사용자 서비스에서 연속 실행한다.
