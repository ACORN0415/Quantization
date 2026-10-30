# TT 연구 최종 개발 확인: K와 V 동시 압축

2026-10-30. 사용자 요청에 따라 K까지 압축한 결과를 확인한 뒤 이 TT 실험 방향을 종료한다. 추가 rank 탐색이나 후속 영상 실험을 자동 시작하지 않는다.

기존 V-only 결과를 본 뒤 설계한 개발 실험이며 확증 사전 등록은 아니다. K를 BF16으로 남긴 것은 이전 실험의 범위 제한이었으며 일반적 권장사항/추세의 근거가 아니다.

## 고정 비교

- p0/p4, seed101, 126 latent frames / 42 chunks / 501 RGB frames, 동일 Wan/Self-Forcing.
- 비트2/3/4 각각: 직접 KV 양자화(rank0 표기), KV TT→양자화 rank192, KV TT→양자화 rank299.
- 총18개 새 영상. 완료된 BF16 두 영상은 hash 검증 후 재사용.
- 모든 층의 완료된 과거 K와 V 모두 압축. 현재 청크 K/V는 같은 ChunkStore BF16 pending 유지.
- 직접 양자화는 K/V 각각 token축 group16 symmetric RTN 실제 packing. TT는 K/V 각각 동일 TT분해·인자 양자화. K 특화 최적 양자화기 비교가 아닌 동일 구현 확장이다.
- K는 기존 저장 경로의 post-RoPE(위치인코딩 적용 후) 값을 압축한다. pre-RoPE 압축과 구분한다. RoPE 위치/attention 읽기 규칙은 바꾸지 않는다.
- rank192는 앞선 메모리 절감 후보, rank299는 동일 저장 예산 비교용으로 모두 보고한다. 비트별 결과를 보고 rank를 변경하지 않는다.

## 측정과 해석

동일 비트의 직접 KV 양자화를 주 기준선으로 삼는다. 과거 V-only 기준선과의 차이는 적용 범위 변경으로 별도 해석한다.

후반 raw RGB MUSIQ, 전체/경계 MUSIQ, DINO 특징일관성, 내부/경계 flow/warp, BF16대비 latent NMSE, 과거 K/V 저장 오차, 생성/encode/restore시간, 실제 K/V/pending 바이트, 최대 GPU allocated/reserved를 보고한다. MP4 프리뷰와 latent를 모두 보존한다.

개발 선별: 두 프롬프트 각각 후반 MUSIQ 손실1점이내+총캐시감소면 후보표시. 이는 품질동등성 검정이 아니며 움직임/경계/일관성 손상을 무시하지 않는다. n=2 seed1, 동일3/7/15수준 symmetricRTN의 제한을 명시한다.

## 검증과 실행

CPU codec 실제 바이트/복원/두 입력 독립 저장 검사. GPU 전체모델 identity8청크 BF16비트동일 및 9개 bit/rank 조합8청크 finite/첫청크동일/인코딩30×7 확인 후 영상 생성. 기준선 model/checkpoint/prompt/source/latent와 새 source를 manifest에 고정한다. 소스변경/입력변경을 섞은 재개를 거부한다.

systemd 사용자 서비스에서 게이트→18영상→raw평가→한국어 최종보고서를 연속 실행한다. 수치실패/메모리오류는 기록 후 조사하고 결과를 선택적으로 버리지 않는다. 완료 후 이 방향은 종료한다.
