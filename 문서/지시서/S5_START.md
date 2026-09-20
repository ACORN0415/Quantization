# 세션 5 시작 상태 (감사 세션이 작성, 2026-09-17)

## 커밋 (이 두 해시가 세션 5의 기준)
- Self-Forcing `45836f41d4d2649c811078c8b4297e39dc7cb4eb` = 33593df + patches/{01,02,20} + 세션 1~4 도구·config·prompts 전부
- kv-quant-longhorizon `226c4058fb36d8314cf8ba9c243d002c67802ef3` = b4c0936 + patches/21
- 재현 절차·superseded 규칙: `~/gpu/patches/README.md`

## bit-exact 확인 (병합·수정 후)
- A1_int4 21f×3p == 세션 3 latent: **21/21** · `--tum_correct` INT2 21f×3p == 감사 §B8(논문 Δ): **21/21**
- 참고(감사): gateA bf16/int4/int2 630/630, 세션1 21f 21/21, attn mass 독립 재계산 18/18 (`~/gpu/audit/AUDIT_REPORT.md`)

## 감사 버그 4건
- A. TUM Δ=2·scale(보정 4배 과대) → **수정됨**(Δ=scale). 이제 `--tum_correct` = 논문 사양 = 회복 +10.4(t=3.2). 구 +28.25는 4배-스케일 결과
- B. F2 "Δ=0 관문" 공허(BF16이라 경로 미발동) → 관문 무효, **§B6 실측으로 대체됨**(경로 자체 무해, t=−1.53). 코드 수정 불요
- C. freeze의 SINK_TENSORS 프롬프트 간 누출 → **수정됨**(inference.py 프롬프트별 리셋)
- D. results/attn_mass/A1_*·A2_bf16 JSON의 구버전 슬롯 라벨 → 코드 무관, **그 JSON 라벨만 믿지 말 것**(mass 값은 유효)

## 재사용 도구 (경로 그대로 실행 가능)
`~/gpu/audit/compare_replication.py`(latent bit-exact 비교) · `~/gpu/audit/tum_zero_driver.py`(Δ=0 컨트롤) ·
`~/gpu/audit/tum_paper_delta_driver.py`(구 4배 재현은 ×0.25→×1로) · `~/gpu/audit/profile_recheck.py`(CUDA event) ·
`~/gpu/audit/attn_mass_indep/indep_full_softmax.py`(+qk_dump.py, mass 독립 검증)

## 알아야 할 코드 사실
- 양자화 q는 **int8 저장**(패킹 없음) → 비트 가드 2..8, 16-bit no-op 불가(no-op 검사는 audit의 IdentityQuantizer 방식)
- self(현재 chunk)는 항상 BF16으로 읽힘: 기록→attention→재양자화 순서 (causal_model.py)
- TUM 명시 경로 발동 조건: `TUM_ENABLED and quantizer is not None and quant_state is not None` (causal_model.py, `--tum_correct`로 켬) — FA2 대비 ~6배 느림
- UCSD 그룹 축: RTN sym per-(block16,head,ch) / KIVI key per-channel(토큰축 축약), value per-token — 논문 사양과 일치 확인됨
- 프롬프트: prompts3 md5 aaef428a…, prompts10 md5 b257926d… · 결과 규약: `results/<실험>/<태그>/{videos,stats.json,run.log}` + `results/latents_*/<태그>/prompt###/chunk_####.pt`

## 미결
- Quantization/tools/ 밑의 구버전 tum_correction.py 사본은 역사 기록으로 남김(정본은 Self-Forcing 커밋)
- E2-e 노이즈 sink의 norm 매칭이 레이어 단위(head별 아님) — 후속 실험 설계 시 참고
