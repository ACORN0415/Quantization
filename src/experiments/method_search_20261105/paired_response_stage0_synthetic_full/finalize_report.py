"""Finalize synthetic evidence only after unchanged driver and audit finish."""
import hashlib,json,time
from pathlib import Path
S=Path(__file__).resolve().parent;H=S.parent/'paired_response_stage0';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
status=json.loads((S/'FULL_FIXTURE_STATUS.json').read_text());audit=json.loads((S/'AUDIT_SYNTHETIC_FINAL.json').read_text());run=json.loads((S/'RUN.json').read_text());build=json.loads((S/'BUILD_RESULT.json').read_text());old=json.loads((S/'failure_v3/FULL_FIXTURE_STATUS.json').read_text())
assert status['status']=='completed' and audit['status']=='passed'
assert sha(S/'diagnose.py')==sha(H/'diagnose.py') and not (H/'LOCK.json').exists() and not (H/'RUN.json').exists()
peak=max(build['peak_RSS_bytes'],max(status['max_RSS_by_pid'].values()),max(old['max_RSS_by_pid'].values()));disk=sum(p.stat().st_size for p in S.rglob('*') if p.is_file())
report=f'''생산 형상 전체 합성 pipeline 검사 — 완료, 실제 방법 효과0
2026-11-05. 이 보고서는 가짜 소형 helper 검사와 구분되는 runtime 완주 증거다.

V4 driver/bit/mean은 원 stage0 파일과 동일 bytes로 sibling에 복사했다. 순수 helper도
source hash를 고정했다. 원본 .pt는 열지 않았다. 합성 binding의 입력70개는 모두 이
sibling inputs 안에 있고 sitecustomize audit hook은 sibling 밖의 .pt 접근을 거부했다.
원 stage0 LOCK/RUN은 생성하지 않았다. fixture LOCK는 synthetic_only/test_authority로
명시했으며 boolean은 변경하지 않은 guard를 실행하기 위한 가짜 계약이다. 실제 최종
Opus 검토/실자료 실험 승인으로 인용하면 안 된다.

생산 형상은 고정했다: 2층18/24, BF16 Q[1,4680,12,128], 6개 과거 K/V 청크,
현재 BF16 K/V, p0 K32760, Wo1536×1536, query128,17팔, read0/3.
모든 입력은 고정 seed의 절차적 난수다. fixed17 입력 상태는 합성 H128/label16이며
논문의 선택 성능을 재현한 것이 아니다. p0 moment/factor 및 패킷은 합성자료로 생성했다.
anchor는 별도 직접 FP32 attention 식과 직접 fixed 회전/RTN 식에서 얻었다.

처음 V3는 rank1 basis byte hash에서 실패했다. 최초 failed RUN/LOCK/log/prepared는
failure_v3/에 보존했다. 과학적 문턱을 완화하지 않고 generic hash helper만 수정한
V4를 동일 입력으로 재검사했다. 입력 재생성/복제는 하지 않았다.
총20분 마감은 첫 준비 시작부터 유지했다. 생성 준비 {build['elapsed_since_fixture_start']:.3f}초,
V4 supervisor {run['elapsed_seconds']:.3f}초, 전체 준비·실패조사·수정·재검사 완료까지
{status['elapsed_seconds']:.3f}초다. 감사 시점까지 {audit['elapsed_from_fixture_start']:.3f}초다.
CPU4, 각 프로세스 측정 최대 RSS는 전체 단계에서 {peak}B, 현 생성 파일 합 {disk}B다.
한 프로세스 누적 RSS를 서로 다른 팔의 독립 peak로 혼동하지 않는다. 기존 실패와
새 실행의 peak도 구분해 원 상태 JSON에 남겼다.

완주: prepare2 + arm worker34 =36개 순차 단계 모두 exit0.
17팔×2층×2read=68행, 각 팔 상태 저장·재로드 및 hash 비교, 원소형상/scale/noquant
연결, 공통 K-only 참조·full-BF16 참고 SSE, 기존 합성 anchor, bytes/cap/fallback,
Omega/chunk/state 연결과 최종 decide까지 도달했다.
별도 읽기 전용 감사 {audit['checks']}개 확인이 통과했다. 이 감사는 저장 JSON/해시와
판정 산술 확인이며 전체 attention을 독립적으로 다시 계산한 것은 아니다.

RESULT.json의 점수와 decide 분기는 절차적 합성 수치다. 성공 또는 중단 분기를
실제 연구 후보의 효과나 종료 판단으로 인용하지 않는다. 실제 캡처 효과 측정0,
GPU0, 모델 로드0. 실제 입력에서의 안정성·분산·시간·메모리·성능과 신규성은 미검증이다.
실제 최종 Opus 소스 검토와 root LOCK는 별도이며 현재 원 stage0 실행은 하지 않았다.
기존 full-H pilot stop_quality는 유지한다.
'''
(S/'REPORT_KO.txt').write_text(report)
evidence=[S/x for x in ['REPORT_KO.txt','RESULT.json','RUN.json','FULL_FIXTURE_STATUS.json','AUDIT_SYNTHETIC_FINAL.json','FIXTURE_REGISTRATION.json','FIXTURE_RETRY_V4.json','SYNTHETIC_EVIDENCE.json','build_inputs.py','sitecustomize.py','run_retry_v4.py','audit_finished.py','diagnose.py','failure_v3/FAILURE_prepare_18_state.json']]
source=json.loads((H/'SOURCES_V4.json').read_text());source['sources'].update({str(p):sha(p) for p in evidence});source['synthetic_full_runtime_verified']=True;source['actual_capture_reads']=0
(H/'SOURCES_V4.json').write_text(json.dumps(source,indent=2)+'\n')
files=[H/x for x in ['diagnose.py','SOURCES_V4.json','AMENDMENT_V4_KO.txt','AMENDMENT_V4_SOURCE.patch','hash_stride_regression_v4.py','HASH_STRIDE_REGRESSION_V4.json']]+evidence
ready={'status':'V4_frozen_full_synthetic_runtime_completed_pending_actual_review','created_utc':time.time(),'source_editing_stopped':True,'sha256':{str(p):sha(p) for p in files},'hash_regression_checks':24,'synthetic_pipeline_stages':36,'synthetic_result_cells':68,'synthetic_audit_checks':audit['checks'],'synthetic_total_seconds':status['elapsed_seconds'],'actual_capture_reads':0,'actual_method_effect_measurements':0,'GPU_calls':0,'actual_Opus_final_source_review_complete':False,'root_LOCK_exists':False,'original_stage0_RUN_exists':False,'active_source_manifest':'SOURCES_V4.json','cap_extra_bytes':1806361,'arms':17,'expected_cells':68,'historical_failures_preserved':['V2 early reloaded variable','V3 singleton-last byte hash'],'scope':'production-shape procedural synthetic complete runtime, not actual captured-input result','prior_pilot':'stop_quality unchanged'}
with (H/'DRIVER_READY_V4.json').open('x') as f:json.dump(ready,f,indent=2);f.write('\n')
manager={'status':'V4_full_synthetic_runtime_complete_pending_actual_Opus_and_root_LOCK','updated_utc':time.time(),'driver_sha256':sha(H/'diagnose.py'),'active_source_manifest_sha256':sha(H/'SOURCES_V4.json'),'ready_sha256':sha(H/'DRIVER_READY_V4.json'),'full_synthetic_result_sha256':sha(S/'RESULT.json'),'full_synthetic_audit_sha256':sha(S/'AUDIT_SYNTHETIC_FINAL.json'),'synthetic_cells':68,'synthetic_stages':36,'hash_regression_checks':24,'actual_capture_reads':0,'actual_method_effect_measurements':0,'GPU_calls':0,'actual_Opus_final_source_review_complete':False,'total_synthetic_seconds':status['elapsed_seconds'],'max_process_RSS_bytes_all_phases':peak,'generated_disk_bytes':disk,'prior_pilot':'stop_quality unchanged'}
for p in [H/'MANAGER_STATUS.json',S/'MANAGER_STATUS.json']:p.write_text(json.dumps(manager,indent=2)+'\n')
print(json.dumps(manager,indent=2))
