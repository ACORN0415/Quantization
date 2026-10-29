"""Read-only CLI status for reconstruction review and experiment jobs."""
import argparse,json,os,subprocess,time
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,default=Path.home()/'sfq_data')
ap.add_argument('--watch',action='store_true');ap.add_argument('--interval',type=int,default=5)
ap.add_argument('--history',action='store_true',help='also show archived reconstruction logs')
a=ap.parse_args()
def show():
    if a.watch:print('\033[2J\033[H',end='')
    print(time.strftime('%Y-%m-%d %H:%M:%S'),'— Quantization progress')
    print('Local data:',a.data)
    change=a.data/'change_mixed/status.json'
    if change.exists() and not a.history:
        try:
            v=json.loads(change.read_text())
            print('\nCHANGE-BASED MIXED PRECISION',v.get('state'),v.get('detail'))
            print('Updated UTC:',v.get('updated_utc'))
            if v.get('state')=='running':
                try:os.kill(v['pid'],0);print('Process:',v['pid'],'alive')
                except ProcessLookupError:print('ATTENTION: worker stopped')
            print('Report: Quantization/src/experiments/change_mixed_development/REPORT_KO.md')
            log=a.data/'change_mixed.log'
            if log.exists():print('\n'.join(log.read_text(errors='replace').splitlines()[-8:]))
        except (OSError,json.JSONDecodeError):print('Change mixed status updating…')
        return
    search=a.data/'rounding_search_quality/status.json'
    if search.exists() and not a.history:
        try:
            v=json.loads(search.read_text())
            print('\nROUNDING SEARCH QUALITY',v.get('state'),v.get('detail'))
            print('Updated UTC:',v.get('updated_utc'))
            if v.get('state')=='running':
                try:os.kill(v['pid'],0);print('Process:',v['pid'],'alive')
                except ProcessLookupError:print('ATTENTION: worker stopped')
            print('Report: Quantization/src/experiments/rounding_search_quality_development/REPORT_KO.md')
            print('Easy explanation: Quantization/docs/RESEARCH_STORY_KO.md')
            log=a.data/'rounding_search_quality.log'
            if log.exists():print('\n'.join(log.read_text(errors='replace').splitlines()[-8:]))
        except (OSError,json.JSONDecodeError):print('Search quality status updating…')
        return
    postwo=a.data/'postwo_rounding/status.json'
    if postwo.exists() and not a.history:
        try:
            v=json.loads(postwo.read_text())
            print('\nPOST-WO ROUNDING DEVELOPMENT',v.get('state'),v.get('detail'))
            print('Updated UTC:',v.get('updated_utc'))
            print('Capture prompts:',len(list((a.data/'postwo_rounding').glob('p*/manifest.json'))),'/ 2')
            if v.get('state')=='running':
                try:os.kill(v['pid'],0);print('Process:',v['pid'],'alive')
                except ProcessLookupError:print('ATTENTION: worker stopped or stage handoff')
            print('Report: Quantization/src/experiments/postwo_rounding_development/REPORT_KO.md')
            log=a.data/'postwo_rounding.log'
            if log.exists():print('\n'.join(log.read_text(errors='replace').splitlines()[-8:]))
        except (OSError,json.JSONDecodeError):print('Post-Wo status updating…')
        return
    transition=a.data/'transition_rounding/status.json'
    if transition.exists() and not a.history:
        try:
            v=json.loads(transition.read_text())
            print('\nTRANSITION ROUNDING FEASIBILITY',v.get('state'),v.get('detail'))
            print('Updated UTC:',v.get('updated_utc'))
            print('GPU model experiment: not started; algebra/property gate only')
            print('Report: Quantization/src/experiments/transition_rounding_development/REPORT_KO.md')
            print('Property checks:',v.get('property_checks'))
        except (OSError,json.JSONDecodeError):print('Transition status updating…')
        return
    dense=a.data/'dense_motion/status.json'
    if dense.exists() and not a.history:
        try:
            v=json.loads(dense.read_text())
            print('\nLOCAL KV CORRESPONDENCE DEVELOPMENT',v.get('state'),v.get('detail'))
            print('Updated UTC:',v.get('updated_utc'))
            if v.get('state')=='running':
                try:os.kill(v['pid'],0);print('Process:',v['pid'],'alive')
                except ProcessLookupError:print('ATTENTION: no dense motion process alive')
            print('Report: Quantization/src/experiments/dense_motion_development/REPORT_KO.md')
            log=a.data/'dense_motion.log'
            if log.exists():print('\n'.join(log.read_text(errors='replace').splitlines()[-8:]))
        except (OSError,json.JSONDecodeError):print('Local correspondence status updating…')
        return
    motion=a.data/'motion_codec/status.json'
    if motion.exists() and not a.history:
        try:
            v=json.loads(motion.read_text())
            print('\nMOTION CODEC DEVELOPMENT',v.get('state'),v.get('detail'))
            print('BF16 capture prompts:',len(list((a.data/'motion_codec').glob('p*/manifest.json'))),'/ 2')
            print('Updated UTC:',v.get('updated_utc'))
            if v.get('state')=='running':
                try:os.kill(v['pid'],0);print('Process:',v['pid'],'alive')
                except ProcessLookupError:print('ATTENTION: no motion codec process alive')
            print('Report: Quantization/src/experiments/motion_codec_development/REPORT_KO.md')
            log=a.data/'motion_codec.log'
            if log.exists():print('\n'.join(log.read_text(errors='replace').splitlines()[-8:]))
        except (OSError,json.JSONDecodeError):print('Motion status updating…')
        return
    cross=a.data/'quantizer_policy_cross/status.json'
    if cross.exists() and not a.history:
        try:
            v=json.loads(cross.read_text())
            print('\nQUANTIZER × POLICY DEVELOPMENT',v.get('state'),v.get('detail'))
            base=a.data/'quantizer_policy_cross'
            print('Full-model gates:', 'available' if (base/'gates.json').exists() else 'pending')
            print('BF16 replay prompts:',len(list((base/'replay').glob('*.json'))),'/ 2; six diagnostic layers')
            print('Development generation:',len(list((base/'development').glob('*/sampling.json'))),'/ 8')
            print('Raw evaluation:',len(list((base/'development').glob('*/quality_resource.json'))),'/ 8')
            print('Fresh held-out generation: not started; separate registration required')
            print('Updated UTC:',v.get('updated_utc'))
            targets=[Path(__file__).resolve().parents[2]/'tools'/name for name in
                     ('quantizer_policy_worker.py','quantizer_policy_study.py')]
            active=[]
            for proc in Path('/proc').glob('[0-9]*/cmdline'):
                try:argv=proc.read_bytes().split(b'\0')
                except (OSError,PermissionError):continue
                for target in targets:
                    if str(target).encode() in argv:active.append((int(proc.parent.name),target.name));break
            print('Active processes:',active or 'none')
            if v.get('state')=='running' and not active:print('ATTENTION: no cross-study worker/coordinator found')
            gpu=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader'],capture_output=True,text=True,timeout=2)
            print('GPU:',gpu.stdout.strip())
            log=a.data/'quantizer_policy_cross.log'
            if log.exists():print('\n'.join(log.read_text(errors='replace').splitlines()[-8:]))
        except (OSError,json.JSONDecodeError,subprocess.TimeoutExpired):print('Cross-study status updating…')
        return
    mechanism=a.data/'mechanism/status.json'
    if mechanism.exists() and not a.history:
        try:
            v=json.loads(mechanism.read_text());print('\nMECHANISM',v.get('state'),v.get('detail'))
            decision=a.data/'mechanism/decision.json'
            if decision.exists():
                d=json.loads(decision.read_text())
                print('Replay:',len(list((a.data/'mechanism/replay').glob('*.json'))),'/ 60 — decision:', 'GO' if d.get('advance') else 'NO-GO')
            performance=[]
            for path in (a.data/'mechanism/performance').glob('*/sampling.json'):
                try:
                    meta=json.loads(path.read_text())
                    if str(meta.get('rep')) in ('0','1','2'):performance.append(meta)
                except json.JSONDecodeError:pass
            print('Performance repetitions complete:',len(performance),'/ 15 (identity/profile runs counted separately)')
            fr=a.data/'mechanism/fr'
            print('Held-out generation:',len(list(fr.glob('*/sampling.json'))),'/ 72; raw quality:',len(list(fr.glob('*/quality.json'))),'/ 72')
            if 'completed' in v:print('Completed:',v['completed'],'/',v.get('total','?'),'ETA hours:',v.get('eta_hours_rough','measuring'))
            elif v.get('state')=='failed':print('STOPPED: inspect current error; no background progress is assumed.')
            else:print('Phase status; performance mode omits per-chunk logging to avoid timing overhead.')
            print('Updated UTC:',v.get('updated_utc'))
            if v.get('state')=='running' and v.get('pid'):
                active_pid=v['pid']
                try:os.kill(active_pid,0);print('Process:',active_pid,'alive')
                except ProcessLookupError:
                    # Status belongs to the previous stage until a new worker
                    # finishes model loading. Find exact script argv tokens;
                    # a coordinator handoff is not a stopped experiment.
                    targets=[Path(__file__).resolve().parents[2]/'tools'/name for name in
                             ('mechanism_worker.py','mechanism_followup.py','mechanism_run.py','mechanism_pipeline.py')]
                    candidates=[]
                    for proc in Path('/proc').glob('[0-9]*/cmdline'):
                        try:argv=proc.read_bytes().split(b'\0')
                        except (OSError,PermissionError):continue
                        for rank,target in enumerate(targets):
                            if str(target).encode() in argv:candidates.append((rank,int(proc.parent.name),target.name));break
                    if candidates:
                        rank,active_pid,name=min(candidates)
                        print('Stage handoff: previous PID exited; active',name,'PID',active_pid)
                    else:print('ATTENTION: no mechanism worker/coordinator found');active_pid=None
                info=subprocess.run(['ps','-p',str(active_pid or 0),'-o','etime=,%cpu=,rss='],capture_output=True,text=True,timeout=2)
                print('Process elapsed / CPU % / RSS KiB:',info.stdout.strip())
            gpu=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader'],capture_output=True,text=True,timeout=2)
            print('GPU:',gpu.stdout.strip())
            log=a.data/'mechanism.log'
            if log.exists():print('\n'.join(log.read_text(errors='replace').splitlines()[-8:]))
        except (OSError,json.JSONDecodeError,subprocess.TimeoutExpired):print('Mechanism status updating…')
        return
    supplement=a.data/'policy_supplement/status.json'
    if supplement.exists() and not a.history:
        try:
            value=json.loads(supplement.read_text())
            print('\nSUPPLEMENT',value.get('state'),value.get('detail'))
            print('Videos:',value.get('completed',0),'/',value.get('total',260),'ETA hours:',value.get('eta_hours'))
            print('Updated UTC:',value.get('updated_utc'))
            if value.get('state')=='running' and value.get('pid'):
                try:os.kill(value['pid'],0);print('Process:',value['pid'],'alive')
                except ProcessLookupError:print('ATTENTION: evaluation process stopped')
            gpu=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader'],capture_output=True,text=True,timeout=2)
            print('GPU:',gpu.stdout.strip())
            log=a.data/'policy_supplement.log'
            if log.exists():print('\n'.join(log.read_text(errors='replace').splitlines()[-8:]))
            print('Integrated report: Quantization/src/experiments/policy_study_results/REPORT_KO.md')
        except (json.JSONDecodeError,OSError,subprocess.TimeoutExpired):print('Supplement status updating…')
        return
    current=a.data/'policy_study/status.json'
    if current.exists() and not a.history:
        request=a.data/'policy_study/pause_request.json'
        if request.exists():
            try:
                r=json.loads(request.read_text())
                print('PAUSE CONTROL:',r['state'],'— requested boundary:',r['target_cases'],'completed cases')
                if r['state']=='paused':print('Resume: bash ~/Quantization/tools/resume_policy_study.sh')
            except json.JSONDecodeError:pass
        try:
            for label,value in json.loads(current.read_text()).items():
                print('\n'+label.upper(),value.get('state','?'),value.get('detail',''))
                print('Updated UTC:',value.get('updated_utc','?'))
                if 'completed_cases' in value:
                    print('Cases:',value['completed_cases'],'/',value.get('total_cases',80),
                          'remaining hours (rough):',value.get('eta_hours_rough','not measured yet'))
                    if request.exists():
                        r=json.loads(request.read_text())
                        if r['state']=='armed':
                            print('Cases remaining until requested pause:',max(0,r['target_cases']-value['completed_cases']))
                if value.get('state')=='running' and value.get('pid'):
                    try:os.kill(value['pid'],0);print('Process:',value['pid'],'alive')
                    except ProcessLookupError:print('ATTENTION: recorded process has stopped; inspect failure log')
        except json.JSONDecodeError:print('Status updating…')
        try:
            gpu=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total',
                                '--format=csv,noheader'],capture_output=True,text=True,timeout=2)
            if gpu.returncode==0:print('GPU utilization / used / total:',gpu.stdout.strip())
        except (OSError,subprocess.TimeoutExpired):pass
        for name in ('WORK_STATUS.md','policy_study.log','policy_study_pipeline.log'):
            path=a.data/name
            if path.exists():
                print('\n'+name)
                print('\n'.join(path.read_text(errors='replace').replace('\r','\n').splitlines()[-10:]))
        return
    for name in ('WORK_STATUS.md','policy_study.log','policy_study_pipeline.log','policy_study_tests.log','feasibility.log','feasibility_tests.log','anchor_validity.log','latent_recovery.log','validity_followup_tests.log','review_tests_final.log','review_identity.log','p3_gate_final.log','p3_queue.log'):
        path=a.data/name
        if not path.exists():continue
        print('\n'+name+' ['+time.strftime('%H:%M:%S',time.localtime(path.stat().st_mtime))+']')
        lines=path.read_text(errors='replace').replace('\r','\n').splitlines()
        print('\n'.join(lines[-8:]))
    for pattern in ('verify/identity_*.log','p3_pilot/*.log'):
        for path in sorted(a.data.glob(pattern)):
            print('\n'+str(path.relative_to(a.data)))
            print('\n'.join(path.read_text(errors='replace').replace('\r','\n').splitlines()[-4:]))
    for relative in ('policy_study/status.json','feasibility/status.json','verify/status.json','latent_recovery/status.json','p3_pilot/queue.json'):
        path=a.data/relative
        if not path.exists():continue
        print('\n'+relative)
        try:
            for key,value in json.loads(path.read_text()).items():
                print(' ',key,':',value.get('state','?'),value.get('detail',''))
        except json.JSONDecodeError:print('  Updating…')
show()
while a.watch:
    time.sleep(max(1,min(a.interval,60)));show()
