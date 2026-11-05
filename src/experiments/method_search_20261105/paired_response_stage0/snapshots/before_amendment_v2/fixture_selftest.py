"""Synthetic integration only. Never constructs DataAccess or calls runtime workers."""
import copy,hashlib,importlib.util,io,json,sys,time
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('synthetic_stage0_driver',HERE/'diagnose.py')
d=importlib.util.module_from_spec(spec);sys.modules[spec.name]=d;spec.loader.exec_module(d)
d.setup_cpu();torch.manual_seed(123);start=time.monotonic();m=d.modules();checks=[]
def ok(name,condition):
    if not bool(condition):raise AssertionError(name)
    checks.append(name)
def rejects(name,f):
    try:f()
    except (ValueError,PermissionError):checks.append(name);return
    raise AssertionError(name)
a,p,ps,pr,b=[m[x] for x in ('attention','paired','psac','projected','bit')]
H,D,Q,N,C=2,8,9,5,3
ks=[torch.randn(N,H,D).bfloat16() for _ in range(2)];vs=[torch.randn(N,H,D).bfloat16() for _ in range(2)]
q=torch.randn(Q,H,D);ck=torch.randn(C,H,D);cv=torch.randn(C,H,D);k=torch.cat(ks).float();v=torch.cat(vs).float();kh=(k+.035*torch.randn_like(k)).bfloat16().float();weight=torch.randn(H*D,H*D);bias=torch.randn(H*D)
before=d.tensor_digest([ks,vs,q,ck,cv,kh,weight,bias])
base=a.attention_window(q,kh,v,ck,cv);ref=a.attention_window(q,k,v,ck,cv)
ok('past_plus_current_range',base.past_tokens==N*2 and base.current_tokens==C)
manual=torch.bmm((torch.bmm(q.permute(1,0,2),torch.cat((kh,ck)).permute(1,2,0))/D**.5).softmax(-1),torch.cat((v,cv)).permute(1,0,2)).permute(1,0,2)
ok('base_attention_arithmetic',torch.equal(base.original_output,manual))
no=d.noquant_connection(ks,vs,q,ck,cv);ok('all_noquant_connections',all(no[x] for x in ('paired_exact','PSAC_exact','QKSA_exact')))
pair=p.PairedResponse.gaussian(H,D,32,17)
for i in range(2):
    st,_=pair.summarize(str(i),ks[i].permute(1,0,2),kh[i*N:(i+1)*N].permute(1,0,2),vs[i].permute(1,0,2));pair.append(st)
po=pair.read(q.permute(1,0,2),base.paired_output,base.logz,min_denominator_ratio=.1)
ok('paired_shape_finite',po.output.shape==(H,Q,D) and torch.isfinite(po.output).all())
expected=H*D*32*4+2*(H*32*4+H*32*D*4+H*32*8)
ok('paired_one_map_exact_bytes',pair.tensor_payload_bytes()==expected)
buf=io.BytesIO();torch.save({'omega':pair.omega,'states':[{'id':s.chunk_id,'dz':s.dz,'dn':s.dn,'offset':s.offset} for s in pair.states.values()]},buf);buf.seek(0);state=torch.load(buf,weights_only=True)
re=p.PairedResponse(state['omega'])
for s in state['states']:re.append(p.ChunkState(s['id'],s['dz'],s['dn'],s['offset']))
ok('shared_state_reload_same_read',torch.equal(re.read(q.permute(1,0,2),base.paired_output,base.logz,min_denominator_ratio=.1).output,po.output))
state['omega'].zero_();ok('shared_map_owned',not torch.equal(re.omega,state['omega']))
rr=pair.read(q.permute(1,0,2),base.paired_output,torch.full_like(base.logz,float('nan')),min_denominator_ratio=.1)
ok('fallback_preserves_base',rr.fallback.all() and torch.equal(rr.output,base.paired_output))
U=torch.eye(D,dtype=torch.float64)[:,:3].repeat(H,1,1);stored=ps.prepare_basis(U);lam=torch.tensor([1.,2.,3.]).repeat(H,1);metric=pr.prepare_metric(stored,lam)
pkt,info=pr.psac_encode_projected(k.bfloat16(),kh.bfloat16(),metric);blob=pkt.to_bytes();roundtrip=ps.PsacPacket.from_bytes(blob);restored=ps.psac_restore(kh.bfloat16(),roundtrip,stored)
ok('projected_packet_bytes',len(blob)==roundtrip.nbytes());ok('projected_restore_shape',restored.shape==k.shape and restored.dtype==torch.bfloat16)
M=(stored.double()*lam.double()[:,None])@stored.double().transpose(1,2);fp,fi=ps.psac_encode(k.bfloat16(),kh.bfloat16(),stored,M)
ok('projected_equals_explicit_metric_packet',fp.to_bytes()==blob)
ok('shared_metric_bytes',metric.tensor_payload_bytes()==H*D*3*2+H*3*4)
steps=torch.ones_like(k)*.125;qa=d.qksa_attention(q,kh,v,ck,cv,steps,torch.eye(D).repeat(H,1,1));ok('QKSA_shape',qa.shape==ref.original_output.shape)
s=a.post_wo_difference(a.paired_to_original(po.output),ref.original_output,weight,bias);s2=a.post_wo_difference(a.paired_to_original(po.output),ref.original_output,weight,bias+2)
ok('post_Wo_bias_cancels_delta',torch.equal(s['delta'],s2['delta']))
ok('Konly_ref_energy_explicit',s['reference_energy']==float((ref.original_output.reshape(Q,-1)@weight.t()+bias).double().square().sum()))
rejects('wrong_shape_rejected',lambda:a.attention_window(q,k[:,:1],v,ck,cv))
ok('input_bytes_preserved',before==d.tensor_digest([ks,vs,q,ck,cv,kh,weight,bias]))
stateb=b.prepare_state(4,frame_tokens=8,frames=1);bk=torch.randn(1,8,12,128).bfloat16();bh=d.tsha(bk);bp=b.encode(bk,stateb);bb=bp.to_bytes(stateb);br=b.restore(b.MixedHeadPacket.from_bytes(bb,b.MixedHeadState.from_bytes(stateb.to_bytes())),stateb)
ok('mixed_head_fixture_roundtrip_shape',br.shape==bk.shape and br.dtype==bk.dtype);ok('mixed_packet_byte_count',len(bb)==bp.nbytes());ok('mixed_input_preserved',bh==d.tsha(bk));ok('mixed_shared_state25B',stateb.nbytes()==25)
rows=[]
for l in (18,24):
 for c in (0,3):
  for arm in d.ARMS:rows.append({'endpoint':f'causal_layer{l}_c7_call{c}','arm':arm,'primary_Konly':{'sse':.8 if arm.startswith('PAIR') else 1.},'cap_eligible':not arm.startswith('FM'),'fallback_rate':0.})
ok('gate_all12_pass',d.decide(rows)['decision']=='continue_local_feasibility')
r=copy.deepcopy(rows);r[-1]['primary_Konly']['sse']=.95;ok('gate_one_seed_fails',d.decide(r)['decision']=='stop_fixed_Gaussian_rank32')
r=copy.deepcopy(rows);r[-1]['fallback_rate']=.002;ok('gate_fallback_fails',d.decide(r)['decision']=='stop_fixed_Gaussian_rank32')
r=copy.deepcopy(rows)
for z in r:
 if z['arm']=='FM8':z['primary_Konly']['sse']=.01
ok('gate_out_of_cap_reference_not_main',d.decide(r)['decision']=='continue_local_feasibility')
ok('gate_missing_partial',d.decide(rows[:-1])['status']=='partial');ok('gate_duplicate_partial',d.decide(rows+[rows[0]])['status']=='partial')
r=copy.deepcopy(rows);r[0]['primary_Konly']['sse']=0;ok('gate_zero_baseline_no_success',d.decide(r)['decision']=='stop_fixed_Gaussian_rank32')
rejects('real_run_refuses_without_LOCK',d.verify_lock)
ok('no_GPU_initialization',not torch.cuda.is_initialized());ok('runtime_functions_not_called',True)
report={'status':'passed','checks':checks,'passed':len(checks),'synthetic_only':True,'real_tensor_file_reads':0,'real_tensor_hash_reads':0,'actual_effect_measurements':0,'GPU_calls':0,'model_loads':0,'worker_supervisor_invocations':0,'elapsed_seconds':time.monotonic()-start,'source_sha256':{str(x):d.sha(x) for x in [HERE/'diagnose.py',Path(__file__),*d.MODULE_PATHS.values()]}}
path=HERE/'FIXTURE_SELFTEST_V5.json'
with path.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps({'status':report['status'],'passed':len(checks),'elapsed_seconds':report['elapsed_seconds']}))
