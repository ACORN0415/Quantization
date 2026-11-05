"""Fail-closed CPU diagnostic driver preparation. Real data requires root LOCK.

Import does not load tensors, fit calibration, invoke legacy guards or start work.
Synthetic fixture tests use pure compute functions without activating DataAccess.
"""
import argparse,gc,hashlib,importlib.util,io,json,math,os,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent
BASE=HERE.parent
ROOT=HERE.parents[3]
CAP_EXTRA=1806361
LIMIT_SECONDS=2700
LIMIT_RSS=8*1024**3
LIMIT_DISK=2*1024**3
ARMS=('R_bf16','R_fp32','FULL','FIXED17','BIT8','RMU','RP3','RP3Q','FULLP1','RQ','FULLQ','FULLP1Q','FM3','FM8','PAIR17','PAIR29','PAIR43')
DENSE_EXTRA={'R_bf16':0,'R_fp32':0,'FULL':786432,'FIXED17':786444,'BIT8':1806361,'RMU':6144,'RP3':1694328,'RP3Q':1694328,'FULLP1':1800600,'RQ':0,'FULLQ':786432,'FULLP1Q':1800600,'FM3':3267048,'FM8':4967208,'PAIR17':1403904,'PAIR29':1403904,'PAIR43':1403904}
PRIMARY_BASELINES=tuple(a for a in ARMS if not a.startswith('PAIR') and a not in ('FM3','FM8'))
QIDS=torch.arange(128)*4680//128
MODULE_PATHS={'paired':BASE/'paired_response_preparation/core.py','attention':BASE/'paired_response_preparation/attention_adapter.py','psac':BASE/'paired_baseline_preparation/core.py','projected':BASE/'paired_baseline_preparation/projected_metric.py','common':BASE/'causal_obq/common.py','bit':HERE/'bit_baseline.py','mean':HERE/'mean_baseline.py'}
_MODULES=None


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()

def tsha(x):
    flat=x.detach().contiguous().reshape(-1)
    if flat.stride(0)!=1:
        flat=torch.empty(flat.numel(),dtype=flat.dtype,device=flat.device).copy_(flat)
    return hashlib.sha256(flat.view(torch.uint8).numpy().tobytes()).hexdigest()
def save_json(path,data):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def tensor_bytes(value):
    if isinstance(value,torch.Tensor):return value.numel()*value.element_size()
    if isinstance(value,dict):return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value,(tuple,list)):return sum(tensor_bytes(v) for v in value)
    return 0

def modules():
    global _MODULES
    if _MODULES is None:
        loaded={}
        for key,path in MODULE_PATHS.items():
            spec=importlib.util.spec_from_file_location('paired_stage0_'+key,path)
            mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod);loaded[key]=mod
        _MODULES=loaded
    return _MODULES

def setup_cpu():
    os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4'
    sys.dont_write_bytecode=True;torch.set_num_threads(4);torch.set_grad_enabled(False)
    if torch.cuda.is_initialized():raise RuntimeError('GPU initialization forbidden')

def checked_tensor(x,shape,dtype,name):
    if not isinstance(x,torch.Tensor) or x.device.type!='cpu' or x.shape!=tuple(shape) or x.dtype!=dtype or not torch.isfinite(x).all():
        raise ValueError('invalid tensor '+name)
    return x

def verify_lock():
    path=HERE/'LOCK.json'
    if not path.exists():raise PermissionError('No root LOCK: real input access forbidden')
    lock=json.loads(path.read_text())
    if not (lock.get('root_execution_authorized') is True and lock.get('actual_opus_review_complete') is True and lock.get('validity_policy_accepted') is True):
        raise PermissionError('Root/actual Opus/validity review incomplete')
    evidence=lock.get('review_evidence_sha256',{})
    if not evidence:raise PermissionError('Missing immutable review evidence')
    for review_file,review_hash in evidence.items():
        if sha(review_file)!=review_hash:raise PermissionError('Review evidence changed')
    if lock.get('dense_extra_bytes')!=DENSE_EXTRA or lock.get('primary_baselines')!=list(PRIMARY_BASELINES):raise PermissionError('Preregistered byte classes changed')
    if lock.get('pending') or lock.get('cap_extra_bytes')!=CAP_EXTRA or lock.get('arms')!=list(ARMS):
        raise PermissionError('Pending contract, cap or arm mismatch')
    if lock.get('total_seconds')!=LIMIT_SECONDS or lock.get('rss_bytes')!=LIMIT_RSS or lock.get('disk_bytes')!=LIMIT_DISK:
        raise PermissionError('Resource contract mismatch')
    declared=json.loads((HERE/'SOURCES_V5.json').read_text())['sources']
    if lock.get('sources')!=declared:raise PermissionError('LOCK source set incomplete or differs from draft')
    for p,h in declared.items():
        if sha(p)!=h:raise PermissionError('Source hash changed: '+p)
    if lock.get('input_bindings_sha256')!=sha(HERE/'INPUT_BINDINGS_DRAFT.json'):
        raise PermissionError('Input binding changed')
    binding=json.loads((HERE/'INPUT_BINDINGS_DRAFT.json').read_text())
    if binding['Q128_indices']!=QIDS.tolist() or [x['layer'] for x in binding['layers']]!=[18,24]:raise PermissionError('Query/layer binding mismatch')
    return lock,sha(path)


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    def __init__(self,start):self.start=start
    def guard(self):
        if time.monotonic()-self.start>LIMIT_SECONDS:raise TimeoutError('45 minute total cap')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>LIMIT_RSS:raise MemoryError('8 GiB process RSS cap')
        if sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())>LIMIT_DISK:raise BudgetExceeded('2 GiB new disk cap')
        if torch.cuda.is_initialized():raise RuntimeError('GPU forbidden')


class DataAccess:
    """Constructed only in an authorized child after the shared experiment clock."""
    def __init__(self,budget,bindings,run):
        _,digest=verify_lock()
        if run['lock_sha256']!=digest or run['supervisor_pid']!=os.getppid():raise PermissionError('Not the registered supervisor child')
        self.budget,self.allowed,self.records=budget,{},[]
        def visit(x):
            if isinstance(x,dict):
                if 'path' in x and 'sha256' in x:self.allowed[str(Path(x['path']).resolve())]=x
                for v in x.values():visit(v)
            elif isinstance(x,list):
                for v in x:visit(v)
        visit(bindings)
    def load(self,descriptor):
        self.budget.guard();p=Path(descriptor['path']).resolve()
        if str(p) not in self.allowed and not p.is_relative_to(HERE/'prepared'):raise PermissionError('Unbound artifact '+str(p))
        started=time.monotonic();actual=sha(p)
        if actual!=descriptor['sha256'] or ('bytes' in descriptor and p.stat().st_size!=descriptor['bytes']):raise ValueError('Artifact hash/size '+str(p))
        value=torch.load(p,map_location='cpu',weights_only=True)
        self.records.append({'path':str(p),'sha256':actual,'load_hash_seconds':time.monotonic()-started})
        self.budget.guard();return value
    def save_tensor(self,name,value):
        self.budget.guard();p=HERE/'prepared'/name;p.parent.mkdir(exist_ok=True)
        if p.exists():raise FileExistsError(str(p))
        torch.save(value,p);record={'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
        restored=self.load(record)
        if tensor_digest(value)!=tensor_digest(restored):raise ValueError('Saved state reload mismatch')
        return record,restored
    def verify_inputs_unchanged(self):
        for item in self.records:
            self.budget.guard()
            if sha(item['path'])!=item['sha256']:raise ValueError('Source changed after read')


def tensor_digest(value):
    if isinstance(value,bytes):return {'bytes':len(value),'sha256':hashlib.sha256(value).hexdigest()}
    if isinstance(value,torch.Tensor):return {'shape':list(value.shape),'dtype':str(value.dtype),'sha256':tsha(value)}
    if isinstance(value,dict):return {str(k):tensor_digest(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [tensor_digest(v) for v in value]
    return value


def packet_restore_checked(packet,kind):
    c=modules()['common'];shape=(3,98,16,12,128) if kind=='K' else (1,293,16,12,128)
    scaleshape=(shape[0],shape[1],1,12,128)
    if packet.get('kind')!=kind or tuple(packet['shape'])!=shape:raise ValueError('packet kind/shape')
    checked_tensor(packet['scale'],scaleshape,torch.float16,'packet scale')
    if not (packet['scale']>0).all() or packet['packed'].dtype!=torch.uint8:raise ValueError('packet representation')
    checked_tensor(packet['packed'],(math.prod(shape)//2,),torch.uint8,'packed code')
    if kind=='K':
        checked_tensor(packet['sentinel'],(1,),torch.uint8,'sentinel')
        if int(packet['sentinel'][0])!=255:raise ValueError('sentinel value')
    if c.nbytes(packet)!=(4515841 if kind=='K' else 4500480):raise ValueError('packet tensor bytes')
    out=c.restore(packet);checked_tensor(out,(1,4680,12,128),torch.bfloat16,'restored '+kind)
    return out


@torch.no_grad()
def fixed17_encode(k,rotation):
    c=modules()['common'];checked_tensor(k,(1,4680,12,128),torch.bfloat16,'fixed17 K')
    checked_tensor(rotation,(12,128,128),torch.float32,'fixed17 R')
    z=torch.einsum('...hd,hde->...he',k.float(),rotation.float()).bfloat16()[0].reshape(3,1560,12,128)
    z=torch.cat((z,z.new_zeros(3,8,12,128)),1).reshape(3,98,16,12,128)
    sn=(z.abs().amax(2,keepdim=True)/7).clamp_min(1e-8);scale=sn.half()
    if not torch.equal(sn,scale.bfloat16()) or not torch.isfinite(scale).all() or not (scale>0).all():raise ValueError('fixed17 scale')
    code=(z.float()/scale.float()).round().clamp(-8,7).to(torch.int8);packed=c.pack(code)
    if not torch.equal(c.unpack(packed,code.shape),code):raise ValueError('fixed17 pack roundtrip')
    restored=(code.bfloat16()*scale.bfloat16()).reshape(3,1568,12,128)[:,:1560].reshape_as(k)
    restored=torch.einsum('...hd,hde->...he',restored.float(),rotation.transpose(-1,-2).float()).bfloat16()
    return restored,{'packed':packed,'scale':scale,'sentinel':torch.tensor([255],dtype=torch.uint8)}


@torch.no_grad()
def qksa_attention(query,key,value,current_key,current_value,steps,rotation):
    m=modules();ps=m['psac'];bias=ps.qksa_bias(query,steps,rotation)
    kt=torch.cat((key,current_key)).permute(1,2,0).contiguous();vt=torch.cat((value,current_value)).permute(1,0,2).contiguous()
    out=torch.empty((len(query),query.shape[1],value.shape[-1]),dtype=torch.float32)
    for begin in range(0,len(query),32):
        logits=torch.bmm(query[begin:begin+32].permute(1,0,2),kt)/math.sqrt(query.shape[-1])
        logits=ps.qksa_apply(logits,bias[:,begin:begin+32])
        out[begin:begin+32]=torch.bmm(logits.softmax(-1),vt).permute(1,0,2)
    if not torch.isfinite(out).all():raise ValueError('nonfinite QKSA attention')
    return out


@torch.no_grad()
def noquant_connection(raw_keys,values,query,current_key,current_value,rank=32):
    m=modules();a=m['attention'];p=m['paired'];ps=m['psac'];heads,dim=raw_keys[0].shape[1:]
    raw=torch.cat(raw_keys).float();vv=torch.cat(values).float()
    base=a.attention_window(query,raw,vv,current_key,current_value)
    pair=p.PairedResponse.gaussian(heads,dim,rank,17)
    for i,(k,v) in enumerate(zip(raw_keys,values)):
        state,_=pair.summarize(str(i),k.permute(1,0,2),k.permute(1,0,2),v.permute(1,0,2));pair.append(state)
        basis=torch.eye(dim)[:,:min(3,dim)].repeat(heads,1,1).bfloat16()
        packet,info=ps.psac_encode(k.bfloat16(),k.bfloat16(),basis,torch.eye(dim,dtype=torch.float64).repeat(heads,1,1))
        if info['active_rows']!=0 or tsha(ps.psac_restore(k.bfloat16(),packet,basis))!=tsha(k.bfloat16()):raise ValueError('PSAC noquant linkage')
    out=pair.read(query.permute(1,0,2),base.paired_output,base.logz,min_denominator_ratio=.1)
    if tsha(out.output)!=tsha(base.paired_output) or out.fallback.any():raise ValueError('paired noquant linkage')
    no_bias=qksa_attention(query,raw,vv,current_key,current_value,torch.zeros_like(raw),torch.eye(dim).repeat(heads,1,1))
    if tsha(no_bias)!=tsha(base.original_output):raise ValueError('QKSA noquant linkage')
    return {'paired_exact':True,'PSAC_exact':True,'QKSA_exact':True,'scope':'codec bypass / exact-zero branch only, not nonzero reader arithmetic or BF16 rotation roundtrip'}


def prepare_layer(binding,access):
    m=modules();c=m['common'];tick=time.monotonic();cal=access.load(binding['p0_calibration']);wo=access.load(binding['Wo'])
    q=checked_tensor(cal['q'],(1,4680,12,128),torch.bfloat16,'p0 Q')
    k=checked_tensor(cal['k'],(1,32760,12,128),torch.bfloat16,'p0 K')
    if tsha(cal['wo'])!=tsha(wo['wo']) or tsha(cal['bias'])!=tsha(wo['bias']):raise ValueError('p0/evaluation W_o differ')
    qhash,khash=tsha(q),tsha(k);cost={};t=time.monotonic();qh=q[0].permute(1,0,2).double();raw=torch.bmm(qh.transpose(1,2),qh)/4680;cost['original_moment_seconds']=time.monotonic()-t
    t=time.monotonic();hd,hr=c.hessian(q,return_raw=True);cost['rotated_moment_seconds']=time.monotonic()-t
    t=time.monotonic();factor=c.factorize(hd);cost['FULL_factor_seconds']=time.monotonic()-t
    if tsha(hr)!=binding['offline_rotated_raw_H_sha256'] or tsha(hd)!=binding['offline_rotated_damped_H_sha256']:raise ValueError('p0 H anchor')
    fixed=access.load(binding['fixed_state_artifact'])
    if tsha(raw)!=binding['fixed_raw_C_sha256'] or tsha(fixed['R'])!=binding['fixed_matrix_sha256'] or tsha(factor)!=binding['fixed_U_sha256']:raise ValueError('fixed p0 state anchor')
    t=time.monotonic();eigen,u=torch.linalg.eigh(raw);cost['PSAC_eigh_seconds']=time.monotonic()-t
    psac={};t=time.monotonic()
    for r in (1,3,8):
        basis=m['psac'].prepare_basis(u[:,:,-r:].flip(-1).contiguous())
        lam=eigen[:,-r:].flip(-1).float().contiguous()
        state=m['projected'].prepare_metric(basis,lam);psac[str(r)]={'U':state.basis,'lambda':state.eigenvalues}
    cost['PSAC_basis_prepare_all_ranks_seconds']=time.monotonic()-t
    rec=[];t=time.monotonic()
    for chunk in range(6):
        access.budget.guard();packet=c.encode_k(k[:,chunk*4680:(chunk+1)*4680],'R_fp32');rec.append(packet_restore_checked(packet,'K'))
    cost['BIT_p0_RTN_reconstruction_seconds']=time.monotonic()-t
    t=time.monotonic();head,scores=m['bit'].select_head(raw,k[:,:28080],torch.cat(rec,1));head=int(head);bit_state=m['bit'].prepare_state(head)
    cost['BIT_head_selection_seconds']=time.monotonic()-t
    t=time.monotonic();mean_state=m['mean'].make_mean(k[:,:28080]);cost['RMU_p0_mean_seconds']=time.monotonic()-t
    state={'raw_original_C':raw,'raw_rotated_H':hr,'damped_H':hd,'FULL_U':factor,'FIXED_R':fixed['R'],'FIXED_choice':fixed['choice'],'psac':psac,'mean_state_bytes':mean_state.to_bytes(),'bit_state_bytes':bit_state.to_bytes(),'bit_head':head,'bit_scores':scores}
    if tsha(q)!=qhash or tsha(k)!=khash:raise ValueError('p0 inputs mutated')
    record,_=access.save_tensor(f'layer{binding["layer"]}.pt',state)
    access.verify_inputs_unchanged()
    return {'status':'completed','layer':binding['layer'],'state':record,'state_tensor_hashes':tensor_digest(state),'prepare_seconds':time.monotonic()-tick,'prepare_components':cost,'source_loads':access.records,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'selected_head':head,'bit_selection':{'scores':scores.tolist(),'raw_C_sha256':tsha(raw),'p0_K_range':[0,28080],'p0_source_sha256':binding['p0_calibration']['sha256'],'reconstruction':'R_fp32'},'raw_C_bytes':tensor_bytes(raw)}


def evaluate_layer(binding,arm,access,prepared):
    m=modules();c=m['common'];a=m['attention'];ps=m['psac'];pr=m['projected'];pairmod=m['paired'];bit=m['bit'];tick=time.monotonic()
    allstate=access.load(prepared['state']);wo=access.load(binding['Wo']);weight=wo['wo'].float();bias=wo['bias'].float()
    checked_tensor(weight,(1536,1536),torch.float32,'Wo');checked_tensor(bias,(1536,),torch.float32,'bias')
    rank=3 if arm in ('RP3','RP3Q','FM3') else 8 if arm=='FM8' else 1 if arm in ('FULLP1','FULLP1Q') else None
    shared={}
    if arm.startswith('FULL'):shared['FULL_U']=allstate['FULL_U']
    if arm=='FIXED17':shared.update(R=allstate['FIXED_R'],choice=allstate['FIXED_choice'])
    if rank is not None:
        shared['basis']=allstate['psac'][str(rank)]['U']
        if arm.startswith('FM'):shared['raw_M']=allstate['raw_original_C']
        else:shared['lambda']=allstate['psac'][str(rank)]['lambda']
    bitstate=bit.MixedHeadState.from_bytes(allstate['bit_state_bytes']) if arm=='BIT8' else None
    meanstate=m['mean'].MeanState.from_bytes(allstate['mean_state_bytes']) if arm=='RMU' else None
    del allstate;gc.collect()
    if arm.startswith('PAIR'):
        pair=pairmod.PairedResponse.gaussian(12,128,32,int(arm[4:]));shared['Omega']=pair.omega
        make_paired_manifest(pair,binding['layer'],int(arm[4:]))  # initial Omega check; no persisted state exists yet
    else:pair=None
    original_keys=[];original_values=[];values=[];decoded=[];k_packets=[];states=[];packet_links=[];write_diagnostics=[];write_seconds=0.;payload_base=0
    for row,write in zip(binding['packets'],binding['writes']):
        access.budget.guard();data=access.load(write)
        rawk=checked_tensor(data['k'],(1,4680,12,128),torch.bfloat16,'write K');rawv=checked_tensor(data['v'],rawk.shape,torch.bfloat16,'write V')
        if data['context']['chunk']!=row['chunk'] or data['context']['call']!=4:raise ValueError('write ordering')
        vp=access.load(row['V']);vv=packet_restore_checked(vp,'V')[0];payload_base+=c.nbytes(vp)
        base_name='O_offline' if arm.startswith('FULL') else 'R_bf16' if arm=='R_bf16' else 'R_fp32'
        kp=access.load(row['K'][base_name]['artifact']);base=packet_restore_checked(kp,'K')
        if tsha(kp['scale'])!=row['K'][base_name]['scale_sha256'] or tsha(kp['packed'])!=row['K'][base_name]['code_sha256']:raise ValueError('packet tensor anchor')
        start=time.monotonic()
        if arm=='FIXED17':base,kp=fixed17_encode(rawk,shared['R'])
        elif arm=='RMU':
            tm=time.monotonic();kp=m['mean'].encode(rawk,meanstate);encode_seconds=time.monotonic()-tm
            tm=time.monotonic();base=m['mean'].restore(kp,meanstate);restore_seconds=time.monotonic()-tm
            write_diagnostics.append({'encode_seconds':encode_seconds,'restore_seconds':restore_seconds})
        elif arm=='BIT8':
            kp=bit.encode(rawk,bitstate);blob=kp.to_bytes(bitstate)
            link={'layer':binding['layer'],'chunk':row['chunk'],'state_sha256':hashlib.sha256(bitstate.to_bytes()).hexdigest(),'selected_head':int(bitstate.selected_head[0]),'packet_sha256':hashlib.sha256(blob).hexdigest()}
            kp=checked_bit_packet(blob,bitstate,link,binding['layer'],row['chunk']);base=bit.restore(kp,bitstate);packet_links.append(link)
        if rank is not None:
            if arm.startswith('FM'):correction,info=ps.psac_encode(rawk[0],base[0],shared['basis'],shared['raw_M'])
            else:
                metric=pr.prepare_metric(shared['basis'],shared['lambda']);correction,info=pr.psac_encode_projected(rawk[0],base[0],metric)
                del metric
            encoded=correction.to_bytes();correction=ps.PsacPacket.from_bytes(encoded)
            base=ps.psac_restore(base[0],correction,shared['basis'])[None];states.append(encoded);write_diagnostics.append(info)
        if pair is not None:
            state,info=pair.summarize(str(row['chunk']),rawk[0].permute(1,0,2),base[0].permute(1,0,2),vv.permute(1,0,2));pair.append(state)
            states.append({'id':state.chunk_id,'dz':state.dz,'dn':state.dn,'offset':state.offset});write_diagnostics.append(info)
        write_seconds+=time.monotonic()-start
        original_keys.append(rawk[0]);original_values.append(rawv[0]);values.append(vv);decoded.append(base[0]);k_packets.append(kp)
        payload_base+=kp.nbytes() if arm=='BIT8' else c.nbytes(kp)
    shared_bytes=tensor_bytes(shared)+(bitstate.nbytes() if bitstate is not None else 0)+(meanstate.nbytes() if meanstate is not None else 0)
    correction_bytes=sum(len(x) for x in states) if rank is not None else sum(tensor_bytes(x) for x in states)
    extra=payload_base-54097926+shared_bytes+correction_bytes
    if extra<0 or extra>DENSE_EXTRA[arm]:raise ValueError('Actual persistent bytes exceed preregistered dense bound')
    if arm not in ('FM3','FM8') and extra>CAP_EXTRA:raise ValueError('Registered primary/candidate arm exceeds cap')
    if pair is not None and shared_bytes+correction_bytes!=pair.tensor_payload_bytes():raise ValueError('paired state bytes')
    new_packets=[kp.to_bytes(bitstate) for kp in k_packets] if arm=='BIT8' else k_packets if arm in ('FIXED17','RMU') else []
    paired_manifest=make_paired_manifest(pair,binding['layer'],int(arm[4:])) if pair is not None else None
    persisted={'paired_manifest':paired_manifest,'packet_links':packet_links,'mean_state_bytes':meanstate.to_bytes() if meanstate else None,'shared':shared,'corrections':states,'bit_state_bytes':bitstate.to_bytes() if bitstate else None,'new_K_packets':new_packets}
    state_record,reloaded=access.save_tensor(f'l{binding["layer"]}_{arm}_state.pt',persisted)
    shared=reloaded['shared']
    if pair is not None:
        pair=pairmod.PairedResponse(shared['Omega'])
        for ss in reloaded['corrections']:
            pair.append(pairmod.ChunkState(ss['id'],ss['dz'],ss['dn'],ss['offset']))
        shared['Omega']=pair.omega
        if make_paired_manifest(pair,binding['layer'],int(arm[4:]))!=reloaded['paired_manifest']:raise ValueError('Paired state-to-Omega binding changed on reload')
    if bitstate is not None:
        bitstate=bit.MixedHeadState.from_bytes(reloaded['bit_state_bytes'])
        for blob,link in zip(reloaded['new_K_packets'],reloaded['packet_links']):checked_bit_packet(blob,bitstate,link,binding['layer'],link['chunk'])
    if meanstate is not None:meanstate=m['mean'].MeanState.from_bytes(reloaded['mean_state_bytes'])
    del persisted,reloaded;gc.collect()
    rows=[];raw=torch.cat(original_keys);origv=torch.cat(original_values).float();v4=torch.cat(values).float();dk=torch.cat(decoded).float()
    for read_desc in binding['reads']:
        access.budget.guard();data=access.load(read_desc);call=data['context']['call']
        checked_tensor(data['q'],(1,4680,12,128),torch.bfloat16,'read Q');checked_tensor(data['native_av'],(1,128,12,128),torch.bfloat16,'sampled native AV')
        current=checked_tensor(data['k_current'],(1,4680,12,128),torch.bfloat16,'current K')[0].float()
        currentv=checked_tensor(data['v_current'],(1,4680,12,128),torch.bfloat16,'current V')[0].float();query=data['q'][0,QIDS].float()
        if data['past_chunk_ids']!=[1,2,3,4,5,6] or data['context']['chunk']!=7 or call not in (0,3):raise ValueError('read timeline')
        validity_tick=time.monotonic();noquant=noquant_connection(original_keys,values,query,current,currentv);validity_seconds=time.monotonic()-validity_tick
        reference=a.attention_window(query,raw.float(),v4,current,currentv).original_output
        fullref=a.attention_window(query,raw.float(),origv,current,currentv).original_output
        start=time.monotonic();diag={'fallback_queries':0};fallback_rate=0.;fallback_count=0;denominator=None;exact_zero_rate=0.
        if arm not in ('RQ','RP3Q','FULLQ','FULLP1Q'):
            base=a.attention_window(query,dk,v4,current,currentv);out=base.original_output
        if pair is not None:
            po=pair.read(query.permute(1,0,2),base.paired_output,base.logz,min_denominator_ratio=.1)
            out=a.paired_to_original(po.output);diag=po.diagnostics
            checked_tensor(po.fallback,(12,128),torch.bool,'fallback mask')
            fallback_count=int(po.fallback.sum());fallback_rate=fallback_count/1536
            exact_zero_rate=1.0 if bool(diag.get('exact_zero_bypass',False)) else 0.0
            denominator=denominator_diagnostics(po.log_denominator_ratio)
        elif arm in ('RQ','RP3Q','FULLQ','FULLP1Q'):
            scales=torch.cat([kp['scale'][:,:,0] for kp in k_packets],0)
            steps=ps.frame_token_steps(scales,1560,16)
            checked_tensor(steps,(28080,12,128),torch.float32,'QKSA unpadded historical steps')
            out=qksa_attention(query,dk,v4,current,currentv,steps,c.hadamard().repeat(12,1,1))
        read_seconds=time.monotonic()-start
        primary=a.post_wo_difference(out,reference,weight,bias);fullscore=a.post_wo_difference(out,fullref,weight,bias)
        floor=a.post_wo_difference(fullref,data['native_av'][0].float(),weight,bias)
        for score in (primary,fullscore,floor):score.pop('delta')
        endpoint=f'causal_layer{binding["layer"]}_c7_call{call}'
        if arm in ('R_bf16','R_fp32','FULL','FIXED17'):
            anchor=binding['anchors'][str(call)][arm]
            if not math.isclose(primary['sse'],anchor,rel_tol=1e-9,abs_tol=1e-9):raise ValueError('existing K-only SSE anchor mismatch')
        rows.append({'endpoint':endpoint,'arm':arm,'primary_Konly':primary,'full_BF16_reference':fullscore,'CPU_vs_native_floor':floor,'extra_persistent_bytes':extra,'cap_eligible':arm not in ('FM3','FM8'),'validity':'valid','dense_extra_bound_bytes':DENSE_EXTRA[arm],'fallback_rate':fallback_rate,'fallback_count':fallback_count,'fallback_pairs_total':1536,'exact_zero_bypass_pair_fraction':exact_zero_rate,'log_denominator_ratio_summary':denominator,'read_diagnostics':diag,'noquant':noquant,'connection_validity_seconds':validity_seconds,'read_seconds':read_seconds})
    access.verify_inputs_unchanged()
    return {'status':'completed','arm':arm,'layer':binding['layer'],'rows':rows,'state_artifact':state_record,'paired_manifest':paired_manifest,'packet_state_links':packet_links,'source_loads':access.records,'write_seconds_including_validation':write_seconds,'write_diagnostics':write_diagnostics,'legacy_K_encode_cost':'new encoding measured' if arm in ('BIT8','FIXED17','RMU') else 'not remeasured: reused artifact, never zero cost','shared_tensor_bytes':shared_bytes,'correction_packet_tensor_bytes':correction_bytes,'cache_KV_payload_bytes':payload_base,'extra_persistent_bytes':extra,'common_H128_bytes_separate':65536,'worker_wall_seconds':time.monotonic()-tick,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}


def make_paired_manifest(pair,layer,seed):
    expected=json.loads((HERE/'OMEGA_BINDINGS_V2.json').read_text())
    digest=tsha(pair.omega)
    if expected['torch_version']!=str(torch.__version__) or expected['seeds'][str(seed)]['sha256']!=digest:raise ValueError('Omega generation differs from preregistered manifest')
    return {'layer':layer,'seed':seed,'torch_version':str(torch.__version__),'dtype':str(pair.omega.dtype),'device':str(pair.omega.device),'omega_sha256':digest,'omega_tensor_bytes':tensor_bytes(pair.omega),'layer_independent_ownership':True,'chunks':[{'layer':layer,'seed':seed,'omega_sha256':digest,'chunk_id':state.chunk_id,'tensors':tensor_digest({'dz':state.dz,'dn':state.dn,'offset':state.offset})} for state in pair.states.values()]}


def checked_bit_packet(blob,state,link,layer,chunk):
    if link['layer']!=layer or link['chunk']!=chunk or link['selected_head']!=int(state.selected_head[0]) or link['state_sha256']!=hashlib.sha256(state.to_bytes()).hexdigest() or link['packet_sha256']!=hashlib.sha256(blob).hexdigest():raise ValueError('BIT8 packet/state/layer provenance mismatch')
    return modules()['bit'].MixedHeadPacket.from_bytes(blob,state)


def denominator_diagnostics(logratio):
    if logratio.device.type!='cpu' or logratio.ndim!=2:raise ValueError('log denominator shape')
    x=logratio.detach().double();finite=x[torch.isfinite(x)];total=x.numel()
    values=[float(v) for v in torch.quantile(finite,torch.tensor([0.,.01,.5,.99,1.],dtype=torch.float64))] if finite.numel() else [None]*5
    return dict(zip(('finite_min','finite_p1','finite_p50','finite_p99','finite_max'),values),total_pairs=total,finite_pairs=finite.numel(),nonfinite_pairs=int((~torch.isfinite(x)).sum()),nan_pairs=int(torch.isnan(x).sum()),positive_infinity_pairs=int(torch.isposinf(x).sum()),negative_infinity_pairs=int(torch.isneginf(x).sum()),above_log10_fraction_all_pairs=float((x>math.log(10)).sum())/total,scope='descriptive only; positive infinity included in upper-tail count')


def decide(rows):
    expected={f'causal_layer{l}_c7_call{c}' for l in (18,24) for c in (0,3)}
    if len(rows)!=len(expected)*len(ARMS) or {(r['endpoint'],r['arm']) for r in rows}!={(e,a) for e in expected for a in ARMS}:return {'status':'partial','decision':'inconclusive_missing_cells'}
    judgments=[]
    for endpoint in sorted(expected):
        rs={r['arm']:r for r in rows if r['endpoint']==endpoint}
        if any(r.get('validity')!='valid' or r.get('extra_persistent_bytes',-1)<0 or r.get('extra_persistent_bytes',-1)>DENSE_EXTRA[r['arm']] for r in rs.values()):return {'status':'invalid','decision':'invalid_cell_or_bound_violation'}
        if any(rs[a].get('cap_eligible') is not True for a in PRIMARY_BASELINES):return {'status':'invalid','decision':'registered_primary_baseline_reclassified'}
        eligible=[rs[a] for a in PRIMARY_BASELINES]
        if len(eligible)!=len(PRIMARY_BASELINES):return {'status':'invalid','decision':'registered_baseline_invalid_or_missing'}
        best=min(eligible,key=lambda r:r['primary_Konly']['sse']);b=best['primary_Konly']['sse']
        for seed in (17,29,43):
            row=rs['PAIR'+str(seed)];gain=1-row['primary_Konly']['sse']/b if b>0 else None
            if row.get('fallback_pairs_total')!=1536 or row.get('fallback_count') not in range(1537):return {'status':'invalid','decision':'fallback_unit_mismatch'}
            passed=b>0 and gain>=.1 and row['fallback_count']/1536<=.001 and row['cap_eligible']
            judgments.append({'endpoint':endpoint,'seed':seed,'best_baseline':best['arm'],'gain':gain,'passed':passed})
    return {'status':'completed','decision':'continue_local_feasibility' if all(x['passed'] for x in judgments) else 'stop_fixed_Gaussian_rank32','comparisons':judgments}


def child(kind,layer,arm=None):
    setup_cpu();lock,lh=verify_lock();run=json.loads((HERE/'RUN.json').read_text());budget=Budget(run['started_monotonic']);budget.guard()
    binding=next(x for x in json.loads((HERE/'INPUT_BINDINGS_DRAFT.json').read_text())['layers'] if x['layer']==layer)
    access=DataAccess(budget,binding,run)
    try:
        result=prepare_layer(binding,access) if kind=='prepare' else evaluate_layer(binding,arm,access,json.loads((HERE/f'PREPARE_{layer}.json').read_text()))
        budget.guard()
    except Exception as exc:
        save_json(HERE/f"FAILURE_{kind}_{layer}_{arm or 'state'}.json",{'status':'partial_budget' if isinstance(exc,(TimeoutError,MemoryError,BudgetExceeded)) else 'invalid','error':repr(exc),'traceback':traceback.format_exc()})
        raise
    save_json(HERE/(f'PREPARE_{layer}.json' if kind=='prepare' else f'RESULT_{layer}_{arm}.json'),result)


def supervisor():
    setup_cpu();started=time.monotonic();lock,lh=verify_lock()
    if (HERE/'RUN.json').exists():raise FileExistsError('No reruns of an existing run')
    run={'status':'running','started_monotonic':started,'started_utc':time.time(),'lock_sha256':lh,'supervisor_pid':os.getpid(),'stages':[]}
    save_json(HERE/'RUN.json',run);budget=Budget(started);child_process=None
    try:
        for layer in (18,24):
            for kind,arm in [('prepare',None)]+[('arm',x) for x in ARMS]:
                budget.guard();label=f'{kind}_{layer}_{arm or "state"}';log=(HERE/(label+'.log')).open('x')
                command=[sys.executable,str(Path(__file__)),'--worker',kind,'--layer',str(layer)]
                if arm:command+=['--arm',arm]
                child_process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','PYTHONDONTWRITEBYTECODE':'1'})
                while child_process.poll() is None:
                    budget.guard()
                    status=Path(f'/proc/{child_process.pid}/status')
                    if status.exists():
                        rss=next((int(x.split()[1])*1024 for x in status.read_text().splitlines() if x.startswith('VmRSS:')),0)
                        if rss>LIMIT_RSS:raise MemoryError('Child RSS cap')
                    time.sleep(.2)
                log.close();run['stages'].append({'label':label,'exit_code':child_process.returncode});save_json(HERE/'RUN.json',run)
                if child_process.returncode:
                    failed=HERE/f'FAILURE_{kind}_{layer}_{arm or "state"}.json'
                    if failed.exists() and json.loads(failed.read_text()).get('status')=='partial_budget':raise BudgetExceeded('Child budget: '+label)
                    raise RuntimeError('Stage failed: '+label)
                child_process=None
        verify_lock();budget.guard()
        rows=[row for layer in (18,24) for arm in ARMS for row in json.loads((HERE/f'RESULT_{layer}_{arm}.json').read_text())['rows']]
        result=decide(rows);result['rows']=rows;save_json(HERE/'RESULT.json',result);run.update(status='completed',decision=result['decision'])
    except Exception as exc:
        run.update(status='partial_budget' if isinstance(exc,(TimeoutError,MemoryError,BudgetExceeded)) else 'invalid',error=repr(exc))
        if child_process is not None and child_process.poll() is None:
            os.killpg(child_process.pid,signal.SIGTERM)
            try:child_process.wait(timeout=3)
            except subprocess.TimeoutExpired:os.killpg(child_process.pid,signal.SIGKILL);child_process.wait()
        raise
    finally:
        run['elapsed_seconds']=time.monotonic()-started;save_json(HERE/'RUN.json',run)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--supervisor',action='store_true');ap.add_argument('--worker',choices=('prepare','arm'));ap.add_argument('--layer',type=int,choices=(18,24));ap.add_argument('--arm',choices=ARMS);args=ap.parse_args()
    if args.supervisor:supervisor()
    elif args.worker:
        if args.layer is None or (args.worker=='arm' and args.arm is None):ap.error('worker arguments missing')
        child(args.worker,args.layer,args.arm)
    else:print(json.dumps({'status':'preparation_only','real_input_access':False,'root_LOCK_required':True,'arms':ARMS,'cap_extra_bytes':CAP_EXTRA}))
