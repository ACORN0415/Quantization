"""Synthetic contracts only; no model, real capture, or GPU access."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
import torch
import bit_baseline as b


def run(output):
    begin = time.monotonic()
    torch.set_num_threads(4)
    here = Path(__file__).resolve().parent
    checks = []

    def check(name, passed, **detail):
        checks.append({'name':name,'passed':bool(passed),**detail})
        if not passed:
            raise AssertionError(name)

    def rejects(name, f):
        try:
            f()
        except ValueError:
            check(name,True)
        else:
            check(name,False)

    def by(x):
        return b.tensor_bytes(x.detach())

    def reference(k, state):
        f,n,g,head = state.validate()
        z=(k.float()@b.hadamard()).bfloat16()[0].reshape(f,n,12,128)
        z=torch.cat((z,z.new_zeros(f,g*16-n,12,128)),1).reshape(f,g,16,12,128)
        mx=z.abs().amax(2,keepdim=True)
        s4=(mx/7).clamp_min(1e-8).half()
        s8=(mx/127).clamp_min(1e-8).half()
        c4=(z.float()/s4.float()).round().clamp(-8,7).to(torch.int8)
        c8=(z.float()/s8.float()).round().clamp(-128,127).to(torch.int8)
        q4=c4.bfloat16()*s4.bfloat16();q8=c8.bfloat16()*s8.bfloat16()
        # Direct headwise output avoids the mixed packet decoder under test.
        q4=q4.reshape(f,g*16,12,128)[:,:n].reshape(1,f*n,12,128)
        q8=q8.reshape(f,g*16,12,128)[:,:n].reshape(1,f*n,12,128)
        return (q4.float()@b.hadamard().t()).bfloat16(),(q8.float()@b.hadamard().t()).bfloat16()

    generator=torch.Generator().manual_seed(1811)
    k=torch.randn(1,3*17,12,128,generator=generator).bfloat16()
    saved=by(k)
    for head in range(12):
        state=b.prepare_state(head,frame_tokens=17)
        packet=b.encode(k,state); decoded=b.restore(packet,state)
        ref4,ref8=reference(k,state)
        others=[h for h in range(12) if h!=head]
        check('head%d_selected_INT8_exact'%head,by(decoded[:,:,head])==by(ref8[:,:,head]))
        check('head%d_other_eleven_INT4_exact'%head,by(decoded[:,:,others])==by(ref4[:,:,others]))
        check('head%d_only_selected_changed'%head,all(torch.equal(decoded[:,:,h],ref4[:,:,h]) for h in others) and not torch.equal(decoded[:,:,head],ref4[:,:,head]))
        state2=b.MixedHeadState.from_bytes(state.to_bytes())
        packet2=b.MixedHeadPacket.from_bytes(packet.to_bytes(state),state2)
        check('head%d_state_packet_roundtrip'%head,packet2.to_bytes(state2)==packet.to_bytes(state) and by(b.restore(packet2,state2))==by(decoded))
        check('head%d_compact_owned_storage'%head,packet.storage_bytes()==packet.nbytes() and state.storage_bytes()==state.nbytes()==25)
        tensors=(state.header,state.selected_head,packet.packed4,packet.codes8,packet.scale,packet.sentinel)
        check('head%d_all_owned_no_alias'%head,len({t.data_ptr() for t in tensors})==len(tensors) and all(t.data_ptr()!=k.data_ptr() for t in tensors))
    check('input_key_unchanged',saved==by(k))
    raw4=torch.arange(-8,8,dtype=torch.int8).repeat(3)
    check('INT4_all_signed_codes_roundtrip',torch.equal(b._unpack4(b._pack4(raw4),raw4.shape),raw4))
    # All 256 signed INT8 byte values survive serialization without unsigned reinterpretation loss.
    state=b.prepare_state(3,frame_tokens=17);packet=b.encode(k,state)
    packet.codes8[:256]=torch.arange(-128,128,dtype=torch.int16).to(torch.int8)
    loaded=b.MixedHeadPacket.from_bytes(packet.to_bytes(state),state)
    check('INT8_all_signed_codes_serialization',torch.equal(loaded.codes8,packet.codes8))

    # Persistent storage and grad ownership, including both shared state tensors.
    with torch.enable_grad():
        kg=k.clone().requires_grad_(); old=by(kg)
        st=b.prepare_state(7,frame_tokens=17);p=b.encode(kg,st);restored=b.restore(p,st)
        tensors=(st.header,st.selected_head,p.packed4,p.codes8,p.scale,p.sentinel,restored)
        check('no_autograd_graph',all(not t.requires_grad and t.grad_fn is None for t in tensors))
        check('grad_input_preserved',kg.requires_grad and kg.grad is None and by(kg)==old and torch.is_grad_enabled())
    payload_path=here/'BIT_SYNTHETIC_STATE.pt'
    if payload_path.exists():
        raise FileExistsError(payload_path)
    torch.save({'state':{'header':st.header,'selected_head':st.selected_head},
                'packet':{'packed4':p.packed4,'codes8':p.codes8,'scale':p.scale,'sentinel':p.sentinel}},payload_path)
    disk=torch.load(payload_path,map_location='cpu',weights_only=True)
    rs=b.MixedHeadState(**disk['state']);rp=b.MixedHeadPacket(**disk['packet'])
    check('full_state_disk_save_reload',rs.to_bytes()==st.to_bytes() and rp.to_bytes(rs)==p.to_bytes(st) and by(b.restore(rp,rs))==by(restored))
    check('disk_state_no_alias',rs.header.data_ptr()!=st.header.data_ptr() and rs.selected_head.data_ptr()!=st.selected_head.data_ptr() and rp.scale.data_ptr()!=p.scale.data_ptr())
    check('synthetic_pt_under_20MiB',payload_path.stat().st_size<20*1024**2,bytes=payload_path.stat().st_size)

    # Selection uses original-coordinate raw moment, with deterministic minimum-index ties.
    moment=torch.eye(128,dtype=torch.float64)[None].repeat(12,1,1)
    original=torch.zeros(2,12,128,dtype=torch.bfloat16);rtn=original.clone()
    rtn[:,3,:]=1;rtn[:,5,:]=1
    mbytes,kbytes,rbytes=by(moment),by(original),by(rtn)
    selected,scores=b.select_head(moment,original,rtn)
    check('selection_tie_lowest_index',selected==3 and scores[3]==scores[5] and scores[3]>0)
    moment2=moment.clone();moment2[5]*=2
    selected2,scores2=b.select_head(moment2,original[None],rtn[None])
    check('selection_full_raw_moment_weight',selected2==5 and scores2[5]==2*scores2[3])
    independent=torch.stack([torch.sum((original[:,h].double()-rtn[:,h].double()) @ moment2[h] * (original[:,h].double()-rtn[:,h].double())) for h in range(12)])
    check('selection_independent_quadratic_form',torch.equal(scores2,independent))
    check('selection_inputs_preserved',by(moment)==mbytes and by(original)==kbytes and by(rtn)==rbytes)
    with torch.enable_grad():
        mg=moment.clone().requires_grad_();og=original.clone().requires_grad_();rg=rtn.clone().requires_grad_()
        _,sg=b.select_head(mg,og,rg)
        check('selection_no_graph_owned_scores',not sg.requires_grad and sg.grad_fn is None and sg.untyped_storage().nbytes()==sg.numel()*8)
    rejects('negative_moment_rejected',lambda:b.select_head(-moment,original,rtn))
    rejects('nonfinite_moment_rejected',lambda:b.select_head(moment*float('nan'),original,rtn))

    # Zero/tiny/overflow scale handling is rejection, never implicit clipping/fallback.
    rejects('zero_scale_rejected',lambda:b.encode(torch.zeros_like(k),state))
    rejects('tiny_scale_rejected',lambda:b.encode((k.float()*1e-10).bfloat16(),state))
    rejects('overflow_scale_rejected',lambda:b.encode((k.float()*1e8).bfloat16(),state))
    rejects('nonfinite_input_rejected',lambda:b.encode(torch.full_like(k,float('inf')),state))
    invalid=b.MixedHeadPacket.from_bytes(p.to_bytes(st),st);invalid.scale.flatten()[0]=0
    rejects('zero_packet_scale_rejected',lambda:b.restore(invalid,st))
    invalid.scale.flatten()[0]=float('inf')
    rejects('nonfinite_packet_scale_rejected',lambda:b.restore(invalid,st))
    rejects('truncated_state_rejected',lambda:b.MixedHeadState.from_bytes(st.to_bytes()[:-1]))
    rejects('truncated_packet_rejected',lambda:b.MixedHeadPacket.from_bytes(p.to_bytes(st)[:-1],st))

    # One full production-shaped synthetic case, against the untouched original R_fp32.
    source=here.parent/'causal_obq/common.py'
    spec=importlib.util.spec_from_file_location('_bit_reference_causal_common',source)
    common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
    big=torch.randn(1,4680,12,128,generator=generator).bfloat16()
    big_saved=hashlib.sha256(by(big)).hexdigest();bigstate=b.prepare_state(9)
    bigpacket=b.encode(big,bigstate);bigout=b.restore(bigpacket,bigstate)
    oldpacket=common.encode_k(big,'R_fp32');oldout=common.restore(oldpacket)
    others=[h for h in range(12) if h!=9]
    check('production_original_Rfp32_eleven_heads_exact',by(bigout[:,:,others])==by(oldout[:,:,others]))
    check('production_scale_eleven_heads_exact',by(bigpacket.scale[:,:,:,others])==by(oldpacket['scale'][:,:,:,others]))
    check('production_input_preserved',hashlib.sha256(by(big)).hexdigest()==big_saved)
    old_code_bytes=oldpacket['packed'].numel();new_code_bytes=bigpacket.packed4.numel()+bigpacket.codes8.numel()
    code_increment=6*(new_code_bytes-old_code_bytes)
    payload_increment=6*(bigpacket.nbytes()-common.nbytes(oldpacket))+bigstate.nbytes()
    check('production_payload_no_double_store',bigpacket.packed4.numel()==4704*11*128//2 and bigpacket.codes8.numel()==4704*128 and new_code_bytes==3913728)
    check('production_payload_compact',bigpacket.nbytes()==bigpacket.storage_bytes()==4816897 and common.nbytes(oldpacket)==4515841)
    check('six_chunk_code_increment',code_increment==1806336,bytes=code_increment)
    check('shared_metadata_and_cap',bigstate.nbytes()==25 and payload_increment==1806361,bytes=payload_increment)
    check('CPU_only',not torch.cuda.is_initialized())
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    check('RSS_below_2GiB',rss<2*1024**3,bytes=rss)
    result={'status':'passed','synthetic_only':True,'real_capture_reads':0,'GPU_calls':0,'model_loads':0,
            'checks_total':len(checks),'checks':checks,'threads':torch.get_num_threads(),'wall_seconds':time.monotonic()-begin,'max_rss_bytes':rss,
            'bytes':{'K4_one_chunk':common.nbytes(oldpacket),'mixed_one_chunk':bigpacket.nbytes(),'packed4':bigpacket.packed4.numel(),'int8':bigpacket.codes8.numel(),
                     'scale':bigpacket.scale.numel()*2,'sentinel':1,'shared_header':24,'shared_head':1,'six_chunk_code_increment':code_increment,'six_chunk_total_increment':payload_increment,
                     'six_mixed_K_only_plus_state':6*bigpacket.nbytes()+bigstate.nbytes(),'common_hadamard_excluded':128*128*4},
            'synthetic_artifact':{'path':str(payload_path),'bytes':payload_path.stat().st_size,'sha256':hashlib.sha256(payload_path.read_bytes()).hexdigest()},
            'source_hashes':{str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in (here/'bit_baseline.py',Path(__file__),source)}}
    with Path(output).open('x') as f:
        json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'status':result['status'],'checks':len(checks),'wall_seconds':result['wall_seconds'],'rss':rss,'bytes':result['bytes']}))


if __name__ == '__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);args=ap.parse_args()
    require_parent=Path(args.output).resolve().parent==Path(__file__).resolve().parent
    if not require_parent:raise ValueError('output must stay in preparation folder')
    run(args.output)
