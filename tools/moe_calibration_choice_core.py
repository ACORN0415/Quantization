"""Pinned Olive primitives and independent small CPU reference checks."""
import ast
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT/'tools/vendor/olive_calibration'


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def tensor_hash(t):
    return hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


def same_bits(a,b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def load_primitive():
    spec = importlib.util.spec_from_file_location('olive_frozen_quant_utils',VENDOR/'quant_utils.py')
    utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils)
    tree = ast.parse((VENDOR/'gptq.py').read_text())
    node = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='gptq_quantize_weight')
    # Execute the original function AST and decorator without importing the Olive package.
    namespace = dict(torch=torch,WeightQuantizer=utils.WeightQuantizer)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(VENDOR/'gptq.py'),'exec'),namespace)
    return utils,namespace['gptq_quantize_weight']


def numpy_gptq(W,H,damp,group=4,block=4):
    """Independent sequential error compensation, FP64 on a tiny CPU fixture."""
    W,H = W.copy(),H.copy()
    dead = np.diag(H)==0
    H[dead,dead]=1
    W[:,dead]=0
    H[np.diag_indices(len(H))] += damp*np.diag(H).mean()
    R = np.linalg.cholesky(np.linalg.inv(H)).T
    Q = np.zeros_like(W)
    scales, zeros = [],[]
    for start in range(0,W.shape[1],block):
        stop=start+block
        local=W[:,start:stop].copy()
        errors=np.zeros_like(local)
        for i in range(block):
            col=start+i
            if col%group==0:
                lo=np.minimum(W[:,col:col+group].min(axis=1),0)
                hi=np.maximum(W[:,col:col+group].max(axis=1),0)
                bound=np.maximum(-lo,hi)
                bound[bound==0]=1
                scale=bound/7.5
                scales.append(scale.copy()); zeros.append(np.full(len(W),8,dtype=np.int32))
            qcode=np.clip(np.rint(local[:,i]/scale)+8,0,15)
            q=(qcode-8)*scale
            Q[:,col]=q
            err=(local[:,i]-q)/R[col,col]
            local[:,i:] -= err[:,None]*R[col,col:stop][None,:]
            errors[:,i]=err
        W[:,stop:] -= errors@R[start:stop,stop:]
    return Q,np.stack(scales,axis=1),np.stack(zeros,axis=1)


def cpu_validation():
    torch.set_num_threads(2)
    utils,primitive=load_primitive()
    rng=np.random.default_rng(20261101)
    results=[]
    for case in range(3):
        w=rng.normal(size=(5,8))
        x=rng.normal(size=(12 if case==0 else 3,8))
        if case==2:
            x[:,0]=0
        h=2*x.T@x/len(x)
        W,H=torch.tensor(w,dtype=torch.float64),torch.tensor(h,dtype=torch.float64)
        hashes=(tensor_hash(W),tensor_hash(H))
        first={}
        for order in ([.01,.05,.1,1.0],[1.0,.1,.05,.01]):
            for damping in order:
                work_w,work_h=W.clone(),H.clone()
                assert work_w.data_ptr()!=W.data_ptr() and work_h.data_ptr()!=H.data_ptr()
                q,s,z=primitive(work_w,work_h,utils.WeightQuantizer(4,True,4),4,damping,False)
                ref,rs,rz=numpy_gptq(w,h,damping)
                assert np.allclose(q.numpy(),ref,atol=1e-10,rtol=1e-10)
                assert np.allclose(s.numpy(),rs,atol=1e-10,rtol=1e-10)
                assert np.array_equal(z.numpy(),rz)
                if damping in first:
                    assert all(same_bits(a,b) for a,b in zip(first[damping],(q,s,z)))
                else:
                    first[damping]=(q.clone(),s.clone(),z.clone())
                    results.append(dict(case=case,damping=damping,max_abs=float(np.max(np.abs(q.numpy()-ref)))))
                assert hashes==(tensor_hash(W),tensor_hash(H))
        quant=utils.WeightQuantizer(4,True,4)
        for q,s,z in first.values():
            code=quant.quantize(q,s,z)
            packed=utils.pack_to_uint8(code,4)
            unpacked=utils.unpack_from_uint8(packed,4,q.shape)
            assert torch.equal(code,unpacked)
            assert same_bits(q,quant.dequantize(unpacked,s,z))
            # Independent little-endian nibble extraction checks every code.
            raw=packed.numpy()
            independent=np.empty(code.shape,dtype=np.int32)
            independent[:,0::2]=raw&15
            independent[:,1::2]=raw>>4
            assert np.array_equal(independent,code.numpy())
    assert not torch.cuda.is_initialized()
    return dict(status='PASS',synthetic_cases=3,unique_synthetic_gptq_results=12,
                actual_model_candidates=0,original_inputs_unchanged=True,
                candidate_order_independence=True,independent_packing=True,
                max_abs_difference=max(r['max_abs'] for r in results),rows=results)
