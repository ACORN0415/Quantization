"""T-ARP Algorithm 8 column selector + least squares, not Algorithm 9 full CUR.
Independent implementation from https://arxiv.org/html/2606.26688v2.
Common indices; full-spectrum weights recovered through rFFT multiplicities.
"""
import torch

def select(v,rank,seed=9101,greedy=False):
    # v [heads,nonredundant frequencies,channels,rank], orthonormal columns.
    # Three-frame case: DC and one conjugate pair.
    if v.shape[1]!=2 or rank!=v.shape[-1]:raise ValueError('requires 3-frame basis')
    v=v.to(torch.complex128);h,f,n,r=v.shape
    weights=torch.tensor([1.,2.],device=v.device,dtype=torch.float64)
    chosen=torch.empty(h,0,device=v.device,dtype=torch.long)
    generator=torch.Generator(device=v.device).manual_seed(seed)
    residual=v.clone();min_pivot=float('inf');max_annihilation=0.
    for step in range(rank):
        score=(residual.abs().square().sum(-1)*weights[None,:,None]).sum(1)
        if step:score.scatter_(1,chosen,0.)
        probs=score/score.sum(-1,keepdim=True)
        if not bool(torch.isfinite(probs).all()):raise RuntimeError('invalid sampling distribution')
        j=probs.argmax(-1,keepdim=True) if greedy else torch.multinomial(probs,1,generator=generator)
        pivot=residual.gather(2,j[:,None,:,None].expand(-1,f,-1,r)).squeeze(2)
        min_pivot=min(min_pivot,float(pivot.abs().square().sum(-1).amin()))
        chosen=torch.cat([chosen,j],dim=1)
        # Recompute the exact selected-row nullspace projection via QR for stability.
        selected=v.gather(2,chosen[:,None,:,None].expand(-1,f,-1,r))
        q,upper=torch.linalg.qr(selected.mH,mode='reduced')
        if float(upper.diagonal(dim1=-2,dim2=-1).abs().amin())<1e-10:
            raise RuntimeError('frequency-wise rank growth failed; do not silently resample')
        residual=v-(v@q)@q.mH
        annihilated=residual.gather(2,chosen[:,None,:,None].expand(-1,f,-1,r))
        max_annihilation=max(max_annihilation,float(annihilated.abs().amax()))
    return chosen,dict(min_frequency_pivot_norm_sq=min_pivot,max_selected_residual=max_annihilation)

def encode(x,rank,seed=9101,greedy=False,kind='adaptive'):
    a=torch.fft.rfft(x,dim=1)
    u,s,vh=torch.linalg.svd(a,full_matrices=False)
    if kind=='svd':
        return dict(columns=torch.fft.irfft(u[...,:rank]*s[...,:rank].unsqueeze(-2),n=3,dim=1).half().contiguous(),coeff=torch.fft.irfft(vh[...,:rank,:],n=3,dim=1).half().contiguous(),fourier=True),{}
    if kind=='static':
        score=vh[...,:rank,:].abs().square().sum(-2)
        ids=(score*score.new_tensor([1,2])[None,:,None]).sum(1).argsort(descending=True,stable=True)[...,:rank]
        diagnostics={}
    else:ids,diagnostics=select(vh[...,:rank,:].mH,rank,seed,greedy)
    c=x.gather(-1,ids[:,None,None,:].expand(-1,3,x.shape[2],-1)).half().contiguous()
    cf=torch.fft.rfft(c.float(),dim=1)
    coeff=torch.fft.irfft(torch.linalg.pinv(cf,rtol=1e-5)@a,n=3,dim=1).half().contiguous()
    return dict(columns=c,coeff=coeff,indices=ids.int().contiguous(),fourier=True),diagnostics
