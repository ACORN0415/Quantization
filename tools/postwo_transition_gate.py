"""CPU synthetic algebra gate for post-W_O cache-deletion error objectives.

No research tensors, model weights, GPU, or network are accessed. Exact-real
scope: fixed queries/keys, deletion-only transition, mutable surviving new-V
error, per-query head renormalization. This is not a novelty or utility verdict.
"""
import itertools
import json

import torch


def metric(matrix, factors, lam):
    """W [out,D], factors [queries,D] with per-head repeated scalar factors."""
    m = matrix.T @ matrix
    c = torch.diag_embed(factors)
    ident = torch.eye(m.shape[0], dtype=m.dtype, device=m.device)
    h = .5*(m + c.transpose(-2,-1) @ m @ c)
    h = h + lam*((c-ident).transpose(-2,-1) @ m @ (c-ident))
    return h


def direct(error, matrix, factors, lam):
    before = error @ matrix.T
    after = (error*factors) @ matrix.T
    return .5*(before.square().sum()+after.square().sum()) + lam*(after-before).square().sum()


def quadratic(error, matrix, factors, lam):
    return torch.einsum('qd,qde,qe->', error, metric(matrix,factors,lam), error)


def check():
    torch.set_num_threads(1)
    rng = torch.Generator(device='cpu').manual_seed(20261030)
    largest_value_error = 0.
    largest_gradient_error = 0.
    cases = 0
    for _ in range(32):
        w = torch.randn(5,6, dtype=torch.float64, generator=rng)
        factors = (1+torch.rand(4,3,dtype=torch.float64,generator=rng)*2).repeat_interleave(2,-1)
        for lam in (0.,.25,1.,4.):
            e = torch.randn(4,6,dtype=torch.float64,generator=rng,requires_grad=True)
            a,b = direct(e,w,factors,lam),quadratic(e,w,factors,lam)
            ga = torch.autograd.grad(a,e,retain_graph=True)[0]
            gb = torch.autograd.grad(b,e)[0]
            torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)
            torch.testing.assert_close(ga,gb,rtol=1e-12,atol=1e-12)
            largest_value_error = max(largest_value_error, abs(float(a-b)))
            largest_gradient_error = max(largest_gradient_error,float((ga-gb).abs().max()))
            cases += 1

    # Constructed counterexample: same head norms but different cross-head signs.
    # H=[[1.875,2.5],[2.5,3.5]], head-local weighting gives diag(1.875,3.5).
    w = torch.tensor([[1.,1.]],dtype=torch.float64)
    factors = torch.tensor([[1.5,2.]],dtype=torch.float64)
    plus = torch.tensor([[1.,1.]],dtype=torch.float64)
    minus = torch.tensor([[1.,-1.]],dtype=torch.float64)
    plus_loss = float(direct(plus,w,factors,1.))
    minus_loss = float(direct(minus,w,factors,1.))
    local_weights = .5*(1+factors.square())+(factors-1).square()
    local_plus = float((local_weights*plus.square()).sum())
    local_minus = float((local_weights*minus.square()).sum())
    assert plus_loss == 10.375 and minus_loss == .375
    assert local_plus == local_minus == 5.375
    m = w.T@w
    h = metric(w,factors,1.)[0]
    assert torch.linalg.matrix_rank(m) == 1 and torch.linalg.matrix_rank(h) == 2

    # Equal renormalization factors: exactly a scalar weight on full post-W_O error.
    w_equal = torch.randn(5,6,dtype=torch.float64,generator=rng)
    e_equal = torch.randn(4,6,dtype=torch.float64,generator=rng)
    c_equal = (1+torch.rand(4,1,dtype=torch.float64,generator=rng)).expand(4,6)
    scalar = .5*(1+c_equal[:,0].square())+(c_equal[:,0]-1).square()
    equal_direct = direct(e_equal,w_equal,c_equal,1.)
    equal_expected = (scalar[:,None]*(e_equal@w_equal.T).square()).sum()
    torch.testing.assert_close(equal_direct,equal_expected,rtol=1e-12,atol=1e-12)

    # Block-diagonal M: weighted within-head projected quadratic metrics.
    w_block = torch.block_diag(torch.tensor([[1.,.4],[.2,2.]],dtype=torch.float64),
                               torch.tensor([[.6,-.3],[.1,1.2]],dtype=torch.float64))
    e_block = torch.randn(3,4,dtype=torch.float64,generator=rng)
    ch = 1+torch.rand(3,2,dtype=torch.float64,generator=rng)
    c_block = ch.repeat_interleave(2,-1)
    weights = .5*(1+ch.square())+(ch-1).square()
    expected = sum((weights[:,head,None]*(e_block[:,2*head:2*head+2]@
                    w_block[2*head:2*head+2,2*head:2*head+2].T).square()).sum() for head in range(2))
    torch.testing.assert_close(direct(e_block,w_block,c_block,1.),expected,rtol=1e-12,atol=1e-12)

    # Exhaustive tiny rounding example: full metric and direct objective select
    # identical codes; a head-local scalar surrogate may select different codes.
    values = torch.tensor([.12,.38,.17,.34],dtype=torch.float64)
    scale = .25
    low = (values/scale).floor()
    candidates = []
    w_round = torch.tensor([[1.,1.],[.3,-.2]],dtype=torch.float64)
    c_round = torch.tensor([[1.2,2.1],[1.7,1.1]],dtype=torch.float64)
    # Frozen toy attention response maps two new values to each of two queries.
    a = torch.tensor([[.4,.6],[.7,.3]],dtype=torch.float64)
    for bits in itertools.product((0.,1.),repeat=4):
        delta = ((low+torch.tensor(bits,dtype=torch.float64))*scale-values).reshape(2,2)
        e = a@delta
        loss = direct(e,w_round,c_round,1.)
        loss_h = quadratic(e,w_round,c_round,1.)
        torch.testing.assert_close(loss,loss_h,rtol=1e-12,atol=1e-12)
        headw = .5*(1+c_round.square())+(c_round-1).square()
        col_norm_sq = w_round.square().sum(0)
        scalar_loss = (headw*col_norm_sq[None,:]*e.square()).sum()
        candidates.append((bits,float(loss),float(loss_h),float(scalar_loss)))
    best_direct = min(candidates,key=lambda x:x[1])
    best_metric = min(candidates,key=lambda x:x[2])
    best_scalar = min(candidates,key=lambda x:x[3])
    assert best_direct[0] == best_metric[0]

    # Fixed old-error offset: exact affine/quadratic expansion, no original-V-free
    # encoder is thereby constructed. d is treated as known oracle information.
    fixed_cases=0;largest_offset_difference=0.
    for lam in (0.,.25,1.,4.):
        w = torch.randn(5,6,dtype=torch.float64,generator=rng)
        factors = (1+torch.rand(4,3,dtype=torch.float64,generator=rng)).repeat_interleave(2,-1)
        e = torch.randn(4,6,dtype=torch.float64,generator=rng,requires_grad=True)
        d = torch.randn(4,6,dtype=torch.float64,generator=rng)
        before=e@w.T;after=(e*factors+d)@w.T
        actual=.5*(before.square().sum()+after.square().sum())+lam*(after-before).square().sum()
        md=d@(w.T@w)
        g=(.5*factors+lam*(factors-1))*md
        expected=quadratic(e,w,factors,lam)+2*(e*g).sum()+(.5+lam)*(d@w.T).square().sum()
        torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
        ga=torch.autograd.grad(actual,e,retain_graph=True)[0]
        gb=torch.autograd.grad(expected,e)[0]
        torch.testing.assert_close(ga,gb,rtol=1e-12,atol=1e-12)
        largest_offset_difference=max(largest_offset_difference,abs(float(actual-expected)))
        fixed_cases+=1

    return dict(passed=True,device='cpu',dtype='float64',seed=20261030,
        objective_gradient_cases=cases,max_absolute_objective_difference=largest_value_error,
        max_absolute_gradient_difference=largest_gradient_error,
        head_sign_counterexample=dict(same_head_norms=True,full_plus=plus_loss,full_minus=minus_loss,
            scalar_plus=local_plus,scalar_minus=local_minus,
            base_metric_rank=int(torch.linalg.matrix_rank(m)),transition_metric_rank=int(torch.linalg.matrix_rank(h))),
        equal_head_factor_special_case_passed=True,block_diagonal_metric_special_case_passed=True,
        rounding_candidates=len(candidates),direct_and_full_metric_same_selection=True,
        direct_best=list(best_direct[0]),head_local_best=list(best_scalar[0]),
        head_local_selection_differs=best_direct[0]!=best_scalar[0],
        fixed_old_offset_cases=fixed_cases,max_absolute_offset_objective_difference=largest_offset_difference,
        conclusion='post-W_O transition objective is a full quadratic metric; head-local scalar weights are generally insufficient',
        limits=['CPU synthetic property checks, not a model experiment or formal verification',
                'quadratic-metric equivalence alone establishes neither novelty nor lack of utility',
                'no real W_O, queries, cache tensors, GPU, or video quality were tested',
                'fixed keys/queries and deletion-only new-V error; full old-error offset needs extra information',
                'changing head signs is a constructed example, not evidence of harmful real-model cancellation'])


if __name__=='__main__':
    print(json.dumps(check(),indent=2))
