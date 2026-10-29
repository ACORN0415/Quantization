"""CPU-only algebra/property gate for same-query deletion and new-V-only rounding.

This is a floating-point property check, not a formal proof or model experiment.
No research tensors, GPU, model weights, or network are accessed. The accompanying
Markdown gives the exact-real derivation and scope. Run with the research Python.
"""
import itertools
import json

import torch


def weighted_terms(probabilities, removed, new_start, delta, lam):
    """Rows are [heads, queries, tokens], delta [heads, new_tokens, channels]."""
    surviving_mass = probabilities[..., removed:].sum(-1)
    c = surviving_mass.reciprocal()
    a0 = probabilities[..., new_start:]
    a1 = a0*c[..., None]
    e0 = a0@delta
    e1 = a1@delta
    direct = .5*(e0.square().sum()+e1.square().sum()) + lam*(e1-e0).square().sum()
    w = .5*(1+c.square()) + lam*(c-1).square()
    equivalent = (w[..., None]*e0.square()).sum()
    return direct, equivalent, e0, e1, c


def check():
    torch.set_num_threads(1)
    generator = torch.Generator(device='cpu').manual_seed(20261029)
    largest_value_error = 0.
    largest_gradient_error = 0.
    cases = 0
    for _ in range(32):
        probabilities = torch.randn(3,7,11,dtype=torch.float64,generator=generator).softmax(-1)
        delta = torch.randn(3,4,5,dtype=torch.float64,generator=generator,requires_grad=True)
        for lam in (0.,.25,1.,4.):
            direct, equivalent, e0, e1, c = weighted_terms(probabilities,3,7,delta,lam)
            torch.testing.assert_close(e1,c[...,None]*e0,rtol=1e-12,atol=1e-12)
            torch.testing.assert_close(direct,equivalent,rtol=1e-12,atol=1e-12)
            gd = torch.autograd.grad(direct,delta,retain_graph=True)[0]
            gw = torch.autograd.grad(equivalent,delta)[0]
            torch.testing.assert_close(gd,gw,rtol=1e-12,atol=1e-12)
            largest_value_error = max(largest_value_error,abs(float(direct-equivalent)))
            largest_gradient_error = max(largest_gradient_error,float((gd-gw).abs().max()))
            cases += 1

    # All 2^4 neighboring-rounding combinations share the same objective ordering.
    probabilities = torch.randn(1,5,8,dtype=torch.float64,generator=generator).softmax(-1)
    values = torch.tensor([.18,.63,-.42,.71],dtype=torch.float64)
    scale = .25
    low = (values/scale).floor()
    candidate_pairs = []
    for bits in itertools.product((0.,1.),repeat=4):
        restored = (low+torch.tensor(bits,dtype=torch.float64))*scale
        delta = (restored-values).reshape(1,4,1)
        direct, equivalent, *_ = weighted_terms(probabilities,2,4,delta,1.)
        candidate_pairs.append((bits,float(direct),float(equivalent)))
    direct_best = min(candidate_pairs,key=lambda row:row[1])
    weighted_best = min(candidate_pairs,key=lambda row:row[2])
    assert direct_best[0] == weighted_best[0]
    for _, direct, equivalent in candidate_pairs:
        assert abs(direct-equivalent)<1e-12

    # The simplification fails if fixed old tokens also have reconstruction error.
    probabilities = torch.randn(1,4,8,dtype=torch.float64,generator=generator).softmax(-1)
    errors = torch.randn(1,8,2,dtype=torch.float64,generator=generator)
    before = probabilities@errors
    after_prob = probabilities.clone();after_prob[...,:2]=0
    after_prob /= after_prob.sum(-1,keepdim=True)
    after = after_prob@errors
    c = probabilities[...,2:].sum(-1).reciprocal()
    full_error_nonproportional = float((after-c[...,None]*before).norm())
    assert full_error_nonproportional>1e-8

    # With old errors fixed, the full-prefix goal is a shifted-target output goal.
    # d depends on removed old error, which a source-discarding codec cannot know.
    delta = torch.randn(1,4,2,dtype=torch.float64,generator=generator,requires_grad=True)
    old_error = errors.clone();old_error[...,4:,:]=0
    removed_error = probabilities[...,:2]@old_error[...,:2,:]
    d = -c[...,None]*removed_error
    full0 = probabilities@old_error + probabilities[...,4:]@delta
    full1 = after_prob@old_error + after_prob[...,4:]@delta
    torch.testing.assert_close(full1,c[...,None]*full0+d,rtol=1e-12,atol=1e-12)
    shifted_cases = 0
    largest_shifted_error = 0.
    for lam in (0.,.25,1.,4.):
        direct = .5*(full0.square().sum()+full1.square().sum()) + lam*(full1-full0).square().sum()
        w = .5*(1+c.square()) + lam*(c-1).square()
        beta = .5*c + lam*(c-1)
        shifted = (w[...,None]*(full0+(beta/w)[...,None]*d).square()).sum()
        constant = (((.5+lam)-beta.square()/w)[...,None]*d.square()).sum()
        equivalent = shifted+constant
        torch.testing.assert_close(direct,equivalent,rtol=1e-12,atol=1e-12)
        gd = torch.autograd.grad(direct,delta,retain_graph=True)[0]
        gw = torch.autograd.grad(equivalent,delta,retain_graph=True)[0]
        torch.testing.assert_close(gd,gw,rtol=1e-12,atol=1e-12)
        largest_shifted_error = max(largest_shifted_error,abs(float(direct-equivalent)))
        shifted_cases += 1

    return dict(passed=True,device='cpu',dtype='float64',seed=20261029,
        random_objective_gradient_cases=cases,neighbor_rounding_candidates=len(candidate_pairs),
        same_candidate_selected=True,max_absolute_objective_difference=largest_value_error,
        max_absolute_gradient_difference=largest_gradient_error,
        full_old_error_counterexample_nonproportional_norm=full_error_nonproportional,
        full_prefix_shifted_objective_cases=shifted_cases,max_absolute_shifted_objective_difference=largest_shifted_error,
        conclusion='new-V-only, same-query cache deletion transition objective equals query-weighted output error',
        limits=['floating property checks, not a formal proof',
                'pre-W_O headwise AV squared loss; post-W_O head mixing is outside the scalar-weight identity',
                'does not rule out changing queries, quantized keys, old-error side information, or different code objectives',
                'full old-V BF16 error adds offsets unavailable to a codec that discards original old V'])


if __name__=='__main__':
    print(json.dumps(check(),indent=2))
