"""Small, deterministic CPU algebra check; no model/capture/performance evidence.

Fixed before execution: epsilon=.25, 3 contexts, degree-8 scalar Taylor
features, 3 queries, 2 chunks. Standard library only. Report to stdout.
"""
import json
import math


def feature(x, degree=8):
    return [x**j / math.sqrt(math.factorial(j)) for j in range(degree + 1)]


def dot(a, b):
    return math.fsum(x*y for x, y in zip(a, b))


def summary(k, v, kh, vh):
    pk, ph = [feature(x) for x in k], [feature(x) for x in kh]
    dz = [math.fsum(a[j]-b[j] for a, b in zip(pk, ph)) for j in range(9)]
    dn = [math.fsum(a[j]*x-b[j]*y for a, b, x, y in zip(pk, ph, v, vh)) for j in range(9)]
    return dz, dn


def main():
    eps = .25
    t = math.exp(eps)
    u, v = 0., -(t-1)/t
    contexts = []
    for kh in [(0., 0.), (1., 0.), (0., 1.)]:
        k = [kh[0]+eps, kh[1]]
        assert [float(round(x)) for x in k] == list(kh)
        a, b = (math.exp(x) for x in kh)
        target = a*t/(a*t+b)
        corrected = (a+u)/(a+b+u+v)
        equation_residual = b*u-a*t*v-a*b*(t-1)
        contexts.append(dict(khat=kh, target=target, fitted_error_only=corrected,
                             equation_residual=equation_residual,
                             output_abs_error=abs(target-corrected)))
    assert all(x['output_abs_error'] < 1e-14 for x in contexts[:2])
    assert contexts[2]['output_abs_error'] > .01

    k = [-.7, .2, .8, -.3]
    v0 = [1., -.5, .2, .9]
    kh = [round(x*2)/2 for x in k]
    vh = [round(x*2)/2 for x in v0]
    dz, dn = summary(k, v0, kh, vh)
    z1, n1 = summary(k[:2], v0[:2], kh[:2], vh[:2])
    z2, n2 = summary(k[2:], v0[2:], kh[2:], vh[2:])
    checks = []
    for q in [-1., -.3, .8]:
        pq = feature(q)
        direct_z = math.fsum(dot(pq, feature(x))-dot(pq, feature(y)) for x, y in zip(k, kh))
        direct_n = math.fsum(dot(pq, feature(x))*a-dot(pq, feature(y))*b for x, y, a, b in zip(k, kh, v0, vh))
        checks.extend([abs(direct_z-dot(pq, dz)), abs(direct_n-dot(pq, dn))])
    zero_z, zero_n = summary(k, v0, k, v0)
    assert all(x == 0. for x in zero_z+zero_n)
    append_error = max(abs(a-b-c) for a, b, c in zip(dz+dn, z1+n1, z2+n2))
    eviction_error = max(abs((a-b)-c) for a, b, c in zip(dz+dn, z1+n1, z2+n2))
    assert max(checks+[append_error, eviction_error]) < 1e-13
    print(json.dumps(dict(
        scope='synthetic scalar algebra only; not KVLinC empirical evaluation or model evidence',
        fixed_epsilon=eps, error_only_fit_from_contexts_1_and_2=dict(u=u,v=v),
        contexts=contexts, paired_feature_summary_max_error=max(checks),
        no_quantization_state_exact_zero=True, append_max_error=append_error,
        eviction_max_error=eviction_error,
        limits=['Finite Taylor features approximate exp, not exact softmax.',
                'No real captures, GPU, quality, novelty or speed claim.']), indent=2))


if __name__ == '__main__':
    main()
