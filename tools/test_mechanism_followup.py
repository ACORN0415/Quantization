"""Pre-heldout checks for inference units, frame ranges, and non-destructive gates."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from analyze_mechanism_fr import infer
from mechanism_worker import strict_mean


def main():
    mean,lo,hi,p=infer([1.]*12)
    assert (mean,lo,hi)==(1.,1.,1.) and p==2/4096
    mean,lo,hi,p=infer([1.,-1.]*6)
    assert mean==0 and p==1 and lo<=0<=hi
    positive=infer(np.arange(12)/10)
    negative=infer(-np.arange(12)/10)
    assert positive[0]==-negative[0] and positive[3]==negative[3]
    assert abs(positive[1]+negative[2])<1e-12 and abs(positive[2]+negative[1])<1e-12
    for values in ([1.]*24,[1.]*11,[float('nan')]*12):
        try:infer(values)
        except ValueError:pass
        else:raise AssertionError('incorrect independent sample count accepted')
    scores=list(range(501));late=scores[249:501]
    boundary=[12*c-3+o for c in range(21,42) for o in range(-2,3)]
    inside=[12*c-3+o for c in range(21,42) for o in range(4,9)]
    assert len(late)==252 and sum(late)/252==374.5
    assert len(set(boundary))==105 and sum(scores[i] for i in boundary)/105==369
    assert sum(scores[i] for i in inside)/105==375
    assert strict_mean([.1,None,.2]) is None and abs(strict_mean([.1,.2])-.15)<1e-12
    print('PASS exact sign-flip, interval sign symmetry, n=12 enforcement, raw frame ranges and no finite-only mean')


if __name__=='__main__':main()
