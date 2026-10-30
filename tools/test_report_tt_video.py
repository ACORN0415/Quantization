"""CPU-only tests for complete-window semantics and report bookkeeping."""
import math
import unittest
from copy import deepcopy
import report_tt_video as report


def quality_fixture():
    musiq=[float(i%73) for i in range(501)]
    mse=[i/50000 for i in range(501)]
    dino=[.5+i/2000 for i in range(500)]
    boundary=[9+12*i for i in range(41)]
    inside=[i for i in range(501) if i not in boundary]
    selections=dict(whole=list(range(501)),late=list(range(249,501)),boundary=boundary,
                    inside=inside,late_boundary=[i for i in boundary if i>=249],late_inside=[i for i in inside if i>=249])
    windows={}
    for name,ids in selections.items():
        m=sum(mse[i] for i in ids)/len(ids)
        ends=[dino[i-1] for i in ids if i>0]
        windows[name]=dict(frame_count=len(ids),musiq=sum(musiq[i] for i in ids)/len(ids),
            pixel_mse=m,pixel_psnr_db=-10*math.log10(m),pixel_psnr_infinite=False,dino_subject=sum(ends)/len(ends))
    pairs=[]
    for c in range(42):
        for a,b in ((12*c+2,12*c+3),(12*c+3,12*c+4)):
            pairs.append(dict(kind='inside',chunk=c,i=a,j=b,warp_error=.1,coverage=.8,flow_pixels=2.,raw_difference=.2))
        if c:
            pairs.append(dict(kind='boundary',chunk=c,i=12*c-4,j=12*c-3,warp_error=.1,coverage=.8,flow_pixels=2.,raw_difference=.2))
    temporal={}
    for name in ('whole_boundary','whole_inside','late_boundary','late_inside'):
        period,kind=name.split('_')
        chosen=[p for p in pairs if p['kind']==kind and (period=='whole' or p['chunk']>=21)]
        temporal[name]=dict(pairs=len(chosen),empty_mask_pairs=0,warp_error=.1,coverage=.8,flow_pixels=2.,raw_difference=.2)
    return dict(frames=501,per_frame_musiq=musiq,per_frame_pixel_mse=mse,per_transition_dino=dino,
        windows=windows,pairs=pairs,temporal_windows=temporal)

class ReportTests(unittest.TestCase):
    def test_windows_and_late_boundary(self):
        q=quality_fixture();report.audit_quality(q)
        self.assertEqual(q['windows']['late']['frame_count'],252)
        self.assertEqual(q['windows']['late_boundary']['frame_count'],21)
        bad=deepcopy(q);bad['windows']['late']['musiq']+=.001
        with self.assertRaises(RuntimeError):report.audit_quality(bad)
        bad=deepcopy(q);bad['pairs'][0]['i']+=1
        with self.assertRaises(RuntimeError):report.audit_quality(bad)

    def test_empty_mask_not_dropped(self):
        q=quality_fixture()
        pair=next(p for p in q['pairs'] if p['kind']=='boundary' and p['chunk']==21)
        pair['coverage']=0.;pair['warp_error']=None
        for name in ('whole_boundary','late_boundary'):
            n=q['temporal_windows'][name]['pairs']
            q['temporal_windows'][name]['empty_mask_pairs']=1
            q['temporal_windows'][name]['warp_error']=None
            q['temporal_windows'][name]['coverage']=.8*(n-1)/n
        report.audit_quality(q)
        q['temporal_windows']['late_boundary']['warp_error']=.1
        with self.assertRaises(RuntimeError):report.audit_quality(q)

    def test_no_nan_or_infinite_zero_psnr(self):
        q=quality_fixture();q['per_frame_pixel_mse']=[0.]*501
        for window in q['windows'].values():
            window.update(pixel_mse=0.,pixel_psnr_db=None,pixel_psnr_infinite=True)
        report.audit_quality(q)
        q['windows']['whole']['pixel_psnr_db']=float('inf')
        with self.assertRaises(RuntimeError):report.audit_quality(q)

if __name__=='__main__':unittest.main()
