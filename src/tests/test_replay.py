import unittest
import torch
from kvq import Quantizer
from kvq.lockstep import anchor_store
from kvq.replay import ReferenceReplay, StorageNoiseQuantizer, RecordingQuantizer, RawState, make_store


class Identity:
    def encode_pair(self,k,v): return RawState(k),RawState(v)


class ReplayTests(unittest.TestCase):
    def test_replay_all_writes_and_snapshot_isolation(self):
        torch.manual_seed(48)
        for policy in ('rolling','chunk'):
            replay=ReferenceReplay(make_store(policy,Quantizer(group=4),12))
            for chunk in range(5):
                work=replay.snapshot()
                for call in range(5):
                    x=torch.randn(1,4,1,4).bfloat16()
                    before=replay.store.global_end
                    work.attend(x,x,x,chunk*4,lambda q,k,v:k)
                    self.assertEqual(replay.store.global_end,before)
                    replay(x,x,x.shape[1]*chunk)
                self.assertEqual(replay.calls,(chunk+1)*5)
                if policy=='rolling':
                    a,b=work.restore(x),replay.store.restore(x)
                    self.assertTrue(all(torch.equal(i,j) for i,j in zip(a,b)))

    def test_chunk_replay_matches_anchor_after_eviction(self):
        torch.manual_seed(20)
        replay=ReferenceReplay(make_store('chunk',Quantizer(group=4),12))
        raw=[]
        for chunk in range(7):
            for call in range(5):
                x=torch.randn(1,4,1,4).bfloat16()
                replay(x,x,chunk*4)
                if call==0: raw.append(x);raw=raw[-3:]
                else: raw[-1]=x
            buf=torch.cat(raw,1)
            reference=dict(k=buf,v=buf.clone(),local_end_index=torch.tensor([buf.shape[1]]),
                           global_end_index=torch.tensor([(chunk+1)*4]))
            anchor=anchor_store(reference,Quantizer(group=4),12,4)
            z=torch.randn_like(x)
            a=replay.snapshot().attend(z,z,z,(chunk+1)*4,lambda q,k,v:k.clone())
            b=anchor.attend(z,z,z,(chunk+1)*4,lambda q,k,v:k.clone())
            self.assertTrue(torch.equal(a,b))

    def test_noise_rng_isolation_and_rounded_norm(self):
        torch.manual_seed(5)
        x=torch.randn(1,16,2,8).bfloat16()
        rng=torch.get_rng_state().clone()
        noise=StorageNoiseQuantizer(Quantizer(group=4))
        a=noise.encode_pair(x,x)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        b=StorageNoiseQuantizer(Quantizer(group=4)).encode_pair(x,x)
        self.assertTrue(torch.equal(a[0].restore(),b[0].restore()))
        for r in noise.records:
            self.assertLess(abs(r['actual_norm']/r['target_norm']-1),.05)
        identity=StorageNoiseQuantizer(Identity())
        self.assertTrue(torch.equal(identity.encode_pair(x,x)[0].restore(),x))

    def test_error_support_preserves_exact_coordinates_and_padding(self):
        torch.manual_seed(11)
        x=torch.randn(1,16,2,8).bfloat16();x[:,8:]=0
        q=Quantizer(group=4)
        target=q.encode_pair(x,x)[0].restore()
        noise=StorageNoiseQuantizer(q,support='quantization_error')
        state=noise.encode_pair(x,x)[0].restore()
        exact=target==x
        self.assertTrue(torch.equal(state[exact],x[exact]))
        self.assertTrue(torch.equal(state[:,8:],x[:,8:]))
        self.assertLess(abs(noise.records[0]['actual_norm']/noise.records[0]['target_norm']-1),.05)

    def test_recorded_fr_amplitudes_prevent_own_state_recalibration(self):
        torch.manual_seed(77)
        inputs=[torch.randn(1,16,2,8).bfloat16() for _ in range(42*5)]
        recorder=RecordingQuantizer(Quantizer(key_bits=2,value_bits=2))
        reference=make_store('rolling',recorder,112)
        for i,x in enumerate(inputs):reference.attend(x,x,x,(i//5)*16,lambda q,k,v: None)
        targets=[r['norm'] for r in recorder.records]
        noise=StorageNoiseQuantizer(Quantizer(key_bits=2,value_bits=2),support='quantization_error',target_norms=targets)
        control=make_store('rolling',noise,112)
        for i,x in enumerate(inputs):control.attend(x,x,x,(i//5)*16,lambda q,k,v: None)
        state=control.restore(inputs[-1])[0].float()
        self.assertTrue(torch.isfinite(state).all())
        self.assertLess(state.norm().item(),3*reference.restore(inputs[-1])[0].float().norm().item())
        self.assertEqual(len(noise.records),len(targets))
        for r,target in zip(noise.records,targets):
            self.assertAlmostEqual(r['target_norm'],target,places=5)
        zero=StorageNoiseQuantizer(Quantizer(),target_norms=[0.,0.])
        self.assertTrue(torch.equal(zero.encode_pair(inputs[0],inputs[0])[0].restore(),inputs[0]))

if __name__=='__main__': unittest.main()
