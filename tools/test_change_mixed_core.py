"""CPU codec contracts; no captured data, model weights, or GPU use."""
import math
import unittest

import torch

from change_mixed_core import encode_uniform, encode_mixed, pack_bits, unpack_bits


class ChangeMixedTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(51030)
        self.x = torch.randn(1, 19, 2, 3).to(torch.bfloat16)
        self.mask = torch.arange(19)%2 == 0

    def test_pack_roundtrip_all_widths_and_lengths(self):
        for bits in (1,2,3,4):
            for count in (0,1,2,3,7,8,9,17,32,35):
                codes = torch.randint(0,1 << bits,(count,),dtype=torch.uint8)
                packed = pack_bits(codes,bits)
                self.assertEqual(packed.numel(),math.ceil(count*bits/8))
                self.assertEqual(packed.untyped_storage().nbytes(),packed.numel())
                self.assertTrue(torch.equal(unpack_bits(packed,bits,count),codes))
        # Explicit byte convention, including a 3-bit field crossing a byte.
        self.assertEqual(pack_bits(torch.tensor([1,2,3,4],dtype=torch.uint8),3).tolist(),[209,8])

    def test_uniform_matches_fixed_original_group_formula(self):
        for bits in (2,3,4):
            state=encode_uniform(self.x,bits,group=4)
            maximum=state.group_max[:,torch.arange(19)//4,0].float()
            qmax=(1 << (bits-1))-1
            scale=maximum/qmax
            expected=(self.x.float()/scale).round().clamp(-qmax,qmax)*scale
            self.assertTrue(torch.equal(state.restore(),expected))
            self.assertEqual(state.restore().dtype,torch.float32)
            self.assertEqual(tuple(state.restore().shape),tuple(self.x.shape))

    def test_mask_never_redefines_group_calibration(self):
        x=self.x.clone();x[:,3]=60
        a=encode_uniform(x,3,group=4)
        b=encode_mixed(x,self.mask,group=4)
        c=encode_mixed(x,~self.mask,group=4)
        self.assertTrue(torch.equal(a.group_max,b.group_max))
        self.assertTrue(torch.equal(a.group_max,c.group_max))
        self.assertEqual(tuple(a.group_max.shape),(1,5,1,2,3))
        # The large token remains in its original group despite precision split.
        self.assertTrue(bool((b.group_max[:,0]==60).all()))

    def test_mixed_restores_original_order_and_equal_bit_identity(self):
        for mask in (self.mask,~self.mask,torch.zeros(19,dtype=torch.bool),torch.ones(19,dtype=torch.bool)):
            state=encode_mixed(self.x,mask,group=4)
            lo=encode_uniform(self.x,2,group=4).restore()
            hi=encode_uniform(self.x,4,group=4).restore()
            expected=torch.where(mask[None,:,None,None],hi,lo)
            self.assertTrue(torch.equal(state.restore(),expected))
            same=encode_mixed(self.x,mask,low_bits=4,high_bits=4,group=4)
            self.assertTrue(torch.equal(same.restore(),hi))

    def test_actual_packed_bytes_and_shared_mask_storage(self):
        state=encode_mixed(self.x,self.mask,group=4)
        hi=int(self.mask.sum());lo=19-hi;features=2*3
        codes=math.ceil(lo*features*2/8)+math.ceil(hi*features*4/8)
        maximum=5*features*2;mask_bytes=math.ceil(19/8)
        self.assertEqual(state.bytes()['resident'],codes+maximum+mask_bytes)
        other=encode_mixed(self.x*2,self.mask,group=4,packed_mask=state.packed_mask)
        self.assertEqual(state.packed_mask.data_ptr(),other.packed_mask.data_ptr())
        self.assertEqual(state.bytes()['packed_estimate'],state.bytes()['resident'])
        uniform=encode_uniform(self.x,3,group=4)
        self.assertEqual(uniform.bytes()['resident'],math.ceil(19*features*3/8)+maximum)
        # Explicit parent-level sharing counts the shared mask only once.
        tensors=[s.group_max for s in (state,other)]+[s.low_stream for s in (state,other)]+[s.high_stream for s in (state,other)]+[state.packed_mask]
        expected=sum(t.untyped_storage().nbytes() for t in tensors)
        self.assertEqual(expected,state.bytes()['resident']+other.bytes()['resident']-mask_bytes)

    def test_zero_and_validation(self):
        zero=torch.zeros_like(self.x)
        self.assertTrue(torch.equal(encode_mixed(zero,self.mask).restore(),zero.float()))
        self.assertTrue(bool(torch.isfinite(encode_uniform(zero).restore()).all()))
        with self.assertRaises(ValueError):pack_bits(torch.tensor([8]),3)
        with self.assertRaises(ValueError):unpack_bits(torch.zeros(2,dtype=torch.uint8),3,1)
        with self.assertRaises(ValueError):encode_uniform(self.x.expand(2,-1,-1,-1))
        with self.assertRaises(ValueError):encode_uniform(self.x*float('nan'))
        with self.assertRaises(ValueError):encode_uniform(torch.full_like(self.x,1e8))
        with self.assertRaises(ValueError):encode_mixed(self.x,self.mask,packed_mask=pack_bits(~self.mask,1))


if __name__=='__main__':unittest.main()
