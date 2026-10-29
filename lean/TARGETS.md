# Formal target scope

| Model | Production interpretation | Number domain | Property/assumptions |
|---|---|---|---|
| GroupBoundary | `src/kvq/quant.py:62` group reshape, `src/kvq/store.py:47` surviving-token shift | Python nonnegative arbitrary-precision indices modelled as Nat; actual experiment uses group 16/24/32, shift 4680, cache 32760 | Shifting coordinates by an integer number of whole groups preserves token offsets and equivalence of group membership. Nonaligned shifts need not preserve it. No claim about RTN floating arithmetic, attention or quality. |

The mathematical quantizer and error-propagation arguments are documented separately in the theory appendix.
Do not treat the arithmetic proof as a formal proof of that entire appendix or of Python/CUDA execution.
