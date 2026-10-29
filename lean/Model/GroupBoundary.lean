import Std.Tactic

/-!
Arithmetic abstraction of `src/kvq/quant.py:62` (token-axis group reshape)
and `src/kvq/store.py:43-49` (surviving-token shift). Python's nonnegative
arbitrary-precision index arithmetic is modelled by Nat, not fixed-width CUDA
integer arithmetic. Group size must be positive. Source cache/chunk constants
are 32760/4680; symbolic theorems hold at every Nat size under their hypotheses.
This proves group-index geometry only: not RTN floating-point idempotence,
attention, complete cache lifecycle, or final video-quality superiority.
-/

namespace GroupBoundary

def groupIndex (group position : Nat) : Nat := position / group
def groupOffset (group position : Nat) : Nat := position % group
def shiftedPosition (group blocks position : Nat) : Nat := position + blocks * group

def SameGroup (group p q : Nat) : Prop := groupIndex group p = groupIndex group q
instance (group p q : Nat) : Decidable (SameGroup group p q) := by
  unfold SameGroup; infer_instance

theorem aligned_offset (group blocks position : Nat) (positive : 0 < group) :
    groupOffset group (shiftedPosition group blocks position) = groupOffset group position := by
  simp [groupOffset, shiftedPosition, Nat.add_mod]

theorem aligned_group_index (group blocks position : Nat) (positive : 0 < group) :
    groupIndex group (shiftedPosition group blocks position) = groupIndex group position + blocks := by
  exact Nat.add_mul_div_right position blocks positive

theorem aligned_membership (group blocks p q : Nat) (positive : 0 < group) :
    SameGroup group p q ↔
      SameGroup group (shiftedPosition group blocks p) (shiftedPosition group blocks q) := by
  unfold SameGroup
  rw [aligned_group_index group blocks p positive, aligned_group_index group blocks q positive]
  omega

def smallInputs : List (Nat × Nat × Nat) :=
  [1,2,3,4,8,16,24,32].flatMap fun g =>
    [0,1,2].flatMap fun k => (List.range 65).map fun p => (g,k,p)

def badInputs : List (Nat × Nat × Nat) :=
  smallInputs.filter fun (g,k,p) =>
    groupOffset g (shiftedPosition g k p) != groupOffset g p ||
    groupIndex g (shiftedPosition g k p) != groupIndex g p + k ||
    (List.range 65).any fun q =>
      decide (SameGroup g p q) !=
        decide (SameGroup g (shiftedPosition g k p) (shiftedPosition g k q))

#eval s!"BOUNDED input-triples={smallInputs.length} bad={badInputs.length} comparison-positions=65"
#guard badInputs.isEmpty
#guard smallInputs.any fun (g,k,p) => g == 24 && k == 1 && p == 64
#guard groupIndex 16 15 == 0 && groupIndex 16 16 == 1

-- A real full cache followed by a fresh 4680-token write evicts 4680 tokens.
#guard 7 * 4680 == 32760
#guard 32760 + 4680 - 32760 == 4680
#guard 195 * 24 == 4680
#guard SameGroup 16 7 8
#guard ¬ SameGroup 16 (7 + 4680) (8 + 4680)
#guard groupOffset 24 (32759 + 4680) == groupOffset 24 32759

def vectorInputs : List (Nat × Nat) :=
  [16,24,32].flatMap fun g => [0,7,8,15,16,4679,4680,32759].map fun p => (g,p)

#eval do
  for (g,p) in vectorInputs do
    IO.println s!"VECTOR group={g} shift=4680 position={p} beforeGroup={groupIndex g p} afterGroup={groupIndex g (p+4680)} beforeOffset={groupOffset g p} afterOffset={groupOffset g (p+4680)}"

end GroupBoundary
