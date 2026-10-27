# Administrative midpoint checkpoint

40/80 policy cases completed; 40 remain. User requested a pause at this count before reviewing this checkpoint.
The planned sample, parameters, endpoints and analysis remain unchanged. Frozen experimental source was not edited.
Completed-case noise instabilities: 0. Infrastructure retry records: 0.
120 final case tensors retained, with paths/sizes and original hashes in [checkpoint.json](checkpoint.json). BF16 references, videos and raw metrics are also retained outside Git.

The registered prompt-paired analysis is deferred until all 80 cases. No interim significance test or quality-driven sample change is performed.

Resume the remaining cases: `bash tools/resume_policy_study.sh`.
Existing completed stages are checked for matching source and tensor hashes before reuse. The original coordinator resumes final analysis and publication after case 80.

## First-half descriptive frame quality

First five prompts × two seeds. Late = chunks 21–41, raw middle-frame MUSIQ. These are point estimates only; no interim p-values or confidence intervals.
The policy difference is paired within prompt/seed and averaged within prompt before averaging five prompts.

| Bits | BF16 | P1 FR | P3 FR | P3−P1 |
|---|---:|---:|---:|---:|
| 4 | 73.84 | 63.20 | 68.22 | +5.02 |
| 2 | 73.84 | 19.83 | 29.83 | +10.00 |

No final policy-superiority, temporal-consistency or causal-propagation claim follows from this checkpoint. The remaining prompts, seeds, endpoints and full-sample analysis stay fixed.
Data: [late windows](late_windows.csv), [paired point differences](paired_descriptive.csv), [descriptive summary](descriptive.json).
