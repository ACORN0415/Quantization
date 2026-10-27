# Post-pilot validity follow-up

Recorded 2026-10-27 after inspecting the completed pilot. This is a validity
check and artifact recovery, not a change to the original analysis or registered
pilot plan. TF/FR interpretation is provisional until the anchoring gate passes.

1. Test the original `caf0b4b` source, whose model/runner fingerprint matches all
   four completed cases. Replace only the quantizer with a clone-on-encode
   identity implementation. Use `compressed_cache(reference=...)`, hence the
   actual `anchor_store` and `ChunkStore` path, for every teacher-forced chunk.
2. Both historical gate prompts, seed 0, 63 frames: require all 42 TF chunks to
   be bit-exact with the lockstep BF16 reference. Independently compare the
   reference chunks with the previously verified `g3_base` captures. Record
   quantizer calls to demonstrate the encoded-segment path was exercised.
3. Only after PASS, regenerate the four original pilot trajectories using the
   unchanged original source and original prompts/seeds. Require each of the
   twelve tensor files to match its archived SHA256 and every latent-distance
   row to match the archived values. Preserve existing quality scores/videos.
4. Retain regenerated tensors as final research artifacts. Remove only
   successfully verified duplicate staging files. New quality exports retain
   tensors by default; do not treat them as disposable intermediates.
5. Align new cross-policy KV diagnostics at the pre-attention read point and
   over the same historical-token mask, excluding the current BF16 chunk.
   Keep post-storage and legacy metrics explicitly separate. These new
   diagnostics do not retroactively alter the pilot latent/MUSIQ measurements.

If the anchoring gate fails, record the failure and suspend TF interpretation;
do not repair the gate criterion or silently replace original pilot numbers.
