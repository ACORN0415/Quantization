"""Session 6 — VBench dimensions alongside MUSIQ.

Session 5 showed MUSIQ is blind in exactly the region the correction moves
(lambda 1..4 changed the attention distribution enormously and MUSIQ did not
move), so every headline table needs a second opinion. Three VBench dimensions
are computed here with VBench's own code:

  subject_consistency   DINO ViT-B/16 feature cosine across frames (identity
                        drift -- the failure long-horizon KV quantization is
                        supposed to cause)
  temporal_flickering   mean absolute difference between adjacent frames on
                        static regions (MUSIQ reads one frame and cannot see it)
  imaging_quality       MUSIQ-based; included because VBench tables report it,
                        but it is NOT independent evidence from our own MUSIQ

Runs in its own venv (~/gpu/vbench-venv) so the inference venv is untouched.
One score per video = per prompt, so downstream paired t-tests use prompts as
the unit, the same as the MUSIQ analysis.

Usage: vbench_dims.py --runs tag=dir [tag=dir ...] --out_csv path
"""
import argparse, json, os, sys

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dims", nargs="+",
                    default=["subject_consistency", "temporal_flickering",
                             "imaging_quality"])
    a = ap.parse_args()

    torch.hub.set_dir(os.path.expanduser("~/gpu/vbench-venv/hub"))
    from vbench import VBench

    runs = dict(s.split("=", 1) for s in a.runs)
    work = os.path.join(os.path.dirname(os.path.abspath(a.out_csv)), "vbench_work")
    os.makedirs(work, exist_ok=True)
    info = os.path.join(os.path.dirname(
        sys.modules["vbench"].__file__), "VBench_full_info.json")

    rows, per_prompt = [], {}
    for tag, vdir in runs.items():
        if not os.path.isdir(vdir):
            print(f"  ! {vdir} missing, skipped")
            continue
        bench = VBench(torch.device(a.device), info, work)
        bench.evaluate(videos_path=os.path.abspath(vdir), name=tag,
                       dimension_list=list(a.dims), mode="custom_input")
        res = json.load(open(os.path.join(work, f"{tag}_eval_results.json")))
        per_prompt[tag] = {}
        for dim in a.dims:
            score, details = res[dim][0], res[dim][1]
            # details: [{"video_path":..., "video_results": float}, ...]
            vals = [d["video_results"] for d in
                    sorted(details, key=lambda x: x["video_path"])]
            per_prompt[tag][dim] = vals
            rows.append((tag, dim, score, len(vals)))
            print(f"  {tag:24s} {dim:22s} {score:.4f}  (n={len(vals)})")
        del bench
        torch.cuda.empty_cache()

    with open(a.out_csv, "w") as f:
        f.write("config,dimension,score,n_videos\n")
        for t, d, s, n in rows:
            f.write(f"{t},{d},{s:.6f},{n}\n")
    pj = os.path.splitext(a.out_csv)[0] + "_per_prompt.json"
    json.dump(per_prompt, open(pj, "w"), indent=1)
    print("wrote", a.out_csv, "and", pj)


if __name__ == "__main__":
    main()
