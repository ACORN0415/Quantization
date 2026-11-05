"""Read-only audit/reduction of a completed locked CPU diagnostic."""
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def audit():
    lock = json.loads((HERE / "REGISTRATION_LOCK.json").read_text())
    result = json.loads((HERE / "RESULT.json").read_text())
    assert result["status"] == "completed"
    assert sha(HERE / "diagnose.py") == lock["code_sha256"] == result["code_sha256"]
    assert sha(HERE / "REGISTRATION_PROPOSAL_KO.txt") == lock["registration_sha256"] == result["registration_sha256"]
    rows = {(r["prompt"], r["layer"]): r for r in result["rows"]}
    expected = {(p, l) for p in (0, 4) for l in (8, 20, 28, 29)}
    assert len(result["rows"]) == len(rows) == 8 and set(rows) == expected
    for row in rows.values():
        assert row["manifest_verified"] is True
        assert sha(Path(row["path"])) == row["source_sha256"]
        assert row["reference_energy"] > 0
        for g in ("frame", "chunk"):
            arms = row["groupings"][g]
            assert [a["head"] for a in arms["fixed_heads"]] == list(range(12))
            sizes = {a["payload_bytes"] for a in arms["fixed_heads"]} | {arms["norm"]["payload_bytes"]}
            assert len(sizes) == 1
            for a in arms["fixed_heads"] + [arms["norm"], arms["base4"]]:
                assert abs(a["nse"] - a["sse"] / row["reference_energy"]) <= 1e-15
    groups = {}
    for g in ("frame", "chunk"):
        cases = []
        for (p, l), row in sorted(rows.items()):
            other = rows[(4 if p == 0 else 0, l)]
            # Select only from the opposite development prompt; current outcome is not consulted.
            fixed = min(other["groupings"][g]["fixed_heads"], key=lambda a: (a["sse"], a["head"]))["head"]
            fixed_norm = max(range(12), key=lambda h: (other["mean_key_norm"][h], -h))
            arms = row["groupings"][g]
            norm_sse = arms["norm"]["sse"]
            base_sse = arms["base4"]["sse"]
            fixed_sse = arms["fixed_heads"][fixed]["sse"]
            fixed_norm_sse = arms["fixed_heads"][fixed_norm]["sse"]
            cases.append({"prompt": p, "layer": l, "norm_heads": row["norm_heads"],
                          "fixed_head": fixed, "fixed_norm_head": fixed_norm,
                          "norm_nse": arms["norm"]["nse"], "fixed_nse": arms["fixed_heads"][fixed]["nse"],
                          "fixed_norm_nse": arms["fixed_heads"][fixed_norm]["nse"],
                          "relative_reduction_vs_best_calibrated_fixed": 1 - norm_sse / fixed_sse,
                          "relative_reduction_vs_norm_calibrated_fixed": 1 - norm_sse / fixed_norm_sse,
                          "relative_reduction_vs_base4": 1 - norm_sse / base_sse,
                          "ratio_to_uniform_fixed_expectation": norm_sse / arms["uniform_fixed_expectation_sse"],
                          "norm_sse": norm_sse, "fixed_sse": fixed_sse, "fixed_norm_sse": fixed_norm_sse,
                          "payload_bytes": arms["norm"]["payload_bytes"], "base_payload_bytes": arms["base_payload_bytes"],
                          "cpu_vs_native_post_nse": row["cpu_vs_native_post_nse"]})
        ns, fs = sum(c["norm_sse"] for c in cases), sum(c["fixed_sse"] for c in cases)
        fn = sum(c["fixed_norm_sse"] for c in cases)
        per_prompt_gain = {str(p): 1 - sum(c["norm_sse"] for c in cases if c["prompt"] == p) /
                           sum(c["fixed_sse"] for c in cases if c["prompt"] == p) for p in (0, 4)}
        gain = 1 - ns / fs
        nonworse = sum(c["norm_sse"] <= c["fixed_sse"] * (1 + 1e-5) for c in cases)
        better = sum(c["norm_sse"] < c["fixed_sse"] * (1 - 1e-5) for c in cases)
        worse = sum(c["norm_sse"] > c["fixed_sse"] * (1 + 1e-5) for c in cases)
        constant = sum(len(set(c["norm_heads"])) == 1 for c in cases)
        continue_signal = gain >= .05 and nonworse >= 6 and all(x > 0 for x in per_prompt_gain.values())
        fixed_sufficient = gain < .01 and (constant >= 6 or ns > fs)
        assert abs(gain - result["summary"][g]["relative_sse_reduction"]) <= 1e-14
        assert nonworse == result["summary"][g]["nonworse_cases"]
        assert continue_signal == result["summary"][g]["continuation_signal"]
        groups[g] = {"cases": cases, "relative_sse_reduction": gain,
                     "relative_sse_reduction_vs_norm_calibrated_fixed": 1 - ns / fn,
                     "per_prompt_reduction": per_prompt_gain, "nonworse_cases": nonworse,
                     "better_cases": better, "tie_cases": len(cases) - better - worse, "worse_cases": worse,
                     "exploratory_geometric_ratio_not_used_for_decision": math.exp(sum(math.log(c["norm_sse"] / c["fixed_sse"]) for c in cases) / len(cases)),
                     "constant_norm_head_cases": constant, "continuation_signal": continue_signal,
                     "fixed_sufficient_under_registered_definition": fixed_sufficient}
    assert result["elapsed_seconds"] <= 2700
    assert result["max_rss_bytes"] <= 8 * 1024**3
    total_bytes = sum(p.stat().st_size for p in HERE.rglob("*") if p.is_file())
    assert total_bytes <= 2 * 1024**3
    return {"status": "pass", "cases": 8, "capture_hashes_verified": 8, "groups": groups,
            "result_sha256": sha(HERE / "RESULT.json"), "code_sha256": result["code_sha256"],
            "registration_sha256": result["registration_sha256"], "elapsed_seconds": result["elapsed_seconds"],
            "max_rss_bytes": result["max_rss_bytes"], "task_directory_bytes": total_bytes,
            "independent_holdout": False, "gpu_calls": 0}


if __name__ == "__main__":
    print(json.dumps(audit(), ensure_ascii=False, indent=2))
