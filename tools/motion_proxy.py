# -*- coding: utf-8 -*-
"""움직임 대체 지표 (사전 등록 §1): 디코드된 영상의 인접 프레임 평균 절대차 (0~1 스케일 픽셀, 프레임·채널 평균).
VBench dynamic_degree 가 RAFT 가중치 부재로 실행되지 못할 때 쓴다. 영상 파일명 정렬 순서 = 프롬프트 순서.
사용: motion_proxy.py --runs TAG=DIR ... --out out.json"""
import argparse, glob, json, os
import numpy as np
try:
    import imageio.v3 as iio
except Exception:
    import imageio as iio

def motion(path):
    fr = iio.imread(path, plugin="pyav") if hasattr(iio, "imread") else iio.mimread(path)
    fr = np.asarray(fr).astype(np.float32) / 255.0                     # [T,H,W,C]
    if fr.ndim == 3: fr = fr[..., None]
    d = np.abs(fr[1:] - fr[:-1]).mean(axis=(1, 2, 3))                  # [T-1]
    return float(d.mean()), [float(x) for x in d]

ap = argparse.ArgumentParser(); ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args(); res = {}
for s in a.runs:
    tag, d = s.split("=", 1); files = sorted(glob.glob(os.path.join(d, "*.mp4")))
    res[tag] = {"files": [os.path.basename(f) for f in files], "mean_absdiff": [], "per_frame": []}
    for f in files:
        m, pf = motion(f); res[tag]["mean_absdiff"].append(m); res[tag]["per_frame"].append(pf)
    print(f"{tag}: n={len(files)} 평균 절대차 " + " ".join(f"{x:.4f}" for x in res[tag]["mean_absdiff"]))
json.dump(res, open(a.out, "w"), indent=1); print("→", a.out)
