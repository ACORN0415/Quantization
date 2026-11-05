"""CPU-only deterministic contact sheets of already generated development videos."""
import hashlib
import json
from pathlib import Path
import resource
import time

resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
import av
from PIL import Image, ImageDraw, ImageOps

HERE = Path(__file__).resolve().parent
REG = HERE / "REGISTRATION.json"
reg = json.loads(REG.read_text())
selected = set(range(0, 501, 8)) | {500}
assert len(selected) == 64
rows = []
for item in reg["inputs"]:
    if time.time() >= reg["absolute_deadline_epoch"]:
        raise TimeoutError("Original screen deadline reached")
    video = Path(item["video"])
    assert hashlib.sha256(video.read_bytes()).hexdigest() == item["video_sha256"]
    sheets = [Image.new("RGB", (2560, 816), "#202020") for _ in range(2)]
    frames = []
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        assert stream.average_rate == 16
        decoded = 0
        for index, frame in enumerate(container.decode(stream)):
            decoded += 1
            if index not in selected:
                continue
            stamp = float(frame.pts * frame.time_base)
            slot = len(frames)
            sheet, pos = divmod(slot, 32)
            col, row = pos % 8, pos // 8
            tile = ImageOps.contain(frame.to_image(), (320, 180))
            x, y = col * 320, row * 204
            sheets[sheet].paste(tile, (x + (320 - tile.width) // 2, y))
            ImageDraw.Draw(sheets[sheet]).text(
                (x + 4, y + 184), f"p{item['id']:02d} frame {index}  t={stamp:.3f}s", fill="white"
            )
            frames.append({"frame": index, "pts_seconds": stamp, "sheet": sheet, "slot": pos})
    assert decoded == 501 and {f["frame"] for f in frames} == selected
    assert all(abs(f["pts_seconds"] - f["frame"] / 16) < 1e-8 for f in frames)
    artifacts = []
    for i, sheet in enumerate(sheets):
        path = HERE / f"p{item['id']:02d}_sheet{i}.jpg"
        sheet.save(path, quality=92, subsampling=0)
        artifacts.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size})
    rows.append({"id": item["id"], "prompt": item["prompt"], "source": str(video), "source_sha256": item["video_sha256"], "decoded_frames": decoded, "viewable_sampled_frames": frames, "sheets": artifacts})
    assert sum(x.stat().st_size for x in HERE.iterdir() if x.is_file()) <= reg["output_cap_bytes"]
    print(f"prepared p{item['id']:02d}: 501 decoded, 64 sampled", flush=True)

result = {
    "status": "prepared", "finished_epoch": time.time(),
    "registration_sha256": hashlib.sha256(REG.read_bytes()).hexdigest(),
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "CPU_decoder_threads": 1, "peak_RSS_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    "new_GPU_model_video_VAE_calls": 0, "items": rows,
    "scope": "Decoded501 per existingMP4; visual samples64 pervideo. No complete-video human inspection, new quality scores or method effects."
}
(HERE / "FRAMES_READY.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"status": result["status"], "clips": len(rows), "sheets": len(rows)*2, "RSS": result["peak_RSS_bytes"]}))
