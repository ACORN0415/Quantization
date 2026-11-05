import hashlib
import json
from pathlib import Path
import resource
import time
resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
import av
from PIL import Image, ImageDraw, ImageOps

root = Path(__file__).resolve().parent
reg = json.loads((root / "REGISTRATION.json").read_text())
decision = json.loads((root / "DENSE_FOLLOWUP_DECISION_ROOT.json").read_text())
item = reg["inputs"][0]
assert hashlib.sha256(Path(item["video"]).read_bytes()).hexdigest() == item["video_sha256"]
assert time.time() < reg["absolute_deadline_epoch"]
start, end = decision["dense_interval_frames"]
full = set(decision["full_resolution_frames"])
out = root / "dense_p00"
out.mkdir(exist_ok=False)
sheets = [Image.new("RGB", (2048, 1280), "#202020") for _ in range((end-start+16)//16)]
records, originals = [], []
with av.open(item["video"]) as container:
    stream = container.streams.video[0]
    stream.codec_context.thread_count = 1
    for i, f in enumerate(container.decode(stream)):
        if i > end:
            break
        if time.time() >= reg["absolute_deadline_epoch"]:
            raise TimeoutError("Original deadline reached")
        if i in full or start <= i <= end:
            rgb = f.to_image()
        if i in full:
            dest = out / f"original_frame{i:03d}.png"
            rgb.save(dest)
            originals.append({"frame": i, "path": str(dest), "sha256": hashlib.sha256(dest.read_bytes()).hexdigest()})
        if start <= i <= end:
            n = i-start
            sheet, pos = divmod(n,16)
            tile = ImageOps.contain(rgb,(512,296))
            x,y = (pos%4)*512,(pos//4)*320
            sheets[sheet].paste(tile,(x+(512-tile.width)//2,y))
            ImageDraw.Draw(sheets[sheet]).text((x+4,y+300),f"p00 frame {i} t={float(f.pts*f.time_base):.4f}s",fill="white")
            records.append({"frame":i,"sheet":sheet,"slot":pos,"pts_seconds":float(f.pts*f.time_base)})
assert len(records)==end-start+1 and len(originals)==len(full)
artifacts=[]
for i,sheet in enumerate(sheets):
    path=out/f"sheet{i:02d}.jpg"
    sheet.save(path,quality=90,subsampling=0)
    artifacts.append({"path":str(path),"bytes":path.stat().st_size,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
size=sum(x.stat().st_size for x in root.rglob("*") if x.is_file())
assert size<=reg["output_cap_bytes"]
result={"status":"ready","finished_epoch":time.time(),"original_deadline_epoch":reg["absolute_deadline_epoch"],"video_sha256":item["video_sha256"],"script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"frames":records,"sheets":artifacts,"full_resolution":originals,"total_screen_output_bytes":size,"peak_RSS_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,"new_GPU_model_video_VAE_calls":0}
(root/"DENSE_READY.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
print(json.dumps({"status":"ready","frames":len(records),"full_resolution":len(originals),"sheets":len(artifacts),"bytes":size,"RSS":result["peak_RSS_bytes"]}))
