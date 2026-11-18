"""Changing a finished clip to the gameplay layout puts the streamer's webcam in the corner.

The webcam used to be looked for only when a clip was first made, and only if the gameplay layout was already
chosen then. A clip made with another layout and switched to gameplay afterwards had no webcam, so it came out
with the game filling the screen and no webcam at all. Here the webcam finder and the video maker are stood in
for, so the test only checks what is asked of them."""
import json, os, sys, tempfile
from pathlib import Path
os.environ["PEAKCUT_DATA"] = tempfile.mkdtemp(prefix="pc-layoutcam-"); os.environ["DATABASE_URL"] = ""
sys.path.insert(0, ".")
from studio import facecam, jobs, render

fails = []
def check(n, got, want=True):
    ok = want(got) if callable(want) else got == want
    print(("PASS " if ok else "FAIL ") + n + ("" if ok else f"   got {got!r}")); ok or fails.append(n)

root = Path(os.environ["PEAKCUT_DATA"])
looked, made = [], []
answer = {"box": {"x": 940, "y": 130, "w": 310, "h": 176, "corner": "top-right", "score": 1.0, "conf": "high"}}

def fake_detect(path, ask=None):
    looked.append(str(path))
    return answer["box"]

def fake_render(src, out_dir, name, clip, words, w, h):
    made.append(dict(clip))
    return {"file": f"{name}.mp4", "thumb": f"{name}.jpg"}

facecam.detect = fake_detect
render.render_clip = fake_render


def new_job(name, live=False):
    d = root / name
    (d / "clips").mkdir(parents=True)
    src = d / "source.mp4"
    src.write_bytes(b"not really a video")
    clip = {"id": "clip001", "start": 10.0, "end": 40.0, "layout": "blur", "style": "bounce", "captions": False,
            "status": "done", "version": 1, "title": "t", "description": "d", "hashtags": [], "hook": "h", "score": 80}
    job = {"id": name, "dir": str(d), "user": None, "creator": None, "kind": "live" if live else "video",
           "duration": 120.0, "width": 1280, "height": 720, "clips": [clip]}
    if live:
        piece = d / "clips" / "clip001.src.ts"
        piece.write_bytes(b"not really a video")
        clip.update(src=str(piece), src_offset=0.0, src_end=60.0)
    else:
        job["source_path"] = str(src)
    return job, clip


# A video clip made with the blurred layout, switched to gameplay
job, clip = new_job("video")
clip["layout"] = "gameplay"
jobs.render_one(job, clip, [])
check("switching to gameplay looks for the webcam", len(looked), 1)
check("the webcam goes into the clip that is made", made[-1].get("facecam"), [940, 130, 310, 176])
check("the clip keeps it for next time", clip.get("facecam"), [940, 130, 310, 176])
check("the whole video remembers it too", job.get("facecam"), [940, 130, 310, 176])

# Another clip of the same video: what was found is used again, not looked for again
clip2 = dict(clip, id="clip002", facecam=None, layout="gameplay")
job["clips"].append(clip2)
jobs.render_one(job, clip2, [])
check("a second clip of the same video does not look again", len(looked), 1)
check("...and gets the same webcam", made[-1].get("facecam"), [940, 130, 310, 176])

# A live clip is looked for in its own saved piece of the stream
looked.clear()
job, clip = new_job("live", live=True)
clip["layout"] = "gameplay"
jobs.render_one(job, clip, [])
check("a live clip is looked for in its own piece of the stream", looked and looked[0].endswith("clip001.src.ts"), True)
check("a live clip gets the webcam", made[-1].get("facecam"), [940, 130, 310, 176])

# Other layouts never look
looked.clear()
job, clip = new_job("blur")
jobs.render_one(job, clip, [])
check("other layouts never look for a webcam", looked, [])

# Nothing sure found: the clip is still made, with the game filling the screen
answer["box"] = None
job, clip = new_job("none")
clip["layout"] = "gameplay"
jobs.render_one(job, clip, [])
check("with no webcam found the clip is still made", clip["status"], "done")
check("...and has no webcam", made[-1].get("facecam"), None)

print("ALL PASSED" if not fails else f"{len(fails)} FAILED")
sys.exit(1 if fails else 0)
