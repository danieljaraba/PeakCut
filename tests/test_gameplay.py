"""The "Gameplay fills the screen, facecam in the corner" layout, checked on the finished video itself.

A made-up stream is used: grey moving noise for the game, and a bright pink webcam picture with a yellow
square in its top-left corner, so the webcam can be found by its colour and its shape can be checked.
For a webcam in each corner (and one near the middle) the finished clip must:
- be 1080x1920, with the game filling it and no black bands;
- show the whole webcam, not stretched, in a top corner;
- show it only once (not a second time inside the game behind it);
- keep the heading and the captions off it.
With no webcam the clip falls back to the whole game with a blurred fill, and says so.

Runs against the code directly, with its own empty data folder."""
import os, shutil, subprocess, sys, tempfile
from pathlib import Path
os.environ["PEAKCUT_DATA"] = tempfile.mkdtemp(prefix="pc-gameplay-"); os.environ["DATABASE_URL"] = ""
sys.path.insert(0, ".")
import numpy as np
from studio import facecam, jobs, render

fails = []
def check(n, got, want=True):
    ok = want(got) if callable(want) else got == want
    print(("PASS " if ok else "FAIL ") + n + ("" if ok else f"   got {got!r}")); ok or fails.append(n)

if not shutil.which("ffmpeg"):
    print("FAIL ffmpeg is needed to make the test videos"); sys.exit(1)

render.install_fonts()
work = Path(tempfile.mkdtemp(prefix="pc-gameplay-work-"))
SW, SH, CW, CH = 1280, 720, 320, 180          # the stream, and the webcam picture in it
WORDS = [{"w": w, "s": 0.2 + i * 0.3, "e": 0.45 + i * 0.3} for i, w in enumerate("one two three four five six seven eight".split())]
HOOK = "THIS HEADING IS LONG ENOUGH FOR TWO LINES"


def stream(name, x, y):
    """A three-second stream with the webcam picture at x, y."""
    path = work / f"{name}.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error",
                    "-f", "lavfi", "-i", f"color=c=gray:s={SW}x{SH}:r=25:d=3,noise=c0s=40:allf=t",
                    "-f", "lavfi", "-i", f"color=c=magenta:s={CW}x{CH}:r=25:d=3",
                    "-f", "lavfi", "-i", f"color=c=yellow:s={CW // 2}x{CH // 2}:r=25:d=3",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=3:sample_rate=48000",
                    "-filter_complex", f"[1][2]overlay=0:0[cam];[0][cam]overlay={x}:{y}[v]",
                    "-map", "[v]", "-map", "3:a", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "12",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path)], check=True)
    return path


def frame(path, t, w, h):
    """One picture of a video, as rows of red, green, blue."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t), "-i", str(path), "-frames:v", "1",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8)[:w * h * 3].reshape(h, w, 3).astype(int)


def size(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout.split()[0]
    return tuple(int(v) for v in out.split(","))


def pink(img):
    return (img[..., 0] > 170) & (img[..., 1] < 90) & (img[..., 2] > 170)


try:
    # Where the webcam is in the stream, and where it is asked to go in the clip. A clip that does not say
    # (made before the setting existed) gets the default: the right side, a fifth of the way down.
    places = {"top left": (20, 20, {}), "top right": (SW - CW - 20, 20, {}), "bottom left": (20, SH - CH - 20, {}),
              "bottom right": (SW - CW - 20, SH - CH - 20, {}), "near the middle": (330, 270, {}),
              "set to the left, low": (SW - CW - 20, 20, {"cam_side": "left", "cam_top": 50}),
              "set to the right, very high": (20, SH - CH - 20, {"cam_side": "right", "cam_top": 4})}
    for where, (x, y, chosen) in places.items():
        name = where.replace(" ", "_").replace(",", "")
        src = stream(name, x, y)
        clip = {"start": 0.0, "end": 2.6, "layout": "gameplay", "facecam": [x, y, CW, CH], "hook": HOOK,
                "style": "bounce", "captions": True, **chosen}
        out = render.render_clip(src, work / "out", name, clip, WORDS, SW, SH)
        mp4 = work / "out" / out["file"]
        check(f"{where}: the clip keeps the gameplay layout", out.get("layout"), "gameplay")
        check(f"{where}: the clip has no warning note", out.get("note"), None)
        check(f"{where}: the clip is 1080x1920", size(mp4), (1080, 1920))
        img = frame(mp4, 1.5, 1080, 1920)
        edges = [img[:8].mean(), img[-8:].mean(), img[:, :8].mean(), img[:, -8:].mean()]
        check(f"{where}: the game fills the frame, no black bands", min(edges), lambda v: v > 60)

        # where the webcam is in the finished clip: its pink pixels
        m = pink(img)
        ys, xs = np.nonzero(m)
        check(f"{where}: the webcam is in the clip", len(xs) > 1000)
        if len(xs) <= 1000:
            continue
        # The rows and columns that are mostly pink are the box around the webcam picture (the pink runs
        # along its whole right side and its whole bottom, around the yellow corner).
        rows, cols = np.nonzero(m.sum(1) > 100)[0], np.nonzero(m.sum(0) > 60)[0]
        x0, x1, y0, y1 = cols.min(), cols.max() + 1, rows.min(), rows.max() + 1
        w, h = x1 - x0, y1 - y0
        left, top = chosen.get("cam_side") == "left", chosen.get("cam_top", 20)
        check(f"{where}: the webcam's top is {top}% of the way down", int(y0), lambda v: abs(v - 1920 * top / 100) <= 4)
        check(f"{where}: ...against the {'left' if left else 'right'} side", bool(x0 < 60 if left else x1 > 1020), True)
        check(f"{where}: the webcam is not stretched", round(w / h, 2), lambda r: abs(r - CW / CH) < 0.06)
        check(f"{where}: the webcam is big enough to see", w >= 300)
        # Every pixel of the corner picture matches the stream's webcam, scaled: nothing is drawn over it
        # (heading, captions) and nothing of it is cut off.
        s = frame(src, 1.5, SW, SH)[y: y + CH, x: x + CW]
        ry = (np.arange(h) * CH / h).astype(int)
        rx = (np.arange(w) * CW / w).astype(int)
        off = np.abs(img[y0: y0 + h, x0: x0 + w] - s[ry][:, rx]).mean(2) > 60     # pixels that are not the webcam's
        check(f"{where}: the corner picture is the stream's webcam, whole, with nothing over it",
              round(float(off.mean()), 3), lambda v: v < 0.02)
        # The webcam is shown once: no pink anywhere outside the corner picture (the game behind it is
        # chosen from the part of the stream without the webcam).
        outside = m.copy()
        outside[max(0, y0 - 6): y1 + 6, max(0, x0 - 6): x1 + 6] = False
        check(f"{where}: the webcam is not shown a second time in the game", int(outside.sum()), lambda n: n < 200)

    # ---- the rule for which part of the game is shown, without making a video
    plan = render.gameplay_plan(1280, 720, [330, 270, 320, 180])
    gx, _, gw, _ = plan["game"]
    check("the game part leaves the webcam out when there is room", gx >= 650 or gx + gw <= 330, True)
    plan = render.gameplay_plan(1280, 720, [940, 20, 320, 180])
    check("a webcam outside the middle leaves the game centred", plan["game"][0], (1280 - plan["game"][2]) // 2)
    plan = render.gameplay_plan(1280, 720, [100, 20, 1000, 300])
    check("a webcam too wide to leave out still gives a full game part", plan["game"][2:], lambda g: g[0] >= 400 and g[1] == 720)
    plan = render.gameplay_plan(1280, 720, [900, 300, 1000, 900])
    check("a box that runs off the stream is trimmed to it", plan["cam"], (900, 300, 380, 420))
    tall = render.gameplay_plan(1280, 720, [0, 0, 200, 600])
    check("a tall webcam is kept to a sensible height", tall["pip"][3], lambda v: v <= render.PIP_MAX_H)

    # the captions stay below even the tallest webcam picture, with a two-line heading under it
    _, _, _, cap_y = render.geometry(1280, 720, "gameplay")
    for face in ([0, 0, 200, 600], [0, 0, 320, 180]):
        _, py, _, ph = render.gameplay_plan(1280, 720, face)["pip"]
        y = render._hook_y("gameplay", 300, face, (1280, 720), HOOK)
        lowest = max(py + ph, y + render._hook_shape(HOOK)[4] / 2)
        check(f"captions stay clear of a {face[2]}x{face[3]} webcam and its heading", cap_y - 130 > lowest, True)

    # ---- the webcam position setting
    from studio import creators
    check("a creator's webcam height is kept inside its range",
          [creators._clean({"cam_top": v})["cam_top"] for v in (-5, 0, 3, 20, 60, 99, "35.6")], [3, 3, 3, 20, 60, 60, 35])
    check("a creator's webcam side is right or left", [creators._clean({"cam_side": v})["cam_side"] for v in ("left", "right", "up")],
          ["left", "right", "right"])
    check("new creators get the default position", (creators.DEFAULTS["cam_side"], creators.DEFAULTS["cam_top"]), ("right", 20))
    check("a video's options take the webcam position too", jobs.clean_options({"cam_side": "left", "cam_top": 99}),
          {"cam_side": "left", "cam_top": 60})
    check("...and refuse nonsense", jobs.clean_options({"cam_side": "middle", "cam_top": 1}), {"cam_side": "right", "cam_top": 3})
    for side, top in (("left", 20), ("right", 20), ("right", 45), ("left", 3), ("right", 60)):
        f = render._filter("gameplay", 1920, [940, 20, 320, 180], (1280, 720), side, top)
        px = 26 if side == "left" else 1080 - 380 - 26
        check(f"the webcam is put at the {side}, {top}% down", f"overlay={px}:{int(1920 * top / 100)}" in f, True)
    for top in (3, 8, 15, 20, 30, 45, 60):
        for face in ([0, 0, 320, 180], [0, 0, 200, 600]):
            for hook in ("HI", HOOK):
                _, py, _, ph = render.gameplay_plan(1280, 720, face, "right", top)["pip"]
                hy, bh = render._hook_y("gameplay", 300, face, (1280, 720), hook, "right", top), render._hook_shape(hook)[4]
                cy = render._caption_y("gameplay", 1380, face, (1280, 720), "right", top)
                boxes = [(py, py + ph), (hy - bh / 2, hy + bh / 2), (cy - render.CAP_HALF, cy + render.CAP_HALF)]
                apart = all(a[1] <= b[0] or b[1] <= a[0] for i, a in enumerate(boxes) for b in boxes[i + 1:])
                inside = hy - bh / 2 >= render.SAFE_TOP and cy + render.CAP_HALF <= render.CAP_LOW + render.CAP_HALF
                check(f"webcam {face[2]}x{face[3]} at {top}%, heading of {len(hook)} letters: heading, webcam and captions"
                      " do not touch, and stay on screen", (apart, inside), (True, True))

    # a clip edited in the clip editor keeps its own position, made safe
    jobs._jobs["edit"] = {"id": "edit", "dir": str(work / "edit"), "user": None, "creator": None, "kind": "video",
                          "clips": [{"id": "c1", "start": 0.0, "end": 10.0, "status": "done", "layout": "gameplay"}]}
    (work / "edit").mkdir()
    jobs.request_render("edit", "c1", {"cam_side": "left", "cam_top": 75})
    check("a clip's own webcam position is saved, made safe", {k: jobs._jobs["edit"]["clips"][0].get(k) for k in ("cam_side", "cam_top")},
          {"cam_side": "left", "cam_top": 60})

    # ---- no webcam: the whole game with a blurred fill, and a note that says why
    src = stream("none", 20, 20)
    clip = {"start": 0.0, "end": 2.6, "layout": "gameplay", "facecam": None, "hook": "HI", "style": "bounce", "captions": True}
    out = render.render_clip(src, work / "out", "none", clip, WORDS, SW, SH)
    check("no webcam: the clip uses the blurred layout instead", out.get("layout"), "blur")
    check("no webcam: the clip carries a note saying so", out.get("note"), lambda v: isinstance(v, str) and "webcam" in v)
    check("no webcam: the clip is still 1080x1920", size(work / "out" / out["file"]), (1080, 1920))
    img = frame(work / "out" / out["file"], 1.5, 1080, 1920)
    check("no webcam: the whole stream is shown, its webcam corner too", int(pink(img[:, :540]).sum()) > 1000)

    # the same through the clip maker: what the page is sent
    jobs.facecam.detect = lambda path, ask=None: None
    d = work / "job"
    (d / "clips").mkdir(parents=True)
    c = {"id": "clip001", "start": 0.0, "end": 2.6, "layout": "gameplay", "style": "bounce", "captions": False,
         "status": "queued", "version": 0, "title": "t", "description": "d", "hashtags": [], "hook": "h", "score": 80}
    job = {"id": "job", "dir": str(d), "user": None, "creator": None, "kind": "video", "duration": 3.0,
           "width": SW, "height": SH, "clips": [c], "source_path": str(src)}
    job["options"] = {"cam_side": "left", "cam_top": 33}
    jobs.render_one(job, c, [])
    check("a clip made without a position takes the video's", (c.get("cam_side"), c.get("cam_top")), ("left", 33))
    check("a clip made with no webcam is saved with the blurred layout", (c["status"], c["layout"]), ("done", "blur"))
    check("...and with the note for the page", c.get("note"), lambda v: bool(v))
    page = Path("web/index.html").read_text(encoding="utf-8")
    check("the page shows a clip's note", "c.note" in page, True)
finally:
    shutil.rmtree(work, ignore_errors=True)
    shutil.rmtree(os.environ["PEAKCUT_DATA"], ignore_errors=True)

print("ALL PASSED" if not fails else f"{len(fails)} FAILED")
sys.exit(1 if fails else 0)
