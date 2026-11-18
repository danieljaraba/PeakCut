"""Where the webcam is in a gaming stream.

The webcam is found by the one thing a webcam always has and a game almost never has: a straight
border that stays in the same place in every frame.

Runs against the code directly, with its own empty data folder."""
import os, shutil, subprocess, sys, tempfile
os.environ["PEAKCUT_DATA"] = tempfile.mkdtemp(prefix="pc-facecam-")
sys.path.insert(0, ".")
from studio import facecam, render
fails = []
def check(n, got, want=True):
    ok = want(got) if callable(want) else got == want
    print(("PASS " if ok else "FAIL ") + n + ("" if ok else f"   got {got!r}")); ok or fails.append(n)

work = tempfile.mkdtemp(prefix="pc-facecam-work-")
FRAMES = "100"          # four seconds at 25 frames. Noise does not compress, so the picture stays small.
NOISE = "nullsrc=s=1280x720:r=25,noise=alls=24:allf=t"


def make(name, extra, where="0:0", more=()):
    """A video with a moving background and one still box in it (and `more` boxes: (picture, where)).

    The background is noise, which never repeats, so the only straight line that stays in the
    same place is the edge of the box."""
    path = os.path.join(work, name)
    boxes = [(extra, where), *more]
    args, graph, last = [], "", "0:v"
    for i, (pic, at) in enumerate(boxes, 1):
        args += ["-f", "lavfi", "-i", pic]
        graph += f"[{last}][{i}:v]overlay={at}[o{i}];"
        last = f"o{i}"
    subprocess.run(["ffmpeg", "-y", "-v", "error",
                    "-f", "lavfi", "-i", NOISE, *args,
                    "-filter_complex", graph.rstrip(";"), "-map", f"[{last}]",
                    "-frames:v", FRAMES,
                    "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", path], check=True)
    return path


def cam_picture(w, h):
    """A webcam picture: a still room with a person in it who sways a little."""
    return (f"color=c=0xC08050:size={w}x{h}:rate=25[room];color=c=0xF0C8A0:size={w * 4 // 15}x{h // 2}:rate=25[me];"
            f"[room][me]overlay=x='{w * 11 // 30}+{w // 12}*sin(t*2.5)':y='{h // 4}+{max(2, h // 20)}*sin(t*4)':shortest=1")


# A panel of the game: a still box with nothing moving in it, like a scoreboard.
PANEL = "color=c=0x203040:size=300x150:rate=25"


def stream(name, cam=None, hud=None, frames=36):
    """A stream drawn picture by picture: see the test that uses it."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter
    rng = np.random.default_rng(7)
    fw, fh = 960, 540
    world = Image.new("RGB", (2400, 1400), (90, 110, 70))
    d = ImageDraw.Draw(world)
    for _ in range(700):
        x, y, s = int(rng.integers(0, 2400)), int(rng.integers(0, 1400)), int(rng.integers(10, 200))
        c = tuple(int(v) for v in rng.integers(0, 256, 3))
        kind = int(rng.integers(0, 4))
        if kind == 0:
            d.rectangle([x, y, x + s, y + int(rng.integers(10, 200))], fill=c)
        elif kind == 1:
            d.ellipse([x, y, x + s, y + s], fill=c)
        elif kind == 2:
            d.line([x, y, x, y + 3 * s], fill=c, width=4)       # long straight walls, which move as it pans
        else:
            d.line([x, y, x + 3 * s, y], fill=c, width=4)
    world = world.filter(ImageFilter.GaussianBlur(1))
    if cam:
        room = Image.new("RGB", cam[2:], (90, 80, 70))
        ImageDraw.Draw(room).rectangle([0, cam[3] * 3 // 4, cam[2], cam[3]], fill=(50, 40, 40))
    path = os.path.join(work, name)
    enc = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{fw}x{fh}",
                            "-r", "3", "-i", "-", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", path],
                           stdin=subprocess.PIPE)
    for i in range(frames):
        t = i / 3
        px, py = int(700 + 650 * np.sin(t * 0.4)), int(430 + 400 * np.cos(t * 0.25))
        im = world.crop((px, py, px + fw, py + fh))
        d = ImageDraw.Draw(im)
        d.rectangle([0, fh - 30, fw, fh], fill=(20, 20, 20))                                  # status bar
        if hud:
            x, y, w, h = hud
            d.rectangle([x, y, x + w, y + h], fill=(15, 25, 35), outline=(240, 240, 240), width=3)
            for k in range(3):
                d.text((x + 12, y + 14 + 20 * k), f"KILLS {(i // 5 + k) % 10}  SCORE {i % 10}0", fill=(255, 255, 255))
        if cam:
            x, y, w, h = cam
            me = room.copy()
            cx, cy, r = w * (0.5 + 0.08 * np.sin(t)), h * 0.45, h * 0.17
            ImageDraw.Draw(me).ellipse([cx - r, cy - r, cx + r, cy + r], fill=(220, 170, 140))
            ImageDraw.Draw(me).ellipse([cx - 2 * r, cy + r, cx + 2 * r, cy + 4 * r], fill=(40, 60, 120))
            im.paste(me, (x, y))
            d.rectangle([x, y, x + w - 1, y + h - 1], outline=(160, 60, 220), width=2)
        enc.stdin.write(np.asarray(im, dtype=np.uint8).tobytes())
    enc.stdin.close()
    enc.wait()
    return path


def jobs_facecam(job):
    from studio import jobs
    return jobs._facecam(job)


try:
    # ---- a webcam in the top-left: a still, lit box
    cam = make("cam.mp4", cam_picture(300, 170))
    box = facecam.detect(cam)
    check("a webcam in the top-left is found", (box or {}).get("corner"), "top-left")
    # the border lies between two pixels, so the box is a pixel short of the drawn one
    check("its size is right", abs((box or {}).get("w", 0) - 300) <= 1, True)
    check("its height is right", abs((box or {}).get("h", 0) - 170) <= 1, True)
    check("it is sure of it", (box or {}).get("conf"), "high")

    # ---- the same box in another corner
    cam_br = make("cam_br.mp4", cam_picture(300, 170), where="980:550")
    box = facecam.detect(cam_br)
    check("a webcam in the bottom-right is found too", (box or {}).get("corner"), "bottom-right")
    check("its position is right", (abs((box or {}).get("x", 0) - 980) <= 1, abs((box or {}).get("y", 0) - 550) <= 1),
          (True, True))

    # ---- a webcam with a gap round it is not stretched out to the edge of the frame
    gap = make("cam_gap.mp4", cam_picture(300, 170), where="40:420")
    box = facecam.detect(gap)
    check("a webcam with a margin round it is found where it is",
          (abs((box or {}).get("x", 0) - 40) <= 2, abs((box or {}).get("y", 0) - 420) <= 2,
           abs((box or {}).get("w", 0) - 300) <= 2, abs((box or {}).get("h", 0) - 170) <= 2), (True, True, True, True))

    # ---- a still panel of the game, with nothing moving in it, is not a webcam
    panel = make("panel.mp4", PANEL, where="20:20")
    check("a still panel with nothing moving in it is never sure to be a webcam",
          (facecam.detect(panel) or {}).get("conf", "none") != "high", True)
    check("so a job does not put a panel of the game in the corner",
          jobs_facecam({"source_path": panel}), None)
    both = make("both.mp4", PANEL, where="20:20", more=[(cam_picture(300, 170), "980:550")])
    box = facecam.detect(both)
    check("with a panel and a webcam on the screen, the webcam is the one found",
          ((box or {}).get("corner"), (box or {}).get("conf")), ("bottom-right", "high"))

    # ---- a "starting soon" screen at the start does not hide the webcam
    intro = make("intro.mp4", cam_picture(300, 170),
                 more=[("color=c=0x101030:size=1280x720:rate=25", "0:0:enable='lt(t,0.8)'")])
    box = facecam.detect(intro)
    check("a webcam is found even after a few seconds of a starting screen",
          ((box or {}).get("corner"), (box or {}).get("conf"), abs((box or {}).get("w", 0) - 300) <= 2),
          ("top-left", "high", True))

    # ---- a video with no webcam in it: nothing that stays still, so nothing is found
    plain = os.path.join(work, "plain.mp4")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", NOISE, "-frames:v", FRAMES,
                    "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", plain], check=True)
    check("a video with nothing still in it finds no webcam", (facecam.detect(plain) or {}).get("conf", "none"), "none")

    # ---- a file that cannot be read must not stop a job
    check("a file that cannot be read gives nothing, not an error",
          facecam.detect(os.path.join(work, "nope.mp4")), None)

    # ---- the job hands the webcam on to the render
    from studio import jobs
    job = {"source_path": cam}
    box = jobs._facecam(job)
    check("the job finds the webcam for its clips", bool(box), True)
    check("the job keeps it on the job", "facecam" in job, True)
    check("the box is the right size", abs(box[2] - 300) <= 1 and abs(box[3] - 170) <= 1, True)
    check("a job with no video finds no webcam", jobs._facecam({"source_path": os.path.join(work, "nope.mp4")}), None)

    # ---- a webcam that is square and runs along one edge, not into a corner
    sq = make("cam_sq.mp4", cam_picture(360, 355), where="920:165")
    box = facecam.detect(sq)
    check("a square webcam along an edge is found", (box or {}).get("corner"), "right")
    check("it starts where the picture starts", abs((box or {}).get("x", 0) - 920) <= 3, True)
    check("it is as tall as the picture", abs((box or {}).get("h", 0) - 355) <= 6, True)

    # ---- something like a real stream: a game that pans over walls and shapes, a status bar, a bordered
    # scoreboard whose numbers change, and a bordered webcam with a person in a room. The scoreboard has
    # stronger borders than the webcam, and used to be the one put in the corner.
    real = stream("real.mp4", cam=(700, 0, 260, 195), hud=(15, 300, 240, 140))
    box = facecam.detect(real)
    check("on a stream with a scoreboard, the webcam is found and not the scoreboard",
          ((box or {}).get("corner"), (box or {}).get("conf")), ("top-right", "high"))
    check("and its box is the webcam picture, to a few pixels",
          all(abs((box or {}).get(k, -99) - v) <= 6 for k, v in zip("xywh", (700, 0, 260, 195))), True)
    real2 = stream("real2.mp4", cam=None, hud=(15, 15, 240, 140))
    check("a stream with only a scoreboard has no webcam to use",
          (facecam.detect(real2) or {}).get("conf", "none") != "high", True)

    # ---- one frame goes to the AI, and its answer is measured again before it is used
    def say(looked, box=None):
        return lambda jpeg: {"looked": looked, "box": box}

    box = facecam.detect(sq, ask=say(True, (0.71, 0.22, 0.29, 0.44)))      # rough, as a model gives it
    check("the box the AI suggests is put onto the borders", abs((box or {}).get("x", 0) - 920) <= 3, True)
    check("a rough box is corrected to the picture's real size", abs((box or {}).get("h", 0) - 355) <= 6, True)
    check("it is still sure of it", (box or {}).get("conf"), "high")
    check("the AI saying there is no webcam is believed", facecam.detect(sq, ask=say(True, None)), None)
    box = facecam.detect(sq, ask=say(True, (0.02, 0.02, 0.06, 0.06)))      # a box on nothing at all
    check("a box that lands on nothing is thrown away and the search carries on",
          abs((box or {}).get("x", 0) - 920) <= 3, True)
    box = facecam.detect(sq, ask=say(False))                              # a model that only reads text
    check("a model that cannot look at pictures changes nothing", abs((box or {}).get("x", 0) - 920) <= 3, True)

    # ---- what the model's one line is turned into
    from unittest.mock import patch
    from studio import brain, ai as _ai
    with patch.object(_ai, "see", return_value=("CAME x=71% y=20% w=29% h=38%", 0, 0)):
        got = brain.look_for_facecam(b"jpeg")
    check("the box in the model's answer is read", [round(v, 2) for v in (got["box"] or ())],
          [0.71, 0.20, 0.29, 0.38])
    with patch.object(_ai, "see", return_value=("NO CAM", 0, 0)):
        check("'no cam' is read as no webcam", brain.look_for_facecam(b"jpeg"), {"looked": True, "box": None})
    with patch.object(_ai, "see", return_value=("I am a text model.", 0, 0)):
        check("an answer that is neither is not believed", brain.look_for_facecam(b"jpeg")["looked"], False)

    # ---- the layout: the game fills the screen and the webcam sits in the corner
    f = render._filter("gameplay", 1920, [0, 0, 300, 170])
    check("the game fills the screen", "scale=1080:1920" in f and "crop=1080:1920" in f, True)
    check("the webcam is cropped out of the source", "crop=300:170:0:0" in f, True)
    check("the webcam is put in the corner", "overlay=" in f, True)
    check("without a webcam the game simply fills the screen",
          "scale=1080:1920" in render._filter("gameplay", 1920, None) and "overlay=" not in render._filter("gameplay", 1920, None), True)
    check("the layout is offered to the user", "gameplay" in render.LAYOUTS, True)
    # the heading must not run under the webcam picture
    pip_h = 380 * 376 / 363
    check("the heading goes below the webcam picture", render._hook_y("gameplay", 300, [0, 0, 363, 376]) > 26 + pip_h, True)
    check("another picture moves the heading with it",
          render._hook_y("gameplay", 300, [0, 0, 300, 170]) < render._hook_y("gameplay", 300, [0, 0, 363, 376]), True)
    check("without a webcam the heading stays at the top", render._hook_y("crop", 300, None), 300)
finally:
    shutil.rmtree(work, ignore_errors=True)      # the pictures are big; they must not be left behind

print("ALL PASSED" if not fails else f"{len(fails)} FAILED")
sys.exit(1 if fails else 0)
