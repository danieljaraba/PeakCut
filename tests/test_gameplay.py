"""The "Streamer on top, game below" layout (saved as "gameplay"), checked on the finished video itself.

The clip is split in two, always the same way, like a TikTok of a stream:
- the top 576 pixels are the streamer's webcam, enlarged to the whole width, cut to fill the strip;
- a line in the colour of the platform (Kick green, Twitch purple, YouTube red, white otherwise) between the two;
- the channel's name in a black pill with the platform's logo, centred on that line;
- the bottom 1344 pixels are the game, taken from the part of the stream without the webcam;
- the heading under the pill, the captions low in the game part, above the apps' own buttons.

A made-up stream is used: grey moving noise for the game, and a bright pink webcam picture with a yellow
square in its top-left corner, so the webcam can be found by its colour.
With no webcam the clip falls back to the whole game with a blurred fill, and says so.

Runs against the code directly, with its own empty data folder."""
import os, shutil, subprocess, sys, tempfile
from pathlib import Path
os.environ["PEAKCUT_DATA"] = tempfile.mkdtemp(prefix="pc-gameplay-"); os.environ["DATABASE_URL"] = ""
sys.path.insert(0, ".")
import numpy as np
from studio import auth, creators, ctx, db, jobs, render
db.init()

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
TOP, LINE_AT = 576, 576                        # the webcam strip's height, and where the line is
PURPLE = (0x91, 0x46, 0xFF)


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


def near(img, colour, tol=60):
    return np.abs(img - np.array(colour)).sum(-1) < tol


def scaled_cam(src_img, crop):
    """The webcam part of the stream that should fill the top strip, scaled to 1080x576 the simple way."""
    x, y, w, h = crop
    part = src_img[y: y + h, x: x + w]
    ry, rx = (np.arange(TOP) * h / TOP).astype(int), (np.arange(1080) * w / 1080).astype(int)
    return part[ry][:, rx]


CHANNEL = {"show_channel": True, "channel_label": "", "channel_auto": "TWITCH.TV/SOMEONE", "channel_platform": "twitch"}

try:
    # ---- the finished video, for a webcam in each corner and one near the middle
    places = {"top left": (20, 20), "top right": (SW - CW - 20, 20), "bottom left": (20, SH - CH - 20),
              "bottom right": (SW - CW - 20, SH - CH - 20), "near the middle": (330, 270)}
    for where, (x, y) in places.items():
        name = where.replace(" ", "_")
        src = stream(name, x, y)
        clip = {"start": 0.0, "end": 2.6, "layout": "gameplay", "facecam": [x, y, CW, CH], "hook": HOOK,
                "style": "bounce", "captions": True, **CHANNEL}
        out = render.render_clip(src, work / "out", name, clip, WORDS, SW, SH)
        mp4 = work / "out" / out["file"]
        check(f"{where}: the clip keeps the gameplay layout", out.get("layout"), "gameplay")
        check(f"{where}: the clip has no warning note", out.get("note"), None)
        check(f"{where}: the clip is 1080x1920", size(mp4), (1080, 1920))
        img = frame(mp4, 1.5, 1080, 1920)
        edges = [img[:8].mean(), img[-8:].mean(), img[:, :8].mean(), img[:, -8:].mean()]
        check(f"{where}: no black bands", min(edges), lambda v: v > 60)

        # the top strip is the webcam, enlarged to the whole width, with nothing drawn over it
        plan = render.gameplay_plan(SW, SH, [x, y, CW, CH])
        want = scaled_cam(frame(src, 1.5, SW, SH), plan["cam"])
        off = np.abs(img[:TOP - 60] - want[:TOP - 60]).mean(2) > 60
        check(f"{where}: the top strip is the stream's webcam, enlarged, with nothing over it",
              round(float(off.mean()), 3), lambda v: v < 0.03)
        cam_px = lambda part: (pink(part) | near(part, (255, 255, 0), 120)).mean()      # noqa: E731
        check(f"{where}: the webcam fills the whole width", bool(cam_px(img[:TOP - 60, :40]) > 0.9
                                                               and cam_px(img[:TOP - 60, -40:]) > 0.9), True)

        # the line in Twitch purple, right across, at the edge between the two halves
        line_rows = [r for r in range(LINE_AT - 10, LINE_AT + 10) if near(img[r, :150], PURPLE).mean() > 0.8]
        check(f"{where}: a purple line runs across at about {LINE_AT}", bool(line_rows) and
              near(img[line_rows[0], -150:], PURPLE).mean() > 0.8, True)

        # the channel's pill: black, with purple writing, in the middle of the line
        band = img[LINE_AT - 30: LINE_AT + 30]
        black = (band.max(-1) < 40).sum(0) > 30          # columns mostly black: the pill, not a dark speck
        cols = np.nonzero(black)[0]
        check(f"{where}: the channel pill is on the line", len(cols) > 200, True)
        if len(cols):
            mid = (cols.min() + cols.max()) / 2
            check(f"{where}: ...centred", abs(mid - 540) < 12, True)
            check(f"{where}: ...with purple writing in it", int(near(band[:, cols.min(): cols.max()], PURPLE).sum()) > 300, True)

        # the bottom part is the game: no webcam repeated in it
        check(f"{where}: the webcam is not shown again in the game", int(pink(img[TOP + 40:]).sum()), lambda n: n < 200)

    # ---- the rule for which part of the game is shown, and which part of the webcam
    plan = render.gameplay_plan(1280, 720, [330, 270, 320, 180])
    gx, _, gw, gh = plan["game"]
    check("the game part leaves the webcam out when there is room", gx >= 650 or gx + gw <= 330, True)
    check("the game part has the shape of the bottom half", round(gw / gh, 2), round(1080 / 1344, 2))
    plan = render.gameplay_plan(1280, 720, [940, 20, 320, 180])
    check("a webcam outside the middle leaves the game centred", plan["game"][0], (1280 - plan["game"][2]) // 2)
    plan = render.gameplay_plan(1280, 720, [900, 300, 1000, 900])
    check("a box that runs off the stream is trimmed to it", plan["box"], (900, 300, 380, 420))
    cx, cy, cw, ch = render.gameplay_plan(1280, 720, [100, 100, 200, 200])["cam"]
    check("a square webcam is cut to the strip's shape", cw / ch, lambda r: abs(r - 1080 / 576) < 0.02)
    check("...keeping its whole width", (cx, cw), (100, 200))
    check("...and its upper part, where the face is", (cy >= 100, cy + ch < 300 - 30), (True, True))
    cx, cy, cw, ch = render.gameplay_plan(1280, 720, [100, 100, 400, 100])["cam"]
    check("a very wide webcam keeps its full height and its middle", (cy, ch, abs(cx + cw / 2 - 300) <= 2), (100, 100, True))
    check("no webcam: no plan", render.gameplay_plan(1280, 720, None), None)

    # ---- the heading and the captions stay off the webcam strip and the pill
    pill_bottom = LINE_AT + render.BADGE_H / 2
    for hook in ("HI", HOOK):
        hy, bh = render._hook_y("gameplay", 300, [0, 0, 320, 180], (1280, 720), hook), render._hook_shape(hook)[4]
        cy = render._caption_y("gameplay", 1380, [0, 0, 320, 180], (1280, 720))
        check(f"the heading ({len(hook)} letters) is under the pill", hy - bh / 2 > pill_bottom, True)
        check(f"...close under it, at the top of the game", hy - bh / 2 < pill_bottom + 60, True)
        check(f"the captions are under the heading ({len(hook)} letters)", cy - render.CAP_HALF > hy + bh / 2, True)
        check("the captions stay above the apps' buttons at the bottom", cy + render.CAP_HALF <= 1920 - 280, True)
        check("...in the lower third of the game", cy > TOP + 1344 * 0.6, True)

    # ---- the channel's name on the video
    t = render.channel_badge_text
    check("Kick", t("https://kick.com/kodd"), "KICK.COM/KODD")
    check("Twitch", t("https://www.twitch.tv/bystaxx"), "TWITCH.TV/BYSTAXX")
    check("Twitch, a past broadcast: the uploader's name", t("https://www.twitch.tv/videos/123456", name="Some One"), "TWITCH.TV/SOMEONE")
    check("YouTube, @handle", t("https://www.youtube.com/@MrBeast/streams"), "YOUTUBE.COM/@MRBEAST")
    check("YouTube, /channel/ with a name", t("https://www.youtube.com/channel/UCX6OQ3DkcsbYNE6H8uQQuVA", name="Mr Beast"),
          "YOUTUBE.COM/@MRBEAST")
    check("YouTube, /channel/ without a name", t("https://www.youtube.com/channel/UCX6OQ3DkcsbYNE6H8uQQuVA"), "YOUTUBE.COM")
    check("YouTube, a video: the uploader's name", t("https://youtu.be/abc", name="Ibai"), "YOUTUBE.COM/@IBAI")
    check("the user's own text is used as it is, in capitals", t("https://kick.com/kodd", label="  Follow me! "), "FOLLOW ME!")
    check("...without line breaks or control characters", t("", label="two\nlines\x07"), "TWO LINES")
    check("...cut to 32 letters", len(t("", label="x" * 80)), 32)
    check("switched off: nothing", t("https://kick.com/kodd", label="mine", show=False), "")
    check("no channel and no text: nothing", t(""), "")
    check("another site: nothing", t("https://example.com/someone", name="Someone"), "")
    check("the platform's colour", [render.badge_colour(p) for p in ("kick", "twitch", "youtube", "")],
          ["#53FC18", "#9146FF", "#FF0033", "#FFFFFF"])

    # ---- the settings: on the creator, on a video's options, on a clip
    check("new creators show their channel, with the automatic text",
          (creators.DEFAULTS["show_channel"], creators.DEFAULTS["channel_label"]), (True, ""))
    check("the old webcam position settings are gone", "cam_side" in creators.DEFAULTS or "cam_top" in creators.DEFAULTS, False)
    check("a creator's channel text is cleaned",
          creators._clean({"show_channel": False, "channel_label": "kick.com/me\n" + "x" * 50}),
          {"show_channel": False, "channel_label": "kick.com/me " + "x" * 20})
    check("a video's options take them too", jobs.clean_options({"show_channel": False, "channel_label": " Me "}),
          {"show_channel": False, "channel_label": "Me"})
    check("...and drop the old webcam position", jobs.clean_options({"cam_side": "left", "cam_top": 30}), {})
    with ctx.as_user(auth.local_user().id):
        c = creators.create({"name": "Kodd", "channel": "https://kick.com/kodd", "layout": "gameplay",
                             "show_channel": False, "channel_label": "Mine"})
        job = jobs.create({"type": "live", "url": "https://kick.com/kodd", "name": "Kodd live"}, {}, enqueue=False,
                          creator=c["id"], kind="live")
    check("a creator's video takes the channel settings", (job["options"]["show_channel"], job["options"]["channel_label"]), (False, "Mine"))
    check("its clips get the creator's channel, even from a worker acting for nobody", jobs._channel(job),
          {"show_channel": False, "channel_label": "Mine", "channel_auto": "KICK.COM/KODD", "channel_platform": "kick"})

    # a clip made from a video is given its channel when it is made; a clip saved with the old webcam
    # position is made again without trouble
    src = stream("job", SW - CW - 20, 20)
    d = work / "job"
    (d / "clips").mkdir(parents=True)
    c1 = {"id": "clip001", "start": 0.0, "end": 2.6, "layout": "gameplay", "style": "bounce", "captions": False,
          "status": "queued", "version": 0, "title": "t", "description": "d", "hashtags": [], "hook": "h", "score": 80,
          "facecam": [SW - CW - 20, 20, CW, CH], "cam_side": "left", "cam_top": 45}
    job = {"id": "job", "dir": str(d), "user": None, "creator": None, "kind": "video", "duration": 3.0,
           "width": SW, "height": SH, "clips": [c1], "source_path": str(src), "uploader": "Some One",
           "source": {"type": "url", "url": "https://www.twitch.tv/videos/123456"}, "options": dict(jobs.DEFAULTS)}
    jobs.render_one(job, c1, [])
    check("a clip saved with the old webcam position is made again", (c1["status"], c1["layout"]), ("done", "gameplay"))
    check("a clip of a Twitch video is given its channel",
          {k: c1.get(k) for k in ("show_channel", "channel_label", "channel_auto", "channel_platform")},
          {"show_channel": True, "channel_label": "", "channel_auto": "TWITCH.TV/SOMEONE", "channel_platform": "twitch"})
    c2 = dict(c1, id="clip002", status="queued")
    for k in ("show_channel", "channel_label", "channel_auto", "channel_platform"):
        c2.pop(k)
    job2 = dict(job, clips=[c2], source={"type": "file", "name": "x.mp4"}, uploader="")
    jobs.render_one(job2, c2, [])
    check("a clip of an uploaded file has no channel text", c2.get("channel_auto"), "")

    # the clip editor: switch it off, or write another text, for this clip only
    jobs._jobs["edit"] = {"id": "edit", "dir": str(work / "edit"), "user": None, "creator": None, "kind": "video",
                          "clips": [{"id": "c1", "start": 0.0, "end": 10.0, "status": "done", "layout": "gameplay"}]}
    (work / "edit").mkdir()
    jobs.request_render("edit", "c1", {"show_channel": False, "channel_label": "my\nchannel " + "y" * 40})
    got = jobs._jobs["edit"]["clips"][0]
    check("a clip's own channel settings are saved, made safe", (got.get("show_channel"), got.get("channel_label")),
          (False, ("my channel " + "y" * 40)[:32]))

    # ---- no webcam: the whole game with a blurred fill, and a note that says why
    src = stream("none", 20, 20)
    clip = {"start": 0.0, "end": 2.6, "layout": "gameplay", "facecam": None, "hook": "HI", "style": "bounce",
            "captions": True, **CHANNEL}
    out = render.render_clip(src, work / "out", "none", clip, WORDS, SW, SH)
    check("no webcam: the clip uses the blurred layout instead", out.get("layout"), "blur")
    check("no webcam: the clip carries a note saying so", out.get("note"), lambda v: isinstance(v, str) and "webcam" in v)
    check("no webcam: the clip is still 1080x1920", size(work / "out" / out["file"]), (1080, 1920))

    # ---- the page
    page = Path("web/index.html").read_text(encoding="utf-8")
    check("the page shows a clip's note", "c.note" in page, True)
    check("the layout's short name on the page", 'gameplay:"Streamer + game"' in page, True)
    check("the layout's name", render.LAYOUTS["gameplay"], "Streamer on top, game below")
    check("the page has no webcam position controls any more", "cam_side" in page or "cam_top" in page, False)
    check("the page has the channel settings, in Setup and in the clip editor",
          all(x in page for x in ('="show_channel"', '="channel_label"', 'channelSet("k"', 'channelSet("f"')), True)
finally:
    db.engine.dispose()
    shutil.rmtree(work, ignore_errors=True)
    shutil.rmtree(os.environ["PEAKCUT_DATA"], ignore_errors=True)

print("ALL PASSED" if not fails else f"{len(fails)} FAILED")
sys.exit(1 if fails else 0)
