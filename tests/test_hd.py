"""Clips made from 1080p streams and videos.

- A live recording asks for the stream at up to 1080p (PEAKCUT_LIVE_HEIGHT sets the top), and the rule picks
  1080p on Twitch and Kick when the stream has it, and the best below it when not.
- A video made with the "crop" or "gameplay" layout is downloaded at 1080p even when the quality setting says
  720: those layouts cut a small part out of the picture, so they need the sharper source. "blur" and "zoom"
  keep the quality setting.
- A 1920x1080 stream: the webcam is found in the right place, in the stream's own pixels, and the gameplay clip
  shows it whole, in its corner, not stretched.

Runs against the code directly, with its own empty data folder."""
import os, shutil, subprocess, sys, tempfile
from pathlib import Path
os.environ["PEAKCUT_DATA"] = tempfile.mkdtemp(prefix="pc-hd-"); os.environ["DATABASE_URL"] = ""
os.environ.pop("PEAKCUT_LIVE_HEIGHT", None)
sys.path.insert(0, ".")
import numpy as np
import yt_dlp
from studio import config, facecam, jobs, live, media, render

fails = []
def check(n, got, want=True):
    ok = want(got) if callable(want) else got == want
    print(("PASS " if ok else "FAIL ") + n + ("" if ok else f"   got {got!r}")); ok or fails.append(n)


# ---------------------------------------------------------------- the live setting
def live_height(value):
    """What the setting becomes when PEAKCUT_LIVE_HEIGHT is `value` (None: not set)."""
    env = {k: v for k, v in os.environ.items() if k != "PEAKCUT_LIVE_HEIGHT"}
    if value is not None:
        env["PEAKCUT_LIVE_HEIGHT"] = value
    out = subprocess.run([sys.executable, "-c", "from studio import config; print(config.LIVE_HEIGHT)"],
                         env=env, capture_output=True, text=True)
    return out.stdout.strip() or out.stderr[-300:]


check("live streams are recorded at up to 1080p unless set otherwise", live_height(None), "1080")
check("PEAKCUT_LIVE_HEIGHT=720 records at up to 720p", live_height("720"), "720")
check("a setting that is not a number falls back to 1080p", live_height("high"), "1080")
check("a setting above 1080 is held at 1080", live_height("4000"), "1080")
check("a setting far too low is raised to 360", live_height("10"), "360")
example = Path(".env.example").read_text(encoding="utf-8")
check("the setting is described in .env.example", "PEAKCUT_LIVE_HEIGHT" in example, True)


# ---------------------------------------------------------------- what the live recording asks yt-dlp for
asked = {}


class FakeYDL:
    """Stands in for yt-dlp: keeps the options it is given and says the channel is live."""
    def __init__(self, opts):
        asked.update(opts)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def extract_info(self, url, download=False):
        return {"is_live": True, "title": "t", "url": "https://example.invalid/stream.m3u8"}


real_ydl = yt_dlp.YoutubeDL
yt_dlp.YoutubeDL = FakeYDL
try:
    live.resolve("https://www.twitch.tv/someone")
    fmt_1080 = asked.get("format", "")
    check("the live recording asks for up to 1080p", "height<=1080" in fmt_1080 and "720" not in fmt_1080, True)
    config.LIVE_HEIGHT = 720
    asked.clear()
    live.resolve("https://kick.com/someone")
    check("...and for up to 720p when set to 720", "height<=720" in asked.get("format", ""), True)
finally:
    yt_dlp.YoutubeDL = real_ydl
    config.LIVE_HEIGHT = 1080


def picked(fmt, formats, site="twitch"):
    """The format yt-dlp itself chooses with `fmt`, from a made-up list like the one Twitch or Kick gives."""
    fs = [{"format_id": fid, "url": f"https://example.invalid/{fid}.m3u8", "ext": "mp4", "protocol": "m3u8_native",
           "height": h, "width": h * 16 // 9 if h else None, "vcodec": "none" if fid == "audio_only" else "avc1.64002A",
           "acodec": "mp4a.40.2", "tbr": (h or 1) * 5, **({"quality": 10} if fid.endswith("source") else {})}
          for fid, h in formats]
    info = {"id": "x", "title": "t", "formats": fs, "is_live": True, "webpage_url": f"https://{site}.invalid/x",
            "extractor": f"{site}:stream", "extractor_key": site.title() + "Stream"}
    with real_ydl({"quiet": True, "no_warnings": True, "format": fmt}) as ydl:
        return ydl.process_ie_result(info, download=False).get("format_id")


twitch = [("audio_only", None), ("160p", 160), ("360p", 360), ("480p", 480), ("720p60", 720), ("1080p60", 1080)]
check("on Twitch, the 1080p60 picture is picked when the stream has it", picked(fmt_1080, twitch), "1080p60")
check("on Twitch, a 720p60 stream (no 1080p) gets 720p60", picked(fmt_1080, twitch[:-1]), "720p60")
check("on Twitch, a 1440p source still gets the 1080p60 copy, not the bigger one",
      picked(fmt_1080, twitch + [("1440p60-source", 1440)]), "1080p60")
check("on Twitch, a 1080p source is picked over its own 1080p copy",
      picked(fmt_1080, twitch[:-1] + [("1080p60", 1080), ("1080p60-source", 1080)]), "1080p60-source")
kick = [("160p", 160), ("360p30", 360), ("480p30", 480), ("720p60", 720), ("1080p60", 1080)]
check("on Kick, the 1080p60 picture is picked", picked(fmt_1080, kick, "kick"), "1080p60")
check("on Kick, a stream that only goes to 720p60 gets that", picked(fmt_1080, kick[:-1], "kick"), "720p60")
check("a stream whose sizes are not given still gets a picture",
      picked(fmt_1080, [("a", None), ("b", None)]), lambda v: v in ("a", "b"))


# ---------------------------------------------------------------- what a video is downloaded at
for layout, quality, want in [("crop", 720, 1080), ("gameplay", 720, 1080), ("gameplay", 480, 1080),
                              ("blur", 720, 720), ("zoom", 720, 720), ("blur", 1080, 1080), ("zoom", 480, 480),
                              ("crop", 1080, 1080)]:
    check(f"a {layout} video with quality {quality} is downloaded at {want}p",
          jobs.fetch_height({**jobs.DEFAULTS, "layout": layout, "quality": quality}), want)
check("the quality setting stays 720 by default (for blur and zoom)", jobs.DEFAULTS["quality"], 720)
check("an old job with no quality saved still gets a height", jobs.fetch_height({"layout": "blur"}), 720)
run_job = Path("studio/jobs.py").read_text(encoding="utf-8").split("def _run_job", 1)[1]
check("the whole-video download uses the height chosen by layout", 'int(o["quality"])' not in run_job, True)

work = Path(tempfile.mkdtemp(prefix="pc-hd-work-"))
heights = []
real_section, real_probe = media.download_section, media.probe


def fake_section(url, out_dir, stem, start, end, height=720, exact=False):
    heights.append(height)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{stem}.mp4"
    p.write_bytes(b"x")
    return p


media.download_section = fake_section
media.probe = lambda p: {"width": 1920, "height": 1080, "duration": 30.0}
real_extract, real_load, real_offset = media.extract_audio, media.load_audio, media.find_offset
media.extract_audio = lambda a, b: None
media.load_audio = lambda p: np.zeros(16000, np.int16)
media.find_offset = lambda full, part, expected, search=20.0: expected
try:
    for layout, want in [("gameplay", 1080), ("crop", 1080), ("blur", 720)]:
        heights.clear()
        job = {"id": "j", "dir": str(work / layout), "duration": 600.0, "source": {"type": "url", "url": "https://x.invalid/v"},
               "options": {**jobs.DEFAULTS, "layout": layout}, "clips": []}
        clip = {"id": "c1", "start": 100.0, "end": 130.0, "layout": layout}
        jobs._fetch_piece(job, clip, np.zeros(16000, np.int16))
        check(f"a long {layout} video fetches each clip's part at {want}p", heights, [want])
        heights.clear()
        job = {"id": "j", "dir": str(work / layout), "duration": 600.0, "source": {"type": "url", "url": "https://x.invalid/v"},
               "options": {**jobs.DEFAULTS, "layout": layout}, "clips": []}
        clip = {"id": "c2", "start": 100.0, "end": 130.0, "layout": layout}
        jobs._piece_again(job, clip)
        check(f"re-making a {layout} clip fetches its part again at {want}p", heights, [want])
    # A video that already has its size keeps it for every part, so the parts never differ from each other
    heights.clear()
    job = {"id": "j", "dir": str(work / "old"), "duration": 600.0, "source": {"type": "url", "url": "https://x.invalid/v"},
           "options": {**jobs.DEFAULTS, "layout": "crop"}, "clips": [], "width": 1280, "height": 720}
    jobs._piece_again(job, {"id": "c3", "start": 100.0, "end": 130.0, "layout": "crop"})
    check("re-making a clip of a video whose parts are 720p fetches 720p too, to match them", heights, [720])
finally:
    media.download_section, media.probe = real_section, real_probe
    media.extract_audio, media.load_audio, media.find_offset = real_extract, real_load, real_offset


# ---------------------------------------------------------------- a real 1920x1080 stream
SW, SH, CW, CH = 1920, 1080, 480, 270            # the stream, and the webcam picture in it
CX, CY = SW - CW - 30, SH - CH - 30              # the webcam, near the bottom-right corner
WORDS = [{"w": w, "s": 0.2 + i * 0.3, "e": 0.45 + i * 0.3} for i, w in enumerate("one two three four five six".split())]


def pink(img):
    return (img[..., 0] > 170) & (img[..., 1] < 90) & (img[..., 2] > 170)


def frame(path, t, w, h):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t), "-i", str(path), "-frames:v", "1",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8)[:w * h * 3].reshape(h, w, 3).astype(int)


def size(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout.split()[0]
    return tuple(int(v) for v in out.split(","))


try:
    if not shutil.which("ffmpeg"):
        check("ffmpeg is needed to make the test video", False)
        raise SystemExit
    render.install_fonts()
    src = work / "hd.mp4"
    # Moving noise for the game (it never repeats, so the only still straight lines are the webcam's edges); the webcam is a still pink room with a yellow corner and a person who sways.
    cam = (f"color=c=magenta:s={CW}x{CH}:r=25:d=4[room];color=c=yellow:s={CW // 2}x{CH // 2}:r=25:d=4[mark];"
           f"color=c=0x208020:s=110x130:r=25:d=4[me];[room][mark]overlay=0:0[r2];"
           f"[r2][me]overlay=x='300+40*sin(t*2.5)':y='110+10*sin(t*4)':shortest=1[cam]")
    subprocess.run(["ffmpeg", "-y", "-v", "error",
                    "-f", "lavfi", "-i", f"nullsrc=s={SW}x{SH}:r=25:d=4,noise=alls=24:allf=t",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=4:sample_rate=48000",
                    "-filter_complex", f"{cam};[0][cam]overlay={CX}:{CY}[v]",
                    "-map", "[v]", "-map", "1:a", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "16",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(src)], check=True)
    check("the test stream is 1920x1080", size(src), (SW, SH))

    box = facecam.detect(src)
    check("the webcam is found in a 1080p stream", (box or {}).get("conf"), "high")
    check("...on the right, where it is", (box or {}).get("corner"), lambda v: v in ("right", "bottom-right"))
    b = box or {}
    check("...at the right place, in the stream's own pixels",
          (abs(b.get("x", 0) - CX) <= 2, abs(b.get("y", 0) - CY) <= 2), (True, True))
    check("...with the right size, not the size of the small picture it was looked at",
          (abs(b.get("w", 0) - CW) <= 2, abs(b.get("h", 0) - CH) <= 2), (True, True))

    def corner_check(where, mp4):
        check(f"{where}: the clip is 1080x1920", size(mp4), (1080, 1920))
        img = frame(mp4, 1.5, 1080, 1920)
        m = pink(img)
        rows, cols = np.nonzero(m.sum(1) > 100)[0], np.nonzero(m.sum(0) > 60)[0]
        check(f"{where}: the webcam is in the clip", len(rows) > 50 and len(cols) > 50, True)
        if len(rows) <= 50 or len(cols) <= 50:
            return
        x0, x1, y0, y1 = cols.min(), cols.max() + 1, rows.min(), rows.max() + 1
        w, h = x1 - x0, y1 - y0
        check(f"{where}: the webcam's top is {render.CAM_TOP}% of the way down", int(y0),
              lambda v: abs(v - 1920 * render.CAM_TOP / 100) <= 4)
        check(f"{where}: ...against the right side", bool(x1 > 1020), True)
        check(f"{where}: the webcam is not stretched", round(w / h, 2), lambda r: abs(r - CW / CH) < 0.06)
        s = frame(src, 1.5, SW, SH)[CY: CY + CH, CX: CX + CW]
        ry = (np.arange(h) * CH / h).astype(int)
        rx = (np.arange(w) * CW / w).astype(int)
        off = np.abs(img[y0: y0 + h, x0: x0 + w] - s[ry][:, rx]).mean(2) > 60
        check(f"{where}: the corner picture is the stream's whole webcam", round(float(off.mean()), 3), lambda v: v < 0.03)
        outside = m.copy()
        outside[max(0, y0 - 6): y1 + 6, max(0, x0 - 6): x1 + 6] = False
        check(f"{where}: the webcam is shown only once", int(outside.sum()), lambda n: n < 200)

    face = [b.get("x", CX), b.get("y", CY), b.get("w", CW), b.get("h", CH)]
    clip = {"start": 0.0, "end": 3.0, "layout": "gameplay", "facecam": face, "hook": "A 1080P STREAM",
            "style": "bounce", "captions": True}
    out = render.render_clip(src, work / "out", "hd", clip, WORDS, SW, SH)
    check("a 1080p gameplay clip keeps the gameplay layout", (out.get("layout"), out.get("note")), ("gameplay", None))
    corner_check("1080p gameplay clip", work / "out" / out["file"])

    # A recording that began at 720p and carries on at 1080p after a restart: the video still says 1280x720,
    # but this clip's own piece is 1920x1080. The clip is made from the piece's real size.
    d = work / "job"
    (d / "clips").mkdir(parents=True)
    c = {"id": "clip001", "start": 0.0, "end": 3.0, "layout": "gameplay", "style": "bounce", "captions": False,
         "status": "queued", "version": 0, "title": "t", "description": "d", "hashtags": [], "hook": "h", "score": 80,
         "src": str(src), "src_offset": 0.0, "src_end": 4.0, "facecam": face}
    job = {"id": "job", "dir": str(d), "user": None, "creator": None, "kind": "live", "duration": 4.0,
           "width": 1280, "height": 720, "clips": [c], "options": dict(jobs.DEFAULTS)}
    jobs.render_one(job, c, [])
    check("a clip whose piece is bigger than the video's saved size is made", c["status"], "done")
    corner_check("1080p piece of a 720p recording", d / "clips" / c["file"])
finally:
    shutil.rmtree(work, ignore_errors=True)
    shutil.rmtree(os.environ["PEAKCUT_DATA"], ignore_errors=True)

print("ALL PASSED" if not fails else f"{len(fails)} FAILED")
sys.exit(1 if fails else 0)
