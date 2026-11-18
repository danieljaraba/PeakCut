"""A live recording that starts with a Twitch advert keeps growing after the advert ends.

Twitch puts an advert in front of a stream. The advert and the stream have different clocks, and the first
pictures after the advert came out of ffmpeg with the stream's own time, thousands of seconds ahead. ffmpeg's
piece-cutter then never cut again: everything after the advert piled into one piece that never finished, the
buffer stopped growing at the length of the advert, and no clip could be cut. Here a small stream like that is
made on this machine (an advert with B-frames, then a stream 2250 seconds ahead without them, as Twitch sends it)
and recorded with the same ffmpeg command a live session uses. Nothing outside this machine is contacted."""
import functools, http.server, os, subprocess, sys, tempfile, threading
from pathlib import Path
os.environ["PEAKCUT_DATA"] = tempfile.mkdtemp(prefix="pc-livead-"); os.environ["DATABASE_URL"] = ""
sys.path.insert(0, ".")
from studio import live, tools
tools.setup()
fails = []
def check(n, got, want=True):
    ok = want(got) if callable(want) else got == want
    print(("PASS " if ok else "FAIL ") + n + ("" if ok else f"   got {got!r}")); ok or fails.append(n)

root = Path(os.environ["PEAKCUT_DATA"])
src = root / "stream"; src.mkdir()
def part(name, secs, offset, b_frames, size):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc2=s={size}:r=30",
                    "-f", "lavfi", "-i", "sine=f=440:r=48000", "-t", str(secs), "-c:v", "libx264", "-g", "60",
                    "-bf", str(b_frames), "-preset", "veryfast", "-pix_fmt", "yuv420p", "-c:a", "aac",
                    "-output_ts_offset", str(offset), "-f", "hls", "-hls_time", "2", "-hls_list_size", "0",
                    "-hls_segment_filename", f"{name}_%03d.ts", f"{name}.m3u8"], cwd=src, check=True)
    return [l for l in (src / f"{name}.m3u8").read_text().splitlines()
            if l.startswith("#EXTINF") or l.endswith(".ts")]
ad = part("ad", 30, 1.4, 2, "640x360")          # the advert: its own clock, B-frames
show = part("show", 60, 2250, 0, "320x240")     # the stream itself: 2250 s ahead, no B-frames
(src / "live.m3u8").write_text("\n".join(["#EXTM3U", "#EXT-X-VERSION:3", "#EXT-X-TARGETDURATION:3",
                                          "#EXT-X-MEDIA-SEQUENCE:0", *ad, "#EXT-X-DISCONTINUITY", *show,
                                          "#EXT-X-ENDLIST", ""]))

# served over HTTP on this machine, the way a stream reaches ffmpeg
quiet = type("Quiet", (http.server.SimpleHTTPRequestHandler,), {"log_message": lambda *a: None})
web = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(quiet, directory=str(src)))
threading.Thread(target=web.serve_forever, daemon=True).start()

job = {"id": "job-livead01", "dir": str(root / "rec"), "creator": None, "user": "u_test01", "clips": []}
(root / "rec").mkdir()
s = live.Session(job, {"buffer_min": 20})
s.part = 1
cmd = live.ingest_cmd([f"http://127.0.0.1:{web.server_address[1]}/live.m3u8"], "p01")
check("the recording command corrects the pictures' times (this ffmpeg can)", any("setts" in x for x in cmd),
      live._can_fix_times())
r = subprocess.run(cmd, cwd=s.buf, capture_output=True, timeout=300)
heard = len(r.stdout) / 2 / live.SR
web.shutdown()
check("the sound of the advert and the stream is all heard", heard, lambda h: h > 85)
s._scan_segments()
buffer_sec = s.segments[-1][2] - s.segments[0][1] if s.segments else 0      # as the page shows it
check("the buffer keeps growing after the advert, not stuck at its 30 seconds", round(buffer_sec), lambda b: b >= 80)
check("the stream after the advert is cut into many pieces, not one that never ends",
      len([x for x in s.segments if x[1] >= 30]), lambda n: n >= 8)
sizes = sorted(f.stat().st_size for f in s.buf.glob("p01_*.ts"))
check("no single piece holds most of the recording", bool(sizes) and sizes[-1] < sum(sizes) / 3)
check("no piece is said to last much longer than a piece should", max((b - a for _, a, b in s.segments), default=0),
      lambda d: 0 < d <= live.SEG_SEC * 2)

print("ALL PASSED" if not fails else f"{len(fails)} failed")
import shutil; shutil.rmtree(os.environ["PEAKCUT_DATA"], ignore_errors=True)
sys.exit(1 if fails else 0)
