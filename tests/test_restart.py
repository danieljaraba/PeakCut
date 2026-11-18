"""Restarting the studio never makes a second entry for a recording, and never hides clips.

Two things went wrong before. A recording that was running when the studio closed was marked as paused, and
when the studio came back a brand new entry was made for the same channel: the person saw two entries for one
stream, deleted one, and lost its clips. And a computer that switched off suddenly left a recording's notes
full of zeros: the studio skipped the whole recording, so its clips vanished from the page although the videos
were still on disk. Here both are acted out in a temporary data folder. No stream, no AI and no network are used."""
import json, os, subprocess, sys, tempfile, time
from pathlib import Path
os.environ["PEAKCUT_DATA"] = tempfile.mkdtemp(prefix="pc-restart-"); os.environ["DATABASE_URL"] = ""
os.environ["PEAKCUT_ACCOUNTS"] = "1"
sys.path.insert(0, ".")
from studio import config, creators, ctx, jobs, live, tools

fails = []
def check(n, got, want=True):
    ok = want(got) if callable(want) else got == want
    print(("PASS " if ok else "FAIL ") + n + ("" if ok else f"   got {got!r}")); ok or fails.append(n)

tools.setup()
UID = "u_restart01"
(config.USERS_DIR / UID).mkdir(parents=True, exist_ok=True)

# Nothing is really recorded: "is the stream live?" is answered here, and a started recording is only noted.
live_now = {}
def fake_resolve(url):
    if not live_now.get(url):
        raise live.NotLive("This channel is not live right now.")
    return {"urls": ["http://127.0.0.1:9/none"], "title": "Still live", "uploader": "Streamer"}
live.resolve = fake_resolve
started = []
def fake_start(self):
    live.SESSIONS[self.job["id"]] = self
    started.append(self)
live.Session.start = fake_start
tools.missing = lambda: []
jobs.submit_live = lambda fn: None      # clips are not really made here


def video(path, secs=2, fmt=None):
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=320x240:r=15", "-f", "lavfi",
           "-i", "sine=f=440:r=44100", "-t", str(secs), "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
           "-c:a", "aac"]
    cmd += (["-f", fmt] if fmt else ["-movflags", "+faststart"]) + [str(path)]
    subprocess.run(cmd, check=True)


def creator_jobs(cid):
    return [j for j in jobs.all_jobs() if j["user"] == UID and j.get("creator") == cid]


def recording(cid, channel, clips=2, duration=120.4, saved_ago=60):
    """A recording that was running when the studio closed, with `clips` clips already made."""
    with ctx.as_user(UID):
        c = creators.create({"name": cid, "channel": channel})
    jid = time.strftime("%Y%m%d-%H%M%S-") + os.urandom(5).hex()
    d = creators_root(c["id"]) / "live" / jid
    (d / "clips").mkdir(parents=True)
    (d / "buffer").mkdir()
    made = []
    for i in range(1, clips + 1):
        cid_ = f"clip{i:03d}"
        (d / "clips" / f"{cid_}.mp4").write_bytes(b"\0\0\0\x20ftypisom" + b"x" * 100)
        made.append({"id": cid_, "start": 20.0 * i, "end": 20.0 * i + 15, "score": 80, "title": f"Moment {i}",
                     "hook": "", "description": "", "hashtags": [], "status": "done", "file": f"{cid_}.mp4",
                     "cover": f"{cid_}.jpg", "version": 1, "made_at": time.time() - 300, "style": "bounce",
                     "layout": "blur", "captions": True, "src": str(d / "clips" / f"{cid_}.src.ts"),
                     "src_offset": 20.0 * i - 10, "src_end": 20.0 * i + 25})
    # an old piece of the stream is still in the buffer, from before the studio closed
    (d / "buffer" / "p01.csv").write_text("p01_00000.ts,0.000000,6.000000\n", encoding="utf-8")
    (d / "buffer" / "p01_00000.ts").write_bytes(b"G" * 188)
    (d / "words.json").write_text(json.dumps([{"w": "hello", "s": 110.0, "e": 110.5}, {"w": "there", "s": 117.2, "e": 117.9}]),
                                  encoding="utf-8")
    url = live.normalize(channel)
    job = {"id": jid, "user": UID, "kind": "live", "creator": c["id"], "dir": str(d), "created": time.time() - 600,
           "source": {"type": "live", "url": url, "name": f"{cid} live"}, "options": dict(jobs.DEFAULTS),
           "status": "live", "stage": "Recording", "progress": 0.0, "title": "The stream", "uploader": "Streamer",
           "duration": duration, "clips": made, "error": None, "ai": {}, "width": 320, "height": 240}
    (d / "job.json").write_text(json.dumps(job), encoding="utf-8")
    t = time.time() - saved_ago
    os.utime(d / "job.json", (t, t))
    return c["id"], jid, url


def creators_root(cid):
    return config.USERS_DIR / UID / "creators" / cid


# ------------------------------------------------------------- 1. the stream is still on: same entry, same clips
cid, jid, url = recording("still-on", "https://www.twitch.tv/stillon")
cid2, jid2, url2 = recording("ended", "https://www.twitch.tv/ended", clips=1)
cid3, jid3, url3 = recording("watched", "https://www.twitch.tv/watched")
cid4, jid4, url4 = recording("long-ago", "https://www.twitch.tv/longago", saved_ago=5 * 3600)
jobs.load_user(UID)
live_now.update({url: True, url3: True, url4: True})

job = jobs._jobs[jid]
check("after the restart, a recording that was running waits to be picked up again", bool(job.get("was_live")))
live._resume_one(UID, jid)
check("the stream is still on: only one entry for the channel", len(creator_jobs(cid)), 1)
check("...and it is the same entry, recording again", (job["id"], job["status"]), (jid, "live"))
s = live.SESSIONS.get(jid)
check("...a recording is running for it", s is not None and s in started)
check("...its two clips are kept as they were", [(c["id"], c["status"], c["title"]) for c in job["clips"]],
      [("clip001", "done", "Moment 1"), ("clip002", "done", "Moment 2")])
check("...the clock goes on after the last second recorded", (s.samples / live.SR, s.t_done, s.seg_clock), (121.0, 121.0, 121.0))
check("...the words heard before the restart are kept", [w["w"] for w in s.words], ["hello", "there"])
check("...the loudness of every second so far is accounted for", len(s.db), lambda n: n <= 121)
check("...no longer waiting to be picked up", job.get("was_live"), None)

s._add_clip({"start": 130.0, "end": 150.0, "score": 70, "title": "After the restart", "hook": "", "description": "",
             "hashtags": [], "reason": ""})
check("the next clip is called clip003", job["clips"][-1]["id"], "clip003")
check("...and the entry now has three clips", len(job["clips"]), 3)

# the recording carries on: its new pieces of stream follow the old clock; the old pieces are not read again
s.part += 1
s.seg_clock = max(s.seg_clock, s.samples / live.SR)
(s.buf / f"p{s.part:02d}.csv").write_text(f"p{s.part:02d}_00000.ts,0.000000,6.000000\n", encoding="utf-8")
s._scan_segments()
check("new pieces of stream get new names, so old ones are never written over", s.part, 2)
check("new pieces of stream start where the recording stopped", [(p.name, a, b) for p, a, b in s.segments],
      [("p02_00000.ts", 121.0, 127.0)])
check("the clip limit counts the time recorded before the restart", time.time() - s.started_at, lambda x: x >= 120)

# a clip deleted earlier never has its name used again
job["clips"] = [c for c in job["clips"] if c["id"] != "clip002"]
s.n = live._last_clip_number(job, s.dir)
check("clip names keep going up after a clip is deleted", s.n, 3)

# ------------------------------------------------------------- 2. the stream ended while the studio was closed
job2 = jobs._jobs[jid2]
live._resume_one(UID, jid2)
check("the stream ended: the entry stays finished", job2["status"], "done")
check("...it is no longer waiting to be picked up", job2.get("was_live"), None)
check("...no new entry is made", len(creator_jobs(cid2)), 1)
check("...and no recording runs for it", jid2 in live.SESSIONS, False)
check("...its clip is kept", [c["id"] for c in job2["clips"]], ["clip001"])

# ------------------------------------------------------------- 3. auto-record finds the stream first
with ctx.as_user(UID):
    got = live.start_session(creators.get(cid3), url3)
check("auto-record picking the channel up first continues the same entry", got["id"], jid3)
check("...no second entry for the channel", len(creator_jobs(cid3)), 1)
check("...its clips are kept", [c["id"] for c in jobs._jobs[jid3]["clips"]], ["clip001", "clip002"])
live._resume_one(UID, jid3)
check("...and picking it up again later does nothing more", len(creator_jobs(cid3)), 1)

# ------------------------------------------------------------- 4. closed for hours: that is a new stream
with ctx.as_user(UID):
    got = live.start_session(creators.get(cid4), url4)
check("a recording that stopped hours ago is not continued: a new stream gets its own entry", got["id"] != jid4)
check("...and the old one stays finished", (jobs._jobs[jid4]["status"], jobs._jobs[jid4].get("was_live")), ("done", None))

# the words heard after the restart get times on the same clock
import numpy as np
from studio import transcribe
transcribe.transcribe = lambda audio: [{"w": "again", "s": 1.0, "e": 1.4}]
s.pending += np.zeros(live.SR * 8, dtype=np.int16).tobytes()
s._transcribe_step()
check("words heard after the restart come after the old ones on the clock", s.words[-1]["s"], 122.0)
check("the loudness is kept with the words, for the next restart", (s.dir / "loud.json").is_file())

# ------------------------------------------------------------- recover and carry on, in the same folder
cid5, jid5, url5 = recording("both", "https://www.twitch.tv/both")
d5 = Path(jobs._jobs.get(jid5, {}).get("dir") or creators_root(cid5) / "live" / jid5)
(d5 / "buffer" / "p02.csv").write_text("p02_00000.ts,0.000000,6.000000\n", encoding="utf-8")
(d5 / "buffer" / "p02_00000.ts").write_bytes(b"G" * 188)
raw = json.loads((d5 / "job.json").read_text(encoding="utf-8"))
raw["part_clock"] = {"1": 0.0, "2": 100.0}
raw["clips"].append({"id": "clip003", "start": 101.0, "end": 104.0, "status": "rendering", "score": 50, "title": "Half made"})
(d5 / "job.json").write_text(json.dumps(raw), encoding="utf-8")
jobs.load_user(UID)
job5 = jobs._jobs[jid5]
check("a clip cut short by the restart waits to be finished from the buffer", (job5.get("recover"), job5["clips"][-1]["status"]),
      (True, "queued"))
queued = []
jobs.submit_live = queued.append
seen = {}
real_render = live.Session._render
live.Session._render = lambda self, k: seen.update(segments=list(self.segments))
with ctx.as_user(UID):
    live.recover(job5)
live_now[url5] = True
live._resume_one(UID, jid5)
s5 = live.SESSIONS.get(jid5)
check("...while the same recording carries on", s5 is not None and s5.part == 2 and len(creator_jobs(cid5)) == 1)
(d5 / "buffer" / "p03.csv").write_text("p03_00000.ts,0.000000,6.000000\n", encoding="utf-8")
(d5 / "buffer" / "p03_00000.ts").write_bytes(b"G" * 188)
for fn in queued:
    fn()
check("the old pieces are placed where they were on the clock", [(p.name, a) for p, a, b in seen.get("segments", [])],
      [("p01_00000.ts", 0.0), ("p02_00000.ts", 100.0)])
check("finishing the old clips clears only the old pieces, never the new ones",
      sorted(f.name for f in (d5 / "buffer").iterdir()), ["p03.csv", "p03_00000.ts"])
live.Session._render = real_render
jobs.submit_live = lambda fn: None

# ------------------------------------------------------------- 5. notes damaged by a sudden shutdown
UID2 = "u_restart02"
with ctx.as_user(UID2):
    (config.USERS_DIR / UID2).mkdir(parents=True, exist_ok=True)
    cz = creators.create({"name": "Zeroed", "channel": "https://www.twitch.tv/zeroed"})["id"]
zr = config.USERS_DIR / UID2 / "creators" / cz / "live"
d1 = zr / "20261118-103724-28d90ff4cb"
(d1 / "clips").mkdir(parents=True)
(d1 / "job.json").write_bytes(b"\0" * 4096)
video(d1 / "clips" / "clip001.mp4")
(d1 / "clips" / "clip001.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"j" * 50)
(d1 / "clips" / "clip001.json").write_text(json.dumps({"title": "The big moment", "description": "What happened",
                                                       "hashtags": ["#gg"], "hook": "NO WAY", "start": 40.0,
                                                       "end": 58.5, "score": 88}), encoding="utf-8")
d2 = zr / "20261118-110000-aaaaaaaaaa"
(d2 / "clips").mkdir(parents=True)
(d2 / "job.json").write_text('{"id": "20261118-110000-aaaaaaaaaa", "clips": [', encoding="utf-8")   # cut short
(d2 / "words.json").write_bytes(b"\0" * 2048)
video(d2 / "clips" / "clip001.src.ts", secs=4, fmt="mpegts")
(d2 / "clips" / "clip001.mp4").write_bytes(b"\0" * 8192)                                           # the video was lost
(d2 / "clips" / "clip001.json").write_text(json.dumps({"title": "Saved piece", "start": 10.0, "end": 12.0, "score": 50}),
                                           encoding="utf-8")
d3 = config.USERS_DIR / UID2 / "jobs" / "20261118-120000-bbbbbbbbbb"
(d3 / "clips").mkdir(parents=True)
(d3 / "job.json").write_bytes(b"")                                                                 # empty
video(d3 / "clips" / "clip002.mp4")
d4 = zr / "20261118-130000-cccccccccc"
d4.mkdir()
(d4 / "job.json").write_text(json.dumps({"id": "20261118-130000-cccccccccc", "user": "u_someone_else", "kind": "live",
                                         "creator": cz, "created": 1.0, "source": {"type": "live", "url": "", "name": "x"},
                                         "options": {}, "status": "done", "clips": []}), encoding="utf-8")
jobs.load_user(UID2)

j1 = jobs._jobs.get(d1.name)
check("notes full of zeros: the recording still shows", j1 is not None)
check("...it belongs to the person whose folder it is in", j1 and (j1["user"], j1["creator"], j1["kind"]), (UID2, cz, "live"))
check("...with its clip ready and its title", j1 and [(c["id"], c["status"], c["title"]) for c in j1["clips"]],
      [("clip001", "done", "The big moment")])
check("...the clip keeps its times, hook and score", j1 and {k: j1["clips"][0][k] for k in ("start", "end", "hook", "score")},
      {"start": 40.0, "end": 58.5, "hook": "NO WAY", "score": 88})
check("...a copy of the damaged notes is kept", (d1 / "job.json.danado").read_bytes(), b"\0" * 4096)
check("...the notes are written again and read back", json.loads((d1 / "job.json").read_text(encoding="utf-8"))["id"], d1.name)
log = (config.USERS_DIR / UID2 / "creators" / cz / "log.jsonl").read_text(encoding="utf-8")
check("...the creator's log says the notes were rebuilt", "Recording notes were damaged and rebuilt" in log)
check("...the page can show it", j1 and jobs.public(j1)["clips"][0]["title"], "The big moment")

j2 = jobs._jobs.get(d2.name)
check("notes cut short: the recording still shows", j2 is not None)
c2 = (j2 or {}).get("clips", [{}])[0]
check("a clip whose video was lost but whose piece of stream is fine asks to be made again",
      (c2.get("status"), c2.get("error")), ("error", "Press Save and re-make"))
check("...and it can be made again from its piece of stream", bool(c2.get("src")) and c2.get("src_end", 0) > c2.get("src_offset", 99))
check("...with its times inside that piece", c2.get("src_offset", 99) <= c2.get("start", 0) <= c2.get("end", 0) <= c2.get("src_end", 0))
check("a transcript full of zeros is read as no words", jobs.read_words(d2), [])

j3 = jobs._jobs.get(d3.name)
check("empty notes of a plain video: it still shows", j3 and (j3["kind"], j3["creator"], j3["user"]), ("video", None, UID2))
check("...with its clip", j3 and [(c["id"], c["status"]) for c in j3["clips"]], [("clip002", "done")])

check("readable notes keep the owner from the folder, never from the file", jobs._jobs[d4.name]["user"], UID2)

print("ALL PASSED" if not fails else f"{len(fails)} FAILED")
import shutil; shutil.rmtree(os.environ["PEAKCUT_DATA"], ignore_errors=True)
sys.exit(1 if fails else 0)
