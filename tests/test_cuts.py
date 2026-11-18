"""Where a clip starts and stops, and how it sounds and looks.

Checks that a clip keeps the laugh or shout right after the last word, starts at the beginning of a sentence with
no dead air in front, that a live "clip it" is never inside the clip, that two live clips are kept apart by what
they really cover, and that the picture is enlarged with a sharp filter and the sound is evened out in two passes.
Runs against the code directly, with its own empty data folder; the last part needs ffmpeg."""
import os, shutil, subprocess, sys, tempfile, time
os.environ["PEAKCUT_DATA"] = tempfile.mkdtemp(prefix="pc-cuts-"); os.environ["DATABASE_URL"] = ""
sys.path.insert(0, ".")
from pathlib import Path
import numpy as np
from studio import brain, creators, jobs, live, render

fails = []
def check(n, got, want=True):
    ok = want(got) if callable(want) else got == want
    print(("PASS " if ok else "FAIL ") + n + ("" if ok else f"   got {got!r}")); ok or fails.append(n)


def talk(sentences, start=0.0, pause=0.15, punct=True, gaps=None):
    """Words for some sentences, half a second a word. `gaps` adds a pause before the word with that number."""
    out, t, k = [], start, 0
    for s in sentences:
        parts = s.split()
        for i, w in enumerate(parts):
            t += (gaps or {}).get(k, 0)
            out.append({"w": w + ("." if punct and i == len(parts) - 1 else ""), "s": round(t, 2), "e": round(t + 0.4, 2)})
            t += 0.5
            k += 1
        t += pause
    return out


# ---------------------------------------------------------------- R4: the reaction after the last word is kept
# Two sentences whose last word ends at exactly 10.0 s, then nothing is said.
w = talk(["so I jumped off the edge of the map", "and I landed right on top of him"], start=0.6, pause=0.0)
shift = 10.0 - w[-1]["e"]
w = [dict(x, s=round(x["s"] + shift, 2), e=round(x["e"] + shift, 2)) for x in w]
lines = brain.make_lines(w)
flat = np.full(30, -30.0)
loud = flat.copy(); loud[10] = loud[11] = -20.0           # shouting for two seconds after the last word
_, e_old = brain._snap({"start": w[0]["s"], "end": 10.0}, lines, w, 5, 60, 30)
_, e_flat = brain._snap({"start": w[0]["s"], "end": 10.0}, lines, w, 5, 60, 30, loud=brain.relative_loudness(flat))
_, e_loud = brain._snap({"start": w[0]["s"], "end": 10.0}, lines, w, 5, 60, 30, loud=brain.relative_loudness(loud))
check("a loud reaction after the last word is kept: the clip ends when it calms down", e_loud, lambda e: 11.8 <= e <= 12.2)
check("at normal loudness the clip ends just after the last word, as before", (e_flat, e_old), lambda p: p[0] == p[1] and 10.25 <= p[0] <= 10.4)
s_max, e_max = brain._snap({"start": 0.0, "end": 10.0}, brain.make_lines(w), w, 5, 11, 30, loud=brain.relative_loudness(loud))
check("the reaction never makes the clip longer than the longest allowed", e_max - s_max, lambda d: d <= 11.0)
louder_long = flat.copy(); louder_long[10:20] = -20.0
_, e_cap = brain._snap({"start": w[0]["s"], "end": 10.0}, lines, w, 5, 60, 30, loud=brain.relative_loudness(louder_long))
check("the reaction adds three seconds at most", e_cap, lambda e: 12.8 <= e <= 13.05)
w_next = w + [{"w": "anyway", "s": 11.0, "e": 11.4}, {"w": "moving", "s": 11.5, "e": 11.9}, {"w": "on.", "s": 12.0, "e": 12.4}]
_, e_talk = brain._snap({"start": w[0]["s"], "end": 10.0}, brain.make_lines(w_next), w_next, 5, 60, 30,
                        loud=brain.relative_loudness(loud))
check("...and stops where the next sentence starts", e_talk, lambda e: 10.0 <= e <= 11.0)
rel = brain.relative_loudness(np.concatenate([np.full(200, -30.0), np.full(5, -24.0)]))
check("loudness is measured against the three minutes before: +6 dB is about twice as loud", rel[-1], lambda r: 0.9 <= r <= 1.1)
check("...and a steady level is not a jump", abs(rel[100]) < 0.01)

# ---------------------------------------------------------------- R5: start at a sentence start, no dead air
# One long sentence with a breath in the middle: the line breaks at the breath, the sentence does not.
w = talk(["we were walking home from school", "so I walk into the room and there is this huge spider on the wall",
          "I screamed so loud", "my mum came running"], gaps={13: 0.8})
lines = brain.make_lines(w)
breath = w[13]["s"]
check("(the test transcript has a line starting in the middle of a sentence)", any(abs(l["s"] - breath) < 0.01 for l in lines))
s, e = brain._snap({"start": breath + 0.5, "end": 15.0}, lines, w, 5, 60, 40)
sentence2 = w[6]["s"]
check("a start in the middle of a sentence is moved to the start of that sentence", s, lambda x: abs(x - (sentence2 - 0.15)) < 0.05)
w = talk(["that is how it went"]) + talk(["then out of nowhere he came back", "and took the whole thing", "nobody saw it coming",
                                          "I still cannot believe it"], start=7.0)
s, e = brain._snap({"start": 4.6, "end": 20.0}, brain.make_lines(w), w, 8, 60, 40)
first = 7.0
check("two seconds of silence at the start are cut: the clip starts just before the first word", s, lambda x: first - 0.3 - 0.01 <= x < first)
check("...and still is at least as long as the shortest allowed", e - s, lambda d: d >= 8)

# ---------------------------------------------------------------- R2: the AI's marks and the video's own signals make the score
long_talk = talk(["this is a sentence of eight words right here"] * 20)          # about 82 s of talking, no gaps
base = {"start": 0.0, "end": 60.0, "score": 80}
check("an answer with only a score keeps that score when nothing else is known", brain.final_score(base), 80)
check("...and with the words, when the clip is all talking and starts on a sentence", brain.final_score(base, words=long_talk), 80)
quiet = talk(["this is a sentence of eight words right here"] * 4)              # talking only in the first 17 s
check("a clip that is mostly silence scores lower", brain.final_score(base, words=quiet), lambda x: x < 70)
inside = dict(base, payoff_at=40.0)
outside = dict(base, payoff_at=75.0)
check("a punchline outside the clip costs points", brain.final_score(outside, words=long_talk) < brain.final_score(inside, words=long_talk))
loud_clip = np.zeros(90); loud_clip[30:33] = 1.6
check("a big jump in loudness inside the clip raises the score", brain.final_score(base, loud=loud_clip), lambda x: x > 80)
check("...and so does a part viewers rewatched", brain.final_score(base, replayed=np.arange(90) < 60), lambda x: x > 80)
mid = dict(base, start=long_talk[3]["s"] - 0.15)
check("a clip that starts in the middle of a sentence costs points", brain.final_score(dict(mid, end=mid["start"] + 60), words=long_talk), lambda x: x < 80)
rub = dict(base, rubric={"hook": 9, "payoff": 9, "standalone": 8, "emotion": 8})
check("the four marks of the rubric give the AI's part of the score", brain.final_score(rub), 86)
weak = dict(base, rubric={"hook": 2, "payoff": 3, "standalone": 4, "emotion": 2})
check("...a weak rubric gives a weak score even if the AI's single number was kind", brain.final_score(weak), lambda x: x < 40)
br = brain.score_breakdown(rub, words=long_talk)
check("the breakdown says where the score came from", set(br) >= {"ai", "signals", "penalty", "final"})

got = brain._tidy({"score": 70, "rubric": {"hook": "x", "payoff": 50, "standalone": -3, "emotion": 7.6, "junk": 9},
                   "hook_at": "soon", "payoff_at": 12.5}, 0, 30)
check("rubric marks are kept between 0 and 10, junk is dropped", got.get("rubric"), {"payoff": 10, "standalone": 0, "emotion": 8})
check("a second that is not a number is ignored", (got.get("hook_at"), got.get("payoff_at")), (None, 12.5))
got = brain._tidy({"score": 70, "rubric": "great", "hook_at": -4, "payoff_at": float("nan")}, 0, 30)
check("nonsense in place of the rubric is ignored", (got.get("rubric"), got.get("hook_at"), got.get("payoff_at")), (None, None, None))
props = brain.SCHEMA["items"]["properties"]
check("the AI is asked for the rubric, the hook second and the payoff second",
      set(props["rubric"]["properties"]) == {"hook", "payoff", "standalone", "emotion"} and "hook_at" in props and "payoff_at" in props)
rules = brain._rules("en", min_len=30, max_len=90, transcript="x")
check("the rules explain the rubric and the TikTok length", all(x in rules for x in ("rubric", "hook_at", "payoff_at", "TikTok", "45-75")))
check("...and that the first 3 seconds decide", "first 3 seconds" in rules)

# ---------------------------------------------------------------- R5.3: the clip opens near the hook
story = talk(["one two three four five six seven eight"] * 25)                  # 4.15 s a sentence
s0 = story[0]["s"]
hook_at = s0 + 15.0
s, e = brain._snap({"start": s0, "end": s0 + 40, "hook_at": hook_at}, brain.make_lines(story), story, 30, 90, 200)
check("a hook 15 s after the start moves the start closer to it", s, lambda x: 9.0 <= x <= hook_at - 2)
check("...and the clip is still between the shortest and longest allowed", e - s, lambda d: 30 - 0.5 <= d <= 90)
s2, _ = brain._snap({"start": s0, "end": s0 + 40, "hook_at": s0 + 6}, brain.make_lines(story), story, 30, 90, 200)
check("a hook in the first 8 seconds leaves the start alone", s2, lambda x: x <= s0)
s3, e3 = brain._snap({"start": s0 + 50, "end": s0 + 90, "hook_at": s0 + 75}, brain.make_lines(story), story, 30, 90, 100)
check("near the end of the stretch the start only moves as far as the shortest clip still fits", (s3, e3 - s3), lambda p: p[0] > 55 and p[1] >= 29.5)
# R5.4: the clip ends after the punchline
s, e = brain._snap({"start": s0, "end": s0 + 32, "payoff_at": s0 + 45}, brain.make_lines(story), story, 30, 90, 200)
check("the clip runs on to at least 1.5 s after the punchline", e, lambda x: x >= s0 + 46.5)
check("...ending where a sentence ends", any(abs(e - x) < 0.4 for x in brain.sentence_ends(story)))

# ---------------------------------------------------------------- R11a: a live "clip it" is never inside the clip
story = ["okay so we are going in now", "watch the left side for me", "there is one guy on the roof", "I am going to jump him",
         "here we go here we go", "oh my god he is down", "and the second one too", "and the third one is dead"]
w = talk(story, start=100.0)
t = w[-1]["e"] + 0.2
w += [{"w": x, "s": round(t + i * 0.5, 2), "e": round(t + i * 0.5 + 0.4, 2)} for i, x in enumerate("that was insane".split())]
said = w[-1]["e"] + 0.1
w += [{"w": "clip", "s": round(said, 2), "e": round(said + 0.3, 2)}, {"w": "it", "s": round(said + 0.4, 2), "e": round(said + 0.7, 2)}]
said = w[-2]["s"]
prompts = []
def fake_ai(prompt, schema=None, purpose="", timeout=None):
    prompts.append((purpose, prompt))
    if purpose == "Live: reaction judge":
        return [{"start": 100, "end": said + 5, "score": 90, "reason": "loud"}], "m", 5
    return [{"start": 100.0, "end": said + 3, "score": 95, "hook": "HE GOT THREE", "title": "Three in a row",
             "description": "d", "hashtags": ["a"], "reason": "r"}], "m", 5
brain.ask_ai = fake_ai
clips, _ = brain.judge_transcript(w, 90, said + 15, min_len=15, max_len=60, said_at=said)
check("the AI is told when someone said clip, and that the moment is just before it",
      prompts[-1][1], lambda p: f"{said:.0f}" in p and "Someone said" in p and "before" in p.split("Someone said")[1][:200])
check('the clip ends before the word "clip"', clips and clips[0]["end"], lambda e: bool(e) and e <= said)
check('...but keeps "that was insane", said right before it', clips and clips[0]["end"], lambda e: bool(e) and e >= said - 0.2)
clips, _ = brain.judge_transcript(w, 90, said + 15, min_len=15, max_len=60)
check("without a clip word, the judge's prompt has no such note", "Someone said" in prompts[-1][1], False)

# ---------------------------------------------------------------- the live recording itself
jobs.save = lambda j: None
added = []
jobs.submit_live = lambda fn: None
d = Path(os.environ["PEAKCUT_DATA"]) / "livejob"; d.mkdir()
def session(**over):
    st = dict(creators.DEFAULTS, **over)
    job = {"id": "job-live0001", "dir": str(d), "creator": None, "user": None, "clips": [], "ai": {}, "title": "t",
           "uploader": "u", "options": {}}
    s = live.Session(job, st)
    s.words = list(w); s.t_done = said + 20; s.samples = int((said + 20) * live.SR)
    s.db = [-30.0] * int(said + 20)
    return s

# R11a, as a live check: someone said "clip"
s = session(min_gap_min=0, min_len=15)
s._triggers([x for x in w if x["s"] >= said - 0.01])
check('saying "clip" marks the moment for the AI', s.said and abs(s.said["t"] - said) < 0.01)
s._ai_check((said - 60, said + 15), 'Someone said "clip"', said=s.said["t"])
check('a live clip made after "clip it" ends before those words', s.job["clips"] and s.job["clips"][-1]["end"], lambda e: bool(e) and e <= said)

# R11b: the creator's own phrase goes through the same sentence rules
s = session(trigger_words="mark this moment", min_len=10, max_len=60)
t = w[-1]["e"] + 1.0
phrase = [{"w": x, "s": round(t + i * 0.5, 2), "e": round(t + i * 0.5 + 0.4, 2)} for i, x in enumerate("mark this moment".split())]
s.words += phrase
s.t_done = s.samples / live.SR + 30; s.samples = int(s.t_done * live.SR)
s._triggers(phrase)
check("the creator's phrase asks for a clip", len(s.manual), 1)
brain.describe = lambda *a, **k: {"score": 0, "category": "other", "hook": "H", "title": "T", "description": "", "hashtags": [],
                                  "highlight_words": [], "reason": ""}
s._manual_step()
c = s.job["clips"][-1] if s.job["clips"] else {}
starts = {round(v - 0.15, 2) for v in brain.sentence_starts(s.words)}
check("a clip from the creator's phrase starts at the beginning of a sentence", round(c.get("start", -1), 2) in starts or c.get("start") == 0)
check("...and ends before the phrase, not on a fixed two seconds after it", c.get("end", 99), lambda e: e <= t)
check("...and is at least as long as the shortest allowed", c.get("end", 0) - c.get("start", 0), lambda x: x >= 10)

# R11c: two live clips are kept apart by what they cover, not only by when they start
def gap_case(existing, cand, gap_min):
    s = session(min_gap_min=gap_min, live_per_hour=60)
    s.job["clips"] = [dict(existing, id="clip001", score=90, made_at=0)]
    brain.judge_transcript = lambda *a, **k: ([dict(cand, score=95, category="other", hook="", title="", description="",
                                                     hashtags=[], highlight_words=[], reason="r")], 5)
    brain.judge_signals = lambda *a, **k: ([{"start": cand["start"], "end": cand["end"], "score": 90, "reason": ""}], 5)
    s._ai_check((0, said + 15), "test")
    return len(s.job["clips"]) == 2
check("a clip that overlaps one already made is not made again, even with no gap set",
      gap_case({"start": 100, "end": 130}, {"start": 125, "end": 160}, 0), False)
check("a clip whose ending is too close to another's is not made", gap_case({"start": 100, "end": 130}, {"start": 40, "end": 95}, 1), False)
check("two separate moments close together are both made when their endings are far enough apart",
      gap_case({"start": 100, "end": 130}, {"start": 135, "end": 200}, 1), True)

# ---------------------------------------------------------------- R7: sharper enlarging, even sound
check("the picture is enlarged with the sharp (lanczos) filter",
      all("lanczos" in render._filter(l, 1000, f) for l, f in (("crop", None), ("blur", None), ("gameplay", (10, 10, 300, 200)))))
check("...but the blurred background is not (it is blurred anyway)", "boxblur=6:2,scale=1080:1920," in render._filter("blur", 1000))

ff = shutil.which("ffmpeg")
if not ff:
    print("FAIL ffmpeg is needed for the sound check"); fails.append("ffmpeg")
else:
    work = Path(tempfile.mkdtemp(prefix="pc-cuts-r"))
    src = work / "src.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30:d=5",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=5:sample_rate=48000", "-af", "volume=-20dB",
                    "-shortest", "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", str(src)], check=True)
    seen = []
    real_run = subprocess.run
    def spy(cmd, *a, **k):
        seen.append(list(cmd)); return real_run(cmd, *a, **k)
    render.subprocess.run = spy
    try:
        out = render.render_clip(src, work / "out", "c1", {"start": 0.5, "end": 4.5, "hook": "", "layout": "crop"}, [], 640, 360)
    finally:
        render.subprocess.run = real_run
    enc = [c for c in seen if "-filter_complex" in c]
    af = enc and enc[-1][enc[-1].index("-af") + 1]
    check("the sound is measured first, then evened out with what was measured", af, lambda a: bool(a) and "measured_I=" in a and "linear=true" in a)
    check("...and the clip is made", (work / "out" / "c1.mp4").exists())
    cover = [c for c in seen if "-frames:v" in c]
    check("without a better second, the cover is taken 1.2 s in, as before", cover and cover[-1][cover[-1].index("-ss") + 1], "1.20")
    seen.clear()
    render.subprocess.run = spy
    try:
        render.render_clip(src, work / "out", "c2", {"start": 0.5, "end": 4.5, "hook": "", "layout": "crop"}, [], 640, 360,
                           cover_at=2.5)
    finally:
        render.subprocess.run = real_run
    cover = [c for c in seen if "-frames:v" in c]
    check("the cover is taken at the punchline: -ss is the punchline minus the clip's start",
          cover and cover[-1][cover[-1].index("-ss") + 1], "2.50")
    # the clip's punchline second reaches the cover, counted from the clip's start
    asked = {}
    real_rc = render.render_clip
    render.render_clip = lambda *a, **k: asked.update(k) or {"file": "x.mp4", "cover": "x.jpg"}
    try:
        jd = work / "job"; (jd / "clips").mkdir(parents=True)
        (jd / "s.mp4").write_bytes(b"x")
        cj = {"id": "clip01", "start": 100.0, "end": 140.0, "payoff_at": 121.0, "cover_at": 121.0, "layout": "blur",
              "captions": False, "title": "t", "description": "", "hashtags": [], "hook": "", "score": 80}
        jobs.render_one({"id": "job-cover001", "dir": str(jd), "creator": None, "user": None, "duration": 300,
                         "source_path": str(jd / "s.mp4"), "width": 640, "height": 360, "clips": [cj]}, cj, [])
    finally:
        render.render_clip = real_rc
    check("a clip's cover second is passed to the video maker from the clip's start", asked.get("cover_at"), 21.0)
    check("if the sound cannot be measured, it is evened out in one pass as before",
          render._loudnorm(work / "missing.mp4", 0, 4), "loudnorm=I=-14:TP=-1.5:LRA=11")
    shutil.rmtree(work, ignore_errors=True)

print("ALL PASSED" if not fails else f"{len(fails)} failed")
shutil.rmtree(os.environ["PEAKCUT_DATA"], ignore_errors=True)
sys.exit(1 if fails else 0)
