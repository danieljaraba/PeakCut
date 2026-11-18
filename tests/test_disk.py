"""Files PeakCut keeps survive the computer switching off.

A sudden shutdown right after a save used to leave a recording's notes the right size but full of zeros, and its
clips stopped showing. Every save now goes to a temporary name, is pushed all the way to the disk, and only then
replaces the old file. Here the push is counted and the old file is checked to stay whole when a save fails."""
import json, os, sys, tempfile
from pathlib import Path
os.environ["PEAKCUT_DATA"] = tempfile.mkdtemp(prefix="pc-disk-"); os.environ["DATABASE_URL"] = ""
sys.path.insert(0, ".")
from studio import disk, jobs

fails = []
def check(n, got, want=True):
    ok = want(got) if callable(want) else got == want
    print(("PASS " if ok else "FAIL ") + n + ("" if ok else f"   got {got!r}")); ok or fails.append(n)

root = Path(os.environ["PEAKCUT_DATA"])
pushed = []
real_fsync = os.fsync
os.fsync = lambda fd: (pushed.append(fd), real_fsync(fd))[1]

f = root / "notes.json"
disk.write_text(f, '{"a": 1}')
check("a saved file reads back", json.loads(f.read_text(encoding="utf-8")), {"a": 1})
check("the save is pushed to the disk", len(pushed) >= 1, True)
check("no temporary file is left behind", [p.name for p in root.iterdir() if p.name.endswith(".tmp")], [])

real_open = open
def broken_open(path, *a, **k):
    if str(path).endswith(".tmp"):
        raise OSError("disk full")
    return real_open(path, *a, **k)
import builtins
builtins.open = broken_open
try:
    disk.write_text(f, '{"a": 2}')
except OSError:
    pass
builtins.open = real_open
check("a save that fails leaves the old file whole", json.loads(f.read_text(encoding="utf-8")), {"a": 1})

d = root / "job1"
d.mkdir()
pushed.clear()
jobs.save({"id": "job1", "dir": str(d), "user": None, "clips": []})
check("a recording's notes are pushed to the disk when saved", len(pushed) >= 1, True)
check("...and read back", json.loads((d / "job.json").read_text(encoding="utf-8"))["id"], "job1")

clip = root / "clip.mp4"
clip.write_bytes(b"video")
pushed.clear()
disk.flush(clip)
check("a finished clip is pushed to the disk", len(pushed), 1)

print("ALL PASSED" if not fails else f"{len(fails)} FAILED")
sys.exit(1 if fails else 0)
