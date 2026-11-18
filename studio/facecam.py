"""Where the facecam is.

A gaming stream is a game with a small webcam picture in one corner of it. The
short that comes out of it is better when the game fills the screen and the
webcam sits in the corner of it, instead of the whole 16:9 frame squeezed into
a band with blurred fill above and below.

The webcam is found by three things a webcam has and a game almost never has:

- Its edges are straight lines that stay in the same place in every frame. A wall
  in the game moves; the edge of the webcam picture does not. Where the webcam has
  no drawn border, the line is still there: on one side the picture keeps moving
  (the game), on the other it hardly moves (a room).
- Each edge runs the whole length of that side of the box, from corner to corner.
  This is measured along the side itself, not along the whole height or width of
  the frame, so a line from a scoreboard and a line from the webcam can never be
  joined into a box that is neither.
- Something moves inside it: the person. The game's own panels (a scoreboard, an
  inventory, a bar of buttons) are boxes that stay still too, but nothing in them
  moves like a person does. A box with nothing moving in it may still be shown to
  the AI, but it is never used on its own.

One frame can also be shown to the AI (`detect(path, ask=...)`), which is better
than any measurement at telling a person from a game character. Its answer is
measured again on the picture before it is used, so being wrong costs a search,
not a clip.

Only numpy and ffmpeg, which are already here.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

W, H = 320, 180            # the picture is looked at small to find the box: only the lines matter
MAX_FRAMES = 24            # frames spread through the video
FULL_FRAMES = 5            # frames at full size, to put the box on the exact pixel
MIN_SIZE = 0.12            # the smallest a webcam may be, as a part of the width or height
MAX_SIZE = 0.60            # and the largest
ASPECT_MIN = 0.55          # the narrowest: a phone held upright is about 0.56
ASPECT_MAX = 3.20          # the widest: a banner or a ticker is wider than this
STILL_FRAC = 0.35          # how much of the game's own movement a webcam picture may have
STILL_MIN = 4.0            # and the floor, for a game that itself is almost still
STILL_SHARE = 0.30         # at most this much of a webcam picture may be moving, or the game is showing through
EDGE_FLUSH = 0.02          # a side this close to the frame's own edge is the edge
MATCH = 0.35               # how much a box the AI points at must cover a measured box to be that box
EDGE_MIN = 10.0            # a still line is at least this much brighter or darker on one side (0 to 255)
EDGE_REL = 2.5             # and this many times more than the picture's usual line
COVER = 0.80               # how much of a side a line must run along
END_COVER = 0.75           # and how much of each end of it: the line must reach the corners
LIFE = 0.03                # how much of a webcam picture must move at some point: the person in it


def _frames(path: str | Path, count: int = MAX_FRAMES, size: tuple[int, int] | None = (W, H)) -> np.ndarray:
    """The video as a few grey frames, spread through it. Small unless `size` is None."""
    fw, fh = size if size else _size(path)
    if not fw or not fh:
        return np.zeros((0, H, W), dtype=np.uint8)
    try:
        dur = _duration(path)
        rate = max(count / dur, 0.02) if dur > 0 else 0.2
        scale = f"scale={fw}:{fh}," if size else ""
        proc = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(path),
             "-vf", f"fps={rate:.4f},{scale}format=gray",
             "-frames:v", str(count), "-f", "rawvideo", "-"],
            capture_output=True, timeout=180)
    except Exception:  # noqa: BLE001 - no ffmpeg or an unreadable file: no facecam, not a failure
        return np.zeros((0, fh, fw), dtype=np.uint8)
    raw = proc.stdout or b""
    one = fw * fh
    n = len(raw) // one
    if n < (4 if size else 1):
        return np.zeros((0, fh, fw), dtype=np.uint8)
    return np.frombuffer(raw[:n * one], dtype=np.uint8).reshape(n, fh, fw)


def _duration(path: str | Path) -> float:
    try:
        proc = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                               "-of", "default=nw=1:nk=1", str(path)],
                              capture_output=True, text=True, timeout=60)
        return float((proc.stdout or "").strip() or 0)
    except Exception:  # noqa: BLE001
        return 0.0


def _size(path: str | Path) -> tuple[int, int]:
    try:
        proc = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                               "-show_entries", "stream=width,height", "-of", "default=nw=1", str(path)],
                              capture_output=True, text=True, timeout=60)
    except Exception:  # noqa: BLE001
        return 0, 0
    w = h = 0
    for line in (proc.stdout or "").splitlines():
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        try:
            if k.strip() == "width":
                w = int(v.strip())
            elif k.strip() == "height":
                h = int(v.strip())
        except ValueError:
            continue
    return w, h


def _stillness(frames: np.ndarray) -> np.ndarray:
    """How much each pixel changes over the frames.

    The middle half of what each pixel shows is used, not all of it: a "starting soon" screen or a loading
    screen at the start of a stream is a few frames out of many, and must not make a still webcam look
    like it moves."""
    lo, hi = np.percentile(frames.astype(np.float32), [25, 75], axis=0)
    return hi - lo


class _Lines:
    """Where the still straight lines are, on the small picture.

    `v[y, b]` says whether there is a still line between columns b-1 and b at row y, give or take a pixel;
    `h[r, x]` the same for a line between rows r-1 and r. Index 0 and the last are the frame's own edges."""

    def __init__(self, frames: np.ndarray, still: np.ndarray):
        f = frames.astype(np.float32)
        ex = np.median(np.abs(np.diff(f, axis=2)), axis=0)            # (H, W-1): still edges across columns
        ey = np.median(np.abs(np.diff(f, axis=1)), axis=0)            # (H-1, W)
        game = float(np.median(still))
        te = max(EDGE_MIN, EDGE_REL * float(np.median(np.concatenate([ex.ravel(), ey.ravel()]))))
        ts = max(2 * STILL_MIN, 0.5 * game)
        # A line between a still picture and the moving game, where the webcam has no border drawn round it.
        # The small picture blends the two over a pixel, so the pixels one step further out are compared.
        sp = np.pad(still, 1, mode="edge")
        sx = np.abs(sp[1:-1, 2:] - sp[1:-1, :-2])[:, 1:]              # (H, W-1), across column boundary b
        sy = np.abs(sp[2:, 1:-1] - sp[:-2, 1:-1])[1:, :]              # (H-1, W)
        okx = (ex >= te) | (sx[:, :ex.shape[1]] >= ts)
        oky = (ey >= te) | (sy[:ey.shape[0], :] >= ts)
        v = np.zeros((H, W + 1), dtype=bool)
        v[:, 1:W] = okx
        h = np.zeros((H + 1, W), dtype=bool)
        h[1:H, :] = oky
        # give or take a pixel: a line on the small picture can fall either side of a pixel boundary
        self.v = v | np.roll(v, 1, axis=1) | np.roll(v, -1, axis=1)
        self.h = h | np.roll(h, 1, axis=0) | np.roll(h, -1, axis=0)
        self.v[:, 0] = self.v[:, W] = False
        self.h[0, :] = self.h[H, :] = False
        # running totals, so how much of any side a line covers is two lookups
        self.cv = np.vstack([np.zeros((1, W + 1)), np.cumsum(self.v, axis=0)])    # (H+1, W+1)
        self.ch = np.hstack([np.zeros((H + 1, 1)), np.cumsum(self.h, axis=1)])    # (H+1, W+1)

    def side(self, vertical: bool, at: int, a: int, b: int) -> tuple[float, float]:
        """(how much of the side the line covers, how much of its weaker end). A side on the frame's own
        edge needs no line."""
        edge = W if vertical else H
        if at <= 0 or at >= edge or b <= a:
            return 1.0, 1.0
        c = self.cv[:, at] if vertical else self.ch[at, :]

        def part(lo, hi):
            return float(c[hi] - c[lo]) / max(1, hi - lo)
        end = max(2, int(round(0.15 * (b - a))))
        return part(a, b), min(part(a, a + end), part(b - end, b))

    def candidates(self, vertical: bool, limit: int = 18) -> list[int]:
        """The places where a long still line runs: the sides a webcam box could have."""
        rows = self.v.T if vertical else self.h                     # one row per possible line
        need = int(0.8 * MIN_SIZE * (H if vertical else W))
        scored = []
        for at in range(3, rows.shape[0] - 3):          # a line hard against the frame's edge is that edge
            line = rows[at]
            if line.sum() < need:
                continue
            d = np.diff(np.concatenate([[0], line.astype(np.int8), [0]]))
            run = int((np.flatnonzero(d == -1) - np.flatnonzero(d == 1)).max())
            if run >= need:
                scored.append((run, at))
        out = []
        for run, at in sorted(scored, reverse=True):
            if all(abs(at - o) > 3 for o in out):     # one line shows as a band two or three pixels wide
                out.append(at)
            if len(out) >= limit:
                break
        return sorted(out)


def _plausible(x0, y0, x1, y1, still, light, game, share=STILL_SHARE):
    """Could this rectangle be a webcam picture? Judged on its shape, its stillness and its light.

    Returns how much it moves, or None. Nothing here says which corner it is in: a webcam may sit in a
    corner, along one edge, or with a margin all round."""
    bw, bh = (x1 - x0) / W, (y1 - y0) / H
    if not (MIN_SIZE <= bw <= MAX_SIZE and MIN_SIZE <= bh <= MAX_SIZE):
        return None
    # measured in pixels: a 4:3 picture is 1.33, a phone held upright is 0.56
    aspect = (bw * W) / max(1e-6, bh * H)
    if not (ASPECT_MIN <= aspect <= ASPECT_MAX):
        return None
    a, b, c, d = int(x0), int(y0), int(x1), int(y1)
    inside = still[b:d, a:c]
    if not inside.size:
        return None
    cut = max(STILL_MIN, STILL_FRAC * game)
    moving = float(np.median(inside))
    if moving > cut:
        return None                    # it changes with the game: it is the game
    if float((inside > cut).mean()) > share:
        return None                    # a good part of it moves: the game is showing through
    if float(light[b:d, a:c].mean()) < 0.30 * float(light.mean()):
        return None                    # a black hole, not a picture
    return moving


def _alive(x0, y0, x1, y1, still, game) -> float:
    """How much of the box moves at some point: the person in a webcam. A game's own panel stays still.

    Only the inside of the box is looked at, away from its border, and only movement in patches, not in
    thin strokes: the numbers on a scoreboard change, but they are thin lines, and a person is not."""
    px, py = int(round(0.1 * (x1 - x0))), int(round(0.1 * (y1 - y0)))
    inside = still[int(y0) + py:int(y1) - py, int(x0) + px:int(x1) - px]
    if inside.shape[0] < 3 or inside.shape[1] < 3:
        return 0.0
    moving = inside > max(STILL_MIN, 0.15 * game)
    patch = moving[1:-1, 1:-1].copy()
    for dy in (-1, 0, 1):              # a pixel counts only when the pixels all round it move too
        for dx in (-1, 0, 1):
            patch &= moving[1 + dy:moving.shape[0] - 1 + dy, 1 + dx:moving.shape[1] - 1 + dx]
    return float(patch.mean())


def _boxes(lines: _Lines, still, light) -> list[dict]:
    """Every rectangle whose sides are all still lines from corner to corner, that could be a camera.

    Best first: one with a person moving in it before one without, and of those the bigger, so a webcam
    wins over a poster on the wall behind the streamer."""
    game = float(np.median(still))
    xs = [0] + lines.candidates(True) + [W]
    ys = [0] + lines.candidates(False) + [H]
    out = []
    for i, x0 in enumerate(xs):
        for x1 in xs[i + 1:]:
            if not (MIN_SIZE * W <= x1 - x0 <= MAX_SIZE * W):
                continue
            for j, y0 in enumerate(ys):
                for y1 in ys[j + 1:]:
                    if not (MIN_SIZE * H <= y1 - y0 <= MAX_SIZE * H):
                        continue
                    own = (x0 > 0) + (x1 < W) + (y0 > 0) + (y1 < H)
                    if own < 2:
                        continue       # a box needs two borders of its own, or it is just the frame
                    sides = (lines.side(True, x0, y0, y1), lines.side(True, x1, y0, y1),
                             lines.side(False, y0, x0, x1), lines.side(False, y1, x0, x1))
                    if any(c < COVER or e < END_COVER for c, e in sides):
                        continue
                    moving = _plausible(x0, y0, x1, y1, still, light, game)
                    if moving is None:
                        continue
                    alive = _alive(x0, y0, x1, y1, still, game)
                    score = min(c for c, _ in sides)
                    out.append({"x": x0, "y": y0, "x1": x1, "y1": y1, "score": round(score, 2),
                                "moving": moving, "alive": round(alive, 3),
                                "key": (alive >= LIFE, (x1 - x0) * (y1 - y0), score)})
    return sorted(out, key=lambda c: c["key"], reverse=True)


def _overlap(a, box) -> float:
    """How much two boxes cover the same ground: 1 when they are the same, 0 when they do not touch."""
    ax0, ay0, ax1, ay1 = a["x"], a["y"], a["x1"], a["y1"]
    bx, by, bw, bh = box
    bx0, by0, bx1, by1 = bx * W, by * H, (bx + bw) * W, (by + bh) * H
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union > 0 else 0.0


def _snap(full: np.ndarray, box) -> tuple[float, float, float, float] | None:
    """A box on the small picture, put onto the exact pixel of its borders in the full-size frames.

    Each side is looked for only along that side (not along the whole height or width of the frame, where
    the game has lines of its own), and only near where the small picture put it. The frames are taken
    together by their middle value, so a line in the game that passes by in one frame does not count."""
    if full is None or not len(full):
        return None
    _, fh, fw = full.shape
    sx, sy = fw / W, fh / H
    spread = max(3, int(round(3 * max(sx, sy))))       # the small picture is a few of its pixels out
    x0, y0, x1, y1 = box
    X0, Y0, X1, Y1 = x0 * sx, y0 * sy, x1 * sx, y1 * sy

    def line(vertical, approx, a, b):
        extent = fw if vertical else fh
        if approx <= 0:
            return 0.0                                 # a side on the frame's own edge stays there
        if approx >= extent:
            return float(extent)
        pad = int(0.1 * (b - a))                       # the corners are left out: the other sides are there
        a, b = int(a) + pad, max(int(a) + pad + 1, int(b) - pad)
        lo, hi = max(1, int(round(approx)) - spread), min(extent - 1, int(round(approx)) + spread)
        best, at = -1.0, approx
        for p in range(lo, hi + 1):
            if vertical:
                step = np.abs(full[:, a:b, p].astype(np.int16) - full[:, a:b, p - 1]).sum(axis=1)
            else:
                step = np.abs(full[:, p, a:b].astype(np.int16) - full[:, p - 1, a:b]).sum(axis=1)
            s = float(np.median(step))
            if s > best:
                best, at = s, float(p)
        return at

    return (line(True, X0, Y0, Y1), line(False, Y0, X0, X1), line(True, X1, Y0, Y1), line(False, Y1, X0, X1))


def _corner(x0, y0, x1, y1, w, h) -> str:
    """What part of the frame the box is on: its corner, its edge, or the middle."""
    near = EDGE_FLUSH
    left, right = x0 <= w * near, x1 >= w * (1 - near)
    top, bottom = y0 <= h * near, y1 >= h * (1 - near)
    if left and top:
        return "top-left"
    if right and top:
        return "top-right"
    if left and bottom:
        return "bottom-left"
    if right and bottom:
        return "bottom-right"
    if left or right:
        return "right" if right else "left"
    if top or bottom:
        return "bottom" if bottom else "top"
    return "middle"


def _finish(full, found: dict, w: int, h: int, still: np.ndarray, sure: bool) -> dict | None:
    """The box in source pixels, with how sure it is that this is really a camera.

    `sure` is whether something says a person is in it: a person moving in it, or the AI saying so."""
    snapped = _snap(full, (found["x"], found["y"], found["x1"], found["y1"]))
    if not snapped:
        return None
    x0, y0, x1, y1 = snapped
    x0, y0 = max(0.0, x0), max(0.0, y0)
    x1, y1 = min(float(w), x1), min(float(h), y1)
    if x1 - x0 < w * MIN_SIZE or y1 - y0 < h * MIN_SIZE:
        return None
    inside = still[int(found["y"]):int(found["y1"]), int(found["x"]):int(found["x1"])]
    game = float(np.median(still))
    moving = float(np.median(inside)) if inside.size else 99.0
    # sure of it when there is a person in it, and the picture hardly moves while the game around it races
    conf = "high" if sure and moving <= max(3.0, 0.25 * game) else "low"
    return {"x": int(round(x0)), "y": int(round(y0)), "w": int(round(x1 - x0)), "h": int(round(y1 - y0)),
            "corner": _corner(x0, y0, x1, y1, w, h), "score": found.get("score", 0.0), "conf": conf}


def _frame_jpeg(path: str | Path, width: int = 768) -> bytes:
    """One frame of the stream, small, as a JPEG: what a model is shown."""
    try:
        return subprocess.run(
            ["ffmpeg", "-v", "error", "-ss", f"{max(0.0, _duration(path) / 2.0):.1f}", "-i", str(path),
             "-frames:v", "1", "-vf", f"scale={width}:-2", "-pix_fmt", "yuvj420p", "-q:v", "4",
             "-f", "image2pipe", "-vcodec", "mjpeg", "-"],
            capture_output=True, timeout=120).stdout or b""
    except Exception:  # noqa: BLE001
        return b""


def _asked(path: str | Path, ask):
    """One frame to the model: (did it look, its box as parts of the frame or None)."""
    jpeg = _frame_jpeg(path)
    if not jpeg:
        return False, None
    try:
        answer = ask(jpeg) or {}
    except Exception:  # noqa: BLE001 - a model that cannot look is not a failure
        return False, None
    return bool(answer.get("looked")), answer.get("box")


def _seen(full, box, lines: _Lines, w: int, h: int, still: np.ndarray, light: np.ndarray) -> dict | None:
    """A box the model proposed, put onto the picture's real borders and checked like any other.

    A model reads a picture well but places a box roughly, and it can be wrong about the whole thing. So
    each of its sides is moved onto the still line near it that runs along that side, and the box is only
    believed where those lines are really there and the picture inside really is still."""
    fx, fy, fw, fh = box
    x0, y0, x1, y1 = fx * W, fy * H, (fx + fw) * W, (fy + fh) * H
    span = 0.12

    def near(vertical, approx, a, b):
        extent = W if vertical else H
        if approx <= EDGE_FLUSH * extent * 2:
            return 0, 1.0
        if approx >= extent * (1 - 2 * EDGE_FLUSH):
            return extent, 1.0
        lo = max(1, int(approx - span * extent))
        hi = min(extent - 1, int(approx + span * extent))
        best = max(range(lo, hi + 1), key=lambda p: (lines.side(vertical, p, int(a), int(b))[0], -abs(p - approx)))
        return best, lines.side(vertical, best, int(a), int(b))[0]

    (x0, cl), (x1, cr) = near(True, x0, y0, y1), near(True, x1, y0, y1)
    (y0, ct), (y1, cb) = near(False, y0, x0, x1), near(False, y1, x0, x1)
    if min(cl, cr, ct, cb) < 0.5:
        return None                    # its sides are not on lines that are in the picture
    # The model has already said a person is here, so the picture inside is allowed to move more than a
    # measured box would be; what it may not be is the game itself.
    if _plausible(x0, y0, x1, y1, still, light, float(np.median(still)), share=1.0) is None:
        return None
    found = {"x": x0, "y": y0, "x1": x1, "y1": y1, "score": round(min(cl, cr, ct, cb), 2)}
    return _finish(full, found, w, h, still, sure=True)


def detect(path: str | Path, ask=None) -> dict | None:
    """The facecam's box in the source, in source pixels, or None.

    The box is given as x, y, width, height, with the confidence in "score" and
    how sure we are that it is a camera in "conf". Only "high" should be used for a
    clip: "low" is a box that is there but may be a panel of the game.

    `ask` is shown one frame of the stream and says where the webcam is (brain.look_for_facecam). Its
    answer is measured again on the picture before it is believed, so a model that is wrong costs the
    search an idea, not a broken clip. Without `ask` nothing outside the file is looked at, which is how
    the tests run and how a job with no AI set still works."""
    frames = _frames(path)
    if len(frames) < 4:
        return None
    w, h = _size(path)
    if not w or not h:
        return None
    still = _stillness(frames)
    light = frames.astype(np.float32).mean(axis=0)
    looked, hinted = (False, None)
    if ask is not None:
        looked, hinted = _asked(path, ask)
        if looked and not hinted:
            return None                    # it looked, and there is no webcam on this stream
    lines = _Lines(frames, still)
    found = _boxes(lines, still, light)
    full = _frames(path, FULL_FRAMES, size=None) if (found or (looked and hinted)) else None
    if looked and hinted:
        # The model is better than any measurement at saying "that is a person", and worse at saying
        # exactly where the lines are. So it picks which of the measured boxes it is, and only if it
        # matches none of them is its own box measured and used.
        match = max(found, key=lambda c: _overlap(c, hinted)) if found else None
        if match and _overlap(match, hinted) >= MATCH:
            return _finish(full, match, w, h, still, sure=True)
        seen = _seen(full, hinted, lines, w, h, still, light)
        if seen:
            return seen
    if not found:
        return None
    best = found[0]
    return _finish(full, best, w, h, still, sure=best["alive"] >= LIFE)
