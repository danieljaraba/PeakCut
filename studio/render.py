"""Turns a time range into a vertical clip: 9:16 layout, bouncy word captions, hook title, cover."""
import json
import math
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from . import config, disk, roman
from .media import NO_WINDOW, run

W, H = config.OUT_W, config.OUT_H

# ASS colours are &HBBGGRR. `size` is the ASS font size (line height); _Font turns it into real pixels.
STYLES = {
    "bounce": {"label": "Bounce", "font": "Poppins ExtraBold", "size": 132, "base": "FFFFFF", "active": "00F0FF",
               "strong": "FFFFFF", "outline": 7, "pop": 112, "upper": True, "per": 3},
    "hype": {"label": "Hype", "font": "Anton", "size": 152, "base": "FFFFFF", "active": "3C3CFF",
             "strong": "00D7FF", "outline": 7, "pop": 114, "upper": True, "per": 2},
    "clean": {"label": "Clean", "font": "Poppins ExtraBold", "size": 100, "base": "FFFFFF", "active": "6BD7FF",
              "strong": "FFFFFF", "outline": 5, "pop": 106, "upper": False, "per": 4},
}
HOOK = {"font": "Poppins ExtraBold", "size": 96, "ink": "14100E", "box": "FFFFFF", "pad_x": 36, "pad_y": 22, "radius": 26,
        "max_w": 900}
FONT_FILES = {"Poppins ExtraBold": "Poppins-ExtraBold.ttf", "Anton": "Anton-Regular.ttf", "Archivo Black": "ArchivoBlack-Regular.ttf"}
CAP_MAX_W = 960   # captions never run wider than this, whatever is said


class _Font:
    """Measures text exactly as the renderer will draw it, so every word can be placed by hand: the highlighted
    word can grow without pushing its neighbours, and the heading box fits its text."""
    _cache: dict = {}

    def __init__(self, name):
        import struct
        path = Path(__file__).resolve().parent / "fonts" / FONT_FILES.get(name, FONT_FILES["Poppins ExtraBold"])
        d = path.read_bytes()
        tables = {}
        for i in range(struct.unpack(">H", d[4:6])[0]):
            tag, _, off, _ = struct.unpack(">4sIII", d[12 + 16 * i: 28 + 16 * i])
            tables[tag] = off
        self.upm = struct.unpack(">H", d[tables[b"head"] + 18: tables[b"head"] + 20])[0]
        o = tables[b"OS/2"]
        self.asc, self.desc = struct.unpack(">HH", d[o + 74: o + 78])      # what the renderer sizes the font by
        self.cap = struct.unpack(">h", d[o + 88: o + 90])[0] or int(self.upm * 0.7)
        self.path = str(path)
        self._pil = {}

    @classmethod
    def get(cls, name):
        if name not in cls._cache:
            cls._cache[name] = cls(name)
        return cls._cache[name]

    def em(self, size):
        return size * self.upm / (self.asc + self.desc)

    def width(self, text, size):
        """Pixels wide at this ASS size."""
        try:
            from PIL import ImageFont
            if "f" not in self._pil:
                self._pil["f"] = ImageFont.truetype(self.path, 200)
            return self._pil["f"].getlength(text) * self.em(size) / 200
        except Exception:  # noqa: BLE001 - no measuring library: a fair estimate keeps clips rendering
            return len(text) * self.em(size) * 0.62

    def cap_shift(self, size):
        """How far below the anchor the middle of a capital letter sits when text is centred on a point."""
        return ((self.asc - self.desc) / 2 - self.cap / 2) * size / (self.asc + self.desc)


CAPTION_LEAD = 0.0  # measured: word stamps are within 0.15 s of the voice, so no shift is needed
def install_fonts():
    """Copy the bundled caption fonts next to the data, so captions look the same on every machine and server."""
    src = Path(__file__).resolve().parent / "fonts"
    for f in src.glob("*.ttf"):
        if not (config.FONTS_DIR / f.name).exists():
            shutil.copyfile(f, config.FONTS_DIR / f.name)


LAYOUTS = {"blur": "Full frame + blurred fill", "zoom": "Zoomed in + blurred fill", "crop": "Fill the screen (centre crop)",
           "gameplay": "Gameplay fills the screen, facecam in the corner"}


def _ts(t):
    t = max(0.0, t)
    h, m = int(t // 3600), int(t % 3600 // 60)
    return f"{h}:{m:02d}:{t % 60:05.2f}"


def _clean(text):
    return re.sub(r"[\x00-\x1f\x7f]+", " ", re.sub(r"[{}\\]", "", str(text))).strip()


def _key(word):
    return re.sub(r"[^a-z0-9']", "", word.lower())


def _groups(words, per):
    out, cur = [], []
    for w in words:
        if cur and (len(cur) >= per or sum(len(x["w"]) for x in cur) + len(w["w"]) > 16
                    or w["s"] - cur[-1]["e"] > 0.45 or cur[-1]["w"][-1:] in ".?!"):
            out.append(cur)
            cur = []
        cur.append(w)
    if cur:
        out.append(cur)
    return out


def geometry(src_w, src_h, layout):
    """Where the video sits in the 1080x1920 frame, and where hook and captions go."""
    if src_h >= src_w and layout != "crop":
        layout = "crop"  # already vertical or square: nothing to gain from a blurred fill
    if layout == "blur":
        fg_h = min(H, int(round(W * src_h / src_w / 2)) * 2)
    elif layout == "zoom":
        fg_h = 1000
    else:
        fg_h = H
    top = (H - fg_h) // 2
    if layout in ("crop", "gameplay"):
        return layout, fg_h, 300, 1380
    return layout, fg_h, max(230, top - 190), min(1640, top + fg_h + 180)


def _hook_shape(hook):
    """How the heading is laid out: its lines, letter size, line height, and the width and height of its box.
    None when there is no heading."""
    f, size = _Font.get(HOOK["font"]), HOOK["size"]
    words = hook.split()
    if not words:
        return None
    lines = [" ".join(words)]
    if f.width(lines[0], size) > HOOK["max_w"] and len(words) > 1:      # two lines, split where they come out most even
        cut = min(range(1, len(words)), key=lambda i: max(f.width(" ".join(words[:i]), size), f.width(" ".join(words[i:]), size)))
        lines = [" ".join(words[:cut]), " ".join(words[cut:])]
    widest = max(f.width(l, size) for l in lines)
    if widest > HOOK["max_w"]:
        size = max(40, int(size * HOOK["max_w"] / widest))
        widest = max(f.width(l, size) for l in lines)
    line_h = f.em(size) * 1.16
    return lines, size, line_h, widest + 2 * HOOK["pad_x"], line_h * len(lines) + 2 * HOOK["pad_y"]


def _hook_events(hook, dur, y):
    """The heading: dark text in a white rounded box that fits it, on one line or two balanced ones."""
    shape = _hook_shape(hook)
    if not shape:
        return []
    f = _Font.get(HOOK["font"])
    lines, size, line_h, bw, bh = shape
    x0, y0, r = (W - bw) / 2, y - bh / 2, min(HOOK["radius"], bh / 2)
    x1, y1 = x0 + bw, y0 + bh
    n = lambda v: f"{v:.0f}"                                                         # noqa: E731
    box = (f"m {n(x0 + r)} {n(y0)} l {n(x1 - r)} {n(y0)} b {n(x1)} {n(y0)} {n(x1)} {n(y0)} {n(x1)} {n(y0 + r)} "
           f"l {n(x1)} {n(y1 - r)} b {n(x1)} {n(y1)} {n(x1)} {n(y1)} {n(x1 - r)} {n(y1)} "
           f"l {n(x0 + r)} {n(y1)} b {n(x0)} {n(y1)} {n(x0)} {n(y1)} {n(x0)} {n(y1 - r)} "
           f"l {n(x0)} {n(y0 + r)} b {n(x0)} {n(y0)} {n(x0)} {n(y0)} {n(x0 + r)} {n(y0)}")
    span = f"{_ts(0)},{_ts(dur)}"
    ev = [f"Dialogue: 3,{span},Hook,,0,0,0,,{{\\an7\\pos(0,0)\\fad(160,0)\\c&H{HOOK['box']}&\\p1}}{box}{{\\p0}}"]
    for i, line in enumerate(lines):
        cy = y - line_h * (len(lines) - 1) / 2 + i * line_h - f.cap_shift(size)
        ev.append(f"Dialogue: 4,{span},Hook,,0,0,0,,{{\\an5\\pos({W // 2},{cy:.0f})\\fad(160,0)\\fs{size}}}{line}")
    return ev


def build_ass(words, start, end, hook, style_name, highlight, hook_y, cap_y) -> str:
    st = STYLES.get(style_name, STYLES["bounce"])
    words = [dict(w, s=w["s"] - CAPTION_LEAD, e=w["e"] - CAPTION_LEAD) for w in words]
    strong = {_key(h) for h in highlight if _key(h)}
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{st['font']},{st['size']},&H00{st['base']},&H00{st['base']},&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,{st['outline']},0,5,0,0,0,1
Style: Hook,{HOOK['font']},{HOOK['size']},&H00{HOOK['ink']},&H00{HOOK['ink']},&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,5,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    dur = end - start
    ev = _hook_events(_clean(hook), dur, hook_y)
    f = _Font.get(st["font"])
    inside = [dict(w, w=_clean(w["w"])) for w in words if w["s"] >= start - 0.05 and w["s"] < end]
    inside = sorted((w for w in inside if w["w"]), key=lambda w: w["s"])
    for i, w in enumerate(inside[:-1]):
        w["e"] = max(w["s"], min(w["e"], inside[i + 1]["s"]))
    groups = _groups(inside, st["per"])
    for gi, g in enumerate(groups):
        g_start = max(0.0, g[0]["s"] - start)
        g_end = min(g[-1]["e"] + 0.25, groups[gi + 1][0]["s"] if gi + 1 < len(groups) else end) - start
        if g_end - g_start < 0.02:
            continue
        texts = []
        for x in g:
            t = re.sub(r"[.,;:]+$", "", x["w"])
            texts.append(t.upper() if st["upper"] else t)
        if not all(roman.drawable(t) for t in texts):
            # A script the caption font cannot draw (Japanese, Thai, Chinese...): its words cannot be measured, so
            # they are not placed one by one. The line is drawn whole and centred, in a font the system has for it.
            line = " ".join(texts)
            fs = max(44, min(int(st["size"] * 0.7), int(1500 / max(1, len(line)) * 1.6)))
            ev.append(f"Dialogue: 1,{_ts(g_start)},{_ts(g_end)},Cap,,60,60,0,,{{\\an5\\pos({W // 2},{cap_y:.0f})\\q0\\fs{fs}}}{line}")
            continue
        size = st["size"]
        gap = f.width("n n", size) - 2 * f.width("n", size) + (2 * st["outline"] + 6) * size / st["size"]   # a space, plus room for both outlines
        total = sum(f.width(t, size) for t in texts) + gap * (len(texts) - 1)
        if total > CAP_MAX_W:                                  # a long phrase: shrink it to fit, never run off the frame
            size = max(36, int(size * CAP_MAX_W / total))
            gap = f.width("n n", size) - 2 * f.width("n", size) + (2 * st["outline"] + 6) * size / st["size"]   # a space, plus room for both outlines
            total = sum(f.width(t, size) for t in texts) + gap * (len(texts) - 1)
        cy = cap_y - f.cap_shift(size)
        x = (W - total) / 2
        p = st["pop"]
        for i, (w, t) in enumerate(zip(g, texts)):
            wd = f.width(t, size)
            p = max(104, min(st["pop"], 100 + int(2200 / max(wd, 1))))   # long words grow less, so they never touch their neighbours
            pos = f"\\an5\\pos({x + wd / 2:.0f},{cy:.0f})\\fs{size}"
            x += wd + gap
            a = max(g_start, w["s"] - start)
            b = (g[i + 1]["s"] - start) if i + 1 < len(g) else g_end
            base = st["strong"] if _key(w["w"]) in strong else st["base"]
            if a - g_start >= 0.02:                                                   # waiting to be said
                ev.append(f"Dialogue: 1,{_ts(g_start)},{_ts(a)},Cap,,0,0,0,,{{{pos}\\c&H{base}&}}{t}")
            if b - a >= 0.02:                                                         # being said: colour and a quick pop, in front
                ev.append(f"Dialogue: 2,{_ts(a)},{_ts(b)},Cap,,0,0,0,,{{{pos}\\c&H{st['active']}&\\fscx94\\fscy94"
                          f"\\t(0,90,\\fscx{p}\\fscy{p})\\t(90,200,\\fscx{p - 5}\\fscy{p - 5})}}{t}")
            if g_end - b >= 0.02:                                                     # said
                ev.append(f"Dialogue: 1,{_ts(b)},{_ts(g_end)},Cap,,0,0,0,,{{{pos}\\c&H{base}&\\fscx{p - 5}\\fscy{p - 5}"
                          f"\\t(0,90,\\fscx100\\fscy100)}}{t}")
    return head + "\n".join(ev) + "\n"


PIP_W = 380            # how wide the webcam picture is in the 1080-wide frame
PIP_MAX_H = 520        # and the tallest it may be: a webcam turned upright would otherwise cover a third of the screen
PIP_MARGIN = 26        # its distance from the side of the frame
CAM_SIDE = "right"     # which side of the clip the webcam picture sits on, unless the creator or clip says otherwise
CAM_TOP = 20           # how far down its top edge sits, as a percent of the height (20 % of 1920 is 384 pixels):
                       # low enough to be seen with the face in it, not hidden up in the corner
CAM_TOP_RANGE = (3, 60)  # the highest and lowest it may be set to
CAP_HALF = 130         # half the height of a line of captions, with room for the word that pops up bigger
CAP_GAP = 40           # the space kept between the webcam picture and the captions
CAP_LOW = 1640         # the lowest the captions go: below this the phone apps put their own text
SAFE_TOP = 120         # the strip at the very top that the phone apps cover with their own buttons
HOOK_GAP = 24          # the space between the webcam picture and the heading
NO_WEBCAM = "No webcam was found in this stream, so the whole game is shown with a blurred fill instead."


def cam_place(side=None, top=None):
    """The webcam position asked for, made safe: a side ("right" or "left") and a whole percent inside the range."""
    side = side if side in ("left", "right") else CAM_SIDE
    try:
        top = int(round(float(top)))
    except (TypeError, ValueError):
        top = CAM_TOP
    return side, max(CAM_TOP_RANGE[0], min(CAM_TOP_RANGE[1], top))


def gameplay_plan(src_w, src_h, facecam, side=None, top=None):
    """Where everything goes in the gameplay layout, or None when the webcam box cannot be used.

    "cam" is the webcam box in the stream, trimmed to the stream's edges.
    "game" is the upright slice of the stream that fills the phone screen behind it (x, y, width, height).
    "pip" is where the webcam picture sits in the 1080x1920 clip (x, y, width, height): against the chosen side,
    with its top edge `top` percent of the way down. A tall picture set very low is lifted just enough to leave
    room for the captions under it.

    Which slice of the game is shown: a phone screen is much narrower than a stream, so only about a third
    of its width fits. The middle is used, as that is where games put the action. When the webcam is inside
    that middle part, it would be seen twice (once in the corner, once behind it), so the slice moves just far
    enough to one side to leave the webcam out, to whichever side keeps it closest to the middle. When the
    webcam is too wide to leave out on either side, the slice that shows the least of it is used."""
    if not facecam or len(facecam) != 4 or not src_w or not src_h:
        return None
    fx, fy, fw, fh = (int(round(float(v))) for v in facecam)
    fx, fy = max(0, min(fx, src_w - 1)), max(0, min(fy, src_h - 1))
    fw, fh = min(fw, src_w - fx), min(fh, src_h - fy)
    if fw <= 8 or fh <= 8:
        return None
    gw = min(src_w, int(src_h * W / H) // 2 * 2)
    middle = (src_w - gw) // 2

    def seen(x):                     # how much of the webcam's width a slice starting at x would show
        return max(0, min(x + gw, fx + fw) - max(x, fx))
    starts = {max(0, min(x, src_w - gw)) for x in (middle, fx - gw, fx + fw)}
    gx = min(starts, key=lambda x: (seen(x), abs(x - middle)))
    pw, ph = PIP_W, PIP_W * fh / fw
    if ph > PIP_MAX_H:
        pw, ph = PIP_MAX_H * fw / fh, PIP_MAX_H
    pw, ph = max(2, int(round(pw / 2)) * 2), max(2, int(round(ph / 2)) * 2)
    side, top = cam_place(side, top)
    py = max(0, min(int(H * top / 100), CAP_LOW - CAP_HALF - CAP_GAP - ph))
    return {"cam": (fx, fy, fw, fh), "game": (gx, 0, gw, src_h),
            "pip": (PIP_MARGIN if side == "left" else W - pw - PIP_MARGIN, py, pw, ph)}


def _filter(layout, fg_h, facecam=None, src=None, side=None, top=None):
    # The picture people look at is enlarged with the "lanczos" filter, which keeps edges and text sharper than
    # the default when a small stream is blown up to a phone screen. The blurred background does not need it.
    plan = gameplay_plan(*src, facecam, side, top) if layout == "gameplay" and src else None
    if plan:
        # The game fills the screen, and the webcam sits in the upper part of it, by one side, so a viewer sees
        # the play and the face at the same time.
        (fx, fy, fw, fh), (gx, gy, gw, gh), (px, py, pw, ph) = plan["cam"], plan["game"], plan["pip"]
        return (f"[0:v]fps={config.OUT_FPS},split=2[base][fc];"
                f"[base]crop={gw}:{gh}:{gx}:{gy},scale={W}:{H}:force_original_aspect_ratio=increase:flags=lanczos,"
                f"crop={W}:{H},setsar=1[b];"
                f"[fc]crop={fw}:{fh}:{fx}:{fy},scale={pw}:{ph}:flags=lanczos,setsar=1[p];"
                f"[b][p]overlay={px}:{py}[v0];")
    if layout in ("crop", "gameplay"):
        return f"[0:v]fps={config.OUT_FPS},scale={W}:{H}:force_original_aspect_ratio=increase:flags=lanczos,crop={W}:{H},setsar=1[v0];"
    return (f"[0:v]fps={config.OUT_FPS},split=2[a][b];"
            f"[a]scale=270:480:force_original_aspect_ratio=increase,crop=270:480,boxblur=6:2,"
            f"scale={W}:{H},eq=brightness=-0.14:saturation=1.15[bg];"
            f"[b]scale={W}:{fg_h}:force_original_aspect_ratio=increase:flags=lanczos,crop={W}:{fg_h}[fg];"
            f"[bg][fg]overlay=0:(H-h)/2,setsar=1[v0];")


def _hook_y(layout, hook_y, facecam=None, src=None, hook="", side=None, top=None):
    """Where the middle of the heading goes. In the gameplay layout the heading is nearly as wide as the screen,
    so it can never sit beside the webcam picture without running under it. It stays in its usual place when
    that is clear of the picture; otherwise it goes just above the picture when its whole box fits below the
    strip the phone apps cover, and otherwise just under the picture. The box's edges are what is kept clear,
    not its middle, so a heading on two lines is kept clear too."""
    plan = gameplay_plan(*src, facecam, side, top) if layout == "gameplay" and src else None
    if plan:
        shape = _hook_shape(hook)
        bh = shape[4] if shape else 0
        _, py, _, ph = plan["pip"]
        if hook_y + bh / 2 + HOOK_GAP <= py or hook_y - bh / 2 - HOOK_GAP >= py + ph:
            return hook_y
        above = py - HOOK_GAP - bh / 2
        if above - bh / 2 >= SAFE_TOP:
            return int(round(above))
        return int(round(py + ph + HOOK_GAP + bh / 2))
    return hook_y


def _caption_y(layout, cap_y, facecam=None, src=None, side=None, top=None):
    """Where the middle of the captions goes. In the gameplay layout, a webcam picture set low enough to reach
    the captions pushes them down to just under it (the picture is never set so low that they do not fit)."""
    plan = gameplay_plan(*src, facecam, side, top) if layout == "gameplay" and src else None
    if plan:
        _, py, _, ph = plan["pip"]
        if py < cap_y + CAP_HALF + CAP_GAP and py + ph + CAP_GAP > cap_y - CAP_HALF:
            return py + ph + CAP_GAP + CAP_HALF
    return cap_y


LOUDNESS = "I=-14:TP=-1.5:LRA=11"      # how loud a finished clip is: the level the short-video apps play at


def _loudnorm(src, start, end) -> str:
    """The sound filter for a clip: every clip comes out about equally loud.

    Done in two passes. First the clip's sound is only listened to, to measure how loud it is; then it is turned
    up or down by that one fixed amount. Done in one pass, the level is guessed while playing and keeps being
    corrected, which you hear as the volume pumping up and down in a short clip. If the measuring fails for any
    reason, the one-pass way is used, as before."""
    plain = f"loudnorm={LOUDNESS}"
    try:
        p = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-ss", f"{start:.2f}", "-t", f"{end - start:.2f}",
                            "-i", str(src), "-map", "0:a:0", "-vn", "-af", f"{plain}:print_format=json", "-f", "null", "-"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace", creationflags=NO_WINDOW,
                           timeout=120)
        found = re.findall(r"\{[^{}]*\}", p.stderr or "")
        if p.returncode != 0 or not found:
            return plain
        m = json.loads(found[-1])
        v = [float(m[k]) for k in ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")]
        if not all(math.isfinite(x) for x in v):      # a silent clip measures as "-inf": nothing to go by
            return plain
    except Exception:  # noqa: BLE001
        return plain
    return (f"{plain}:measured_I={v[0]:.2f}:measured_TP={v[1]:.2f}:measured_LRA={v[2]:.2f}"
            f":measured_thresh={v[3]:.2f}:offset={v[4]:.2f}:linear=true")


_qsv_ok = None
RENDER_SECONDS = 600      # one clip's video encode; far beyond normal, so a frozen encoder is stopped, not waited on


def render_clip(src: Path, out_dir: Path, name: str, clip: dict, words: list, src_w: int, src_h: int) -> dict:
    """Writes <name>.mp4, <name>.jpg and <name>.ass into out_dir. Returns file names."""
    out_dir.mkdir(parents=True, exist_ok=True)
    start, end = clip["start"], clip["end"]
    layout, fg_h, hook_y, cap_y = geometry(src_w, src_h, clip.get("layout", "blur"))
    face, note = clip.get("facecam"), None
    if layout == "gameplay" and not gameplay_plan(src_w, src_h, face):
        # No webcam to put in the corner. Cutting the middle out of the game would lose most of it with
        # nothing to show for it, so the whole game is shown with a blurred fill, and the clip says why.
        layout, fg_h, hook_y, cap_y = geometry(src_w, src_h, "blur")
        face, note = None, NO_WEBCAM
    hook, place = clip.get("hook", ""), cam_place(clip.get("cam_side"), clip.get("cam_top"))
    hook_y = _hook_y(layout, hook_y, face, (src_w, src_h), _clean(hook), *place)
    cap_y = _caption_y(layout, cap_y, face, (src_w, src_h), *place)
    ass = build_ass(words if clip.get("captions", True) else [], start, end, hook, clip.get("style", "bounce"),
                    clip.get("highlight_words", []), hook_y, cap_y)
    # The clip is made in a short work folder and moved into place afterwards. The caption step of ffmpeg cannot
    # open files whose full path is longer than 260 characters on Windows, which a data folder deep inside other
    # folders easily reaches. Inside the work folder every name is short and relative (a drive colon would also
    # break the filter syntax).
    work = Path(tempfile.mkdtemp(prefix="pk"))
    try:
        out = _render_in(work, src, out_dir, name, start, end, layout, fg_h, ass, face=face, src_size=(src_w, src_h),
                         place=place)
        out["note"] = note     # also clears an old note once the webcam is found
        return out
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _render_in(work: Path, src: Path, out_dir: Path, name: str, start, end, layout, fg_h, ass: str, face=None, src_size=None, place=(None, None)) -> dict:
    global _qsv_ok
    (work / f"{name}.ass").write_text(ass, encoding="utf-8")
    (work / "f").mkdir()
    for f in config.FONTS_DIR.glob("*.ttf"):
        shutil.copyfile(f, work / "f" / f.name)
    fc = _filter(layout, fg_h, face, src_size, *place) + f"[v0]ass={name}.ass:fontsdir=f[v]"
    base = ["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.2f}", "-t", f"{end - start:.2f}", "-i", str(src),
            "-filter_complex", fc, "-map", "[v]", "-map", "0:a:0", "-af", _loudnorm(src, start, end),
            "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-movflags", "+faststart", "-pix_fmt"]
    encoders = []
    if _qsv_ok is not False:
        encoders.append(("qsv", ["nv12", "-c:v", "h264_qsv", "-global_quality", "23", "-preset", "fast"]))
    encoders.append(("x264", ["yuv420p", "-c:v", "libx264", "-crf", "21", "-preset", "veryfast"]))
    err = None
    for label, enc in encoders:
        p = subprocess.run(base + enc + [f"{name}.mp4"], cwd=work, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", creationflags=NO_WINDOW, timeout=RENDER_SECONDS)
        if p.returncode == 0:
            if label == "qsv":
                _qsv_ok = True
            break
        err = p.stderr[-1200:]
        if label == "qsv":
            _qsv_ok = False
    else:
        raise RuntimeError(f"Render failed: {err}")
    cover_t = min(1.2, (end - start) / 2)
    run(["ffmpeg", "-y", "-v", "error", "-ss", f"{cover_t:.2f}", "-i", f"{name}.mp4", "-frames:v", "1",
         "-q:v", "3", f"{name}.jpg"], cwd=work)
    for ext in ("mp4", "jpg", "ass"):
        shutil.move(str(work / f"{name}.{ext}"), str(out_dir / f"{name}.{ext}"))
        disk.flush(out_dir / f"{name}.{ext}")   # a clip finished just before the computer switches off is kept whole
    return {"file": f"{name}.mp4", "cover": f"{name}.jpg", "layout": layout}
