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
           "gameplay": "Streamer on top, game below"}


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


SPLIT_H = 576          # the top part of the gameplay layout: the streamer's webcam, the whole width (30 % of 1920)
GAME_H = H - SPLIT_H   # the bottom part: the game (1344 pixels)
FACE_KEEP = 0.25       # a webcam taller than the top part is cut mostly from its bottom: the face is in its upper part
LINE_W = 6             # the line between the webcam and the game
BADGE_H = 84           # the channel's black pill, centred on that line
BADGE_BOX = 100        # the picture with the line and the pill, laid over the clip with the line in its middle
BADGE_MAX = 32         # the longest channel text, in letters
PLATFORM_COLOURS = {"kick": "#53FC18", "twitch": "#9146FF", "youtube": "#FF0033"}
CAP_HALF = 130         # half the height of a line of captions, with room for the word that pops up bigger
CAP_LOW = 1640         # the lowest the captions go: below this the phone apps put their own text
SAFE_TOP = 120         # the strip at the very top that the phone apps cover with their own buttons
HOOK_GAP = 22          # the space between the channel's pill and the heading under it
NO_WEBCAM = "No webcam was found in this stream, so the whole game is shown with a blurred fill instead."


def badge_colour(platform) -> str:
    """The colour of the line and the channel's name: the platform's own, white for anything else."""
    return PLATFORM_COLOURS.get(platform or "", "#FFFFFF")


def clean_label(text) -> str:
    """The channel text a person typed, made safe to keep: one line, no control characters, not too long."""
    return re.sub(r"\s+", " ", re.sub(r"[\x00-\x1f\x7f]+", " ", str(text or ""))).strip()[:BADGE_MAX].strip()


def _badge_clean(text) -> str:
    """Text for the channel's pill: as above, in capitals."""
    return clean_label(text).upper()


def _handle(v) -> str:
    return re.sub(r"[^\w.\-]", "", str(v or ""))


def channel_badge(url="", label="", show=True, name="") -> tuple:
    """The channel's name as it is written on the video, and its platform ("kick", "twitch", "youtube" or "").

    Written by the user (`label`): used as it is, in capitals. Otherwise made from the channel's link:
    kick.com/name, twitch.tv/name, youtube.com/@handle. A link that does not name the channel (a past broadcast,
    a YouTube video or a /channel/ link) uses `name`, the uploader's or creator's name. Nothing for other sites."""
    from urllib.parse import urlparse
    from .creators import platform_key
    url = str(url or "").strip()
    platform = platform_key(url) if url else ""
    if not show:
        return "", platform
    if _badge_clean(label):
        return _badge_clean(label), platform
    parts = [p for p in urlparse(url if "://" in url else "https://" + url).path.split("/") if p] if url else []
    if platform in ("kick", "twitch"):
        site = "KICK.COM" if platform == "kick" else "TWITCH.TV"
        not_names = ("video", "videos", "categories", "directory", "p", "clip", "clips")
        who = _handle(parts[0] if parts and parts[0].lower() not in not_names else name)
        return _badge_clean(f"{site}/{who}" if who else site), platform
    if platform == "youtube":
        at = next((p[1:] for p in parts if p.startswith("@")), "")
        if not at and len(parts) > 1 and parts[0] in ("c", "user"):
            at = parts[1]
        at = _handle(at or name)
        return _badge_clean(f"YOUTUBE.COM/@{at}" if at else "YOUTUBE.COM"), platform
    return "", platform


def channel_badge_text(url="", label="", show=True, name="") -> str:
    return channel_badge(url, label, show, name)[0]


def _logo(draw, platform, x, y, s, colour):
    """The platform's logo, drawn with simple shapes, `s` pixels tall, its top-left corner at x, y."""
    if platform == "kick":       # the blocky K
        u = s / 5
        for r, cs in enumerate(((0, 1, 3, 4), (0, 1, 2, 3), (0, 1, 2), (0, 1, 2, 3), (0, 1, 3, 4))):
            for c in cs:
                draw.rectangle([x + c * u, y + r * u, x + (c + 1) * u - 1, y + (r + 1) * u - 1], fill=colour)
    elif platform == "twitch":   # the speech bubble with two eyes
        w = s * 0.86
        pts = [(0.12, 0), (1, 0), (1, 0.6), (0.76, 0.84), (0.56, 0.84), (0.36, 1.0), (0.36, 0.84), (0.1, 0.84), (0.1, 0.18)]
        draw.polygon([(x + px * w, y + py * s) for px, py in pts], fill=colour)
        for ex in (0.44, 0.7):
            draw.rectangle([x + ex * w, y + 0.2 * s, x + (ex + 0.1) * w, y + 0.5 * s], fill="#000000")
    elif platform == "youtube":  # the play button
        w, h = s * 1.4, s * 0.98
        draw.rounded_rectangle([x, y, x + w, y + h], radius=h * 0.28, fill=colour)
        draw.polygon([(x + w * 0.4, y + h * 0.28), (x + w * 0.4, y + h * 0.72), (x + w * 0.68, y + h * 0.5)], fill="#FFFFFF")


LOGO_W = {"kick": 1.0, "twitch": 0.86, "youtube": 1.4}   # each logo's width, for its height


def make_badge(path, text, platform):
    """The picture laid over the clip at the edge between webcam and game: a line right across in the platform's
    colour and, when there is a text, the channel's name in a black pill in the middle of it, with the logo."""
    from PIL import Image, ImageDraw, ImageFont
    colour = badge_colour(platform)
    img = Image.new("RGBA", (W, BADGE_BOX), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    mid = BADGE_BOX // 2
    d.rectangle([0, mid - LINE_W // 2, W - 1, mid - LINE_W // 2 + LINE_W - 1], fill=colour)
    if text:
        font_path = Path(__file__).resolve().parent / "fonts" / FONT_FILES["Poppins ExtraBold"]
        size, logo_h, pad, gap = 50, 44, 30, 14
        while True:                         # a long name gets smaller letters, never wider than the screen
            font = ImageFont.truetype(str(font_path), size)
            lw = logo_h * LOGO_W.get(platform, 0)
            bw = pad * 2 + font.getlength(text) + (lw + gap if lw else 0)
            if bw <= W - 60 or size <= 24:
                break
            size, logo_h = size - 2, max(24, logo_h - 2)
        x0, y0 = (W - bw) / 2, mid - BADGE_H / 2
        d.rounded_rectangle([x0, y0, x0 + bw, y0 + BADGE_H], radius=BADGE_H / 2, fill="#000000", outline=colour, width=3)
        x = x0 + pad
        if lw:
            _logo(d, platform, x, mid - logo_h / 2, logo_h, colour)
            x += lw + gap
        d.text((x, mid), text, font=font, fill=colour, anchor="lm")
    img.save(path)


def gameplay_plan(src_w, src_h, facecam):
    """Where everything comes from in the gameplay layout, or None when the webcam box cannot be used.

    "box" is the webcam box in the stream, trimmed to the stream's edges.
    "cam" is the part of the webcam that fills the top of the clip (x, y, width, height), in the shape of that
    top part (1080x576). A webcam taller than that keeps its whole width and its upper part, where the face
    is; a wider one keeps its whole height and its middle.
    "game" is the upright slice of the stream that fills the bottom of the clip (x, y, width, height).

    Which slice of the game is shown: the middle, as that is where games put the action. When the webcam is
    inside that middle part it would be seen twice (on top, and again in the game), so the slice moves just far
    enough to one side to leave it out, to whichever side keeps it closest to the middle. When the webcam is
    too wide to leave out on either side, the slice that shows the least of it is used."""
    if not facecam or len(facecam) != 4 or not src_w or not src_h:
        return None
    fx, fy, fw, fh = (int(round(float(v))) for v in facecam)
    fx, fy = max(0, min(fx, src_w - 1)), max(0, min(fy, src_h - 1))
    fw, fh = min(fw, src_w - fx), min(fh, src_h - fy)
    if fw <= 8 or fh <= 8:
        return None
    if fw / fh < W / SPLIT_H:                 # taller than the top part: the whole width, the upper part
        ch = max(2, int(round(fw * SPLIT_H / W)))
        cam = (fx, fy + int(round((fh - ch) * FACE_KEEP)), fw, ch)
    else:                                      # wider: the whole height, the middle
        cw = max(2, int(round(fh * W / SPLIT_H)))
        cam = (fx + (fw - cw) // 2, fy, cw, fh)
    gw = min(src_w, int(src_h * W / GAME_H) // 2 * 2)
    middle = (src_w - gw) // 2

    def seen(x):                     # how much of the webcam's width a slice starting at x would show
        return max(0, min(x + gw, fx + fw) - max(x, fx))
    starts = {max(0, min(x, src_w - gw)) for x in (middle, fx - gw, fx + fw)}
    gx = min(starts, key=lambda x: (seen(x), abs(x - middle)))
    return {"box": (fx, fy, fw, fh), "cam": cam, "game": (gx, 0, gw, src_h)}


def _filter(layout, fg_h, facecam=None, src=None, badge=False):
    # The picture people look at is enlarged with the "lanczos" filter, which keeps edges and text sharper than
    # the default when a small stream is blown up to a phone screen. The blurred background does not need it.
    plan = gameplay_plan(*src, facecam) if layout == "gameplay" and src else None
    if plan:
        # The webcam on top, the whole width; the game under it; then the line with the channel's name (the
        # second input, a picture) laid over the edge between them.
        (cx, cy, cw, ch), (gx, gy, gw, gh) = plan["cam"], plan["game"]
        fit = "force_original_aspect_ratio=increase:flags=lanczos"
        out = (f"[0:v]fps={config.OUT_FPS},split=2[g][c];"
               f"[c]crop={cw}:{ch}:{cx}:{cy},scale={W}:{SPLIT_H}:{fit},crop={W}:{SPLIT_H},setsar=1[top];"
               f"[g]crop={gw}:{gh}:{gx}:{gy},scale={W}:{GAME_H}:{fit},crop={W}:{GAME_H},setsar=1[bot];"
               f"[top][bot]vstack=inputs=2")
        return out + (f"[st];[st][1:v]overlay=0:{SPLIT_H - BADGE_BOX // 2}[v0];" if badge else "[v0];")
    if layout in ("crop", "gameplay"):
        return f"[0:v]fps={config.OUT_FPS},scale={W}:{H}:force_original_aspect_ratio=increase:flags=lanczos,crop={W}:{H},setsar=1[v0];"
    return (f"[0:v]fps={config.OUT_FPS},split=2[a][b];"
            f"[a]scale=270:480:force_original_aspect_ratio=increase,crop=270:480,boxblur=6:2,"
            f"scale={W}:{H},eq=brightness=-0.14:saturation=1.15[bg];"
            f"[b]scale={W}:{fg_h}:force_original_aspect_ratio=increase:flags=lanczos,crop={W}:{fg_h}[fg];"
            f"[bg][fg]overlay=0:(H-h)/2,setsar=1[v0];")


def _hook_y(layout, hook_y, facecam=None, src=None, hook=""):
    """Where the middle of the heading goes. In the gameplay layout: at the top of the game, just under the
    channel's pill, so it covers neither the pill nor the streamer's face."""
    if layout == "gameplay" and src and gameplay_plan(*src, facecam):
        shape = _hook_shape(hook)
        bh = shape[4] if shape else 0
        return int(round(SPLIT_H + BADGE_H / 2 + HOOK_GAP + bh / 2))
    return hook_y


def _caption_y(layout, cap_y, facecam=None, src=None):
    """Where the middle of the captions goes. In the gameplay layout: low in the game, as low as they can go
    above the strip the phone apps cover with their own text."""
    if layout == "gameplay" and src and gameplay_plan(*src, facecam):
        return CAP_LOW - CAP_HALF
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


def render_clip(src: Path, out_dir: Path, name: str, clip: dict, words: list, src_w: int, src_h: int, cover_at=None) -> dict:
    """Writes <name>.mp4, <name>.jpg and <name>.ass into out_dir. Returns file names.
    cover_at is the second of the clip (from its start) the cover picture is taken at: the punchline, when known."""
    out_dir.mkdir(parents=True, exist_ok=True)
    start, end = clip["start"], clip["end"]
    layout, fg_h, hook_y, cap_y = geometry(src_w, src_h, clip.get("layout", "blur"))
    face, note = clip.get("facecam"), None
    if layout == "gameplay" and not gameplay_plan(src_w, src_h, face):
        # No webcam to put in the corner. Cutting the middle out of the game would lose most of it with
        # nothing to show for it, so the whole game is shown with a blurred fill, and the clip says why.
        layout, fg_h, hook_y, cap_y = geometry(src_w, src_h, "blur")
        face, note = None, NO_WEBCAM
    hook = clip.get("hook", "")
    hook_y = _hook_y(layout, hook_y, face, (src_w, src_h), _clean(hook))
    cap_y = _caption_y(layout, cap_y, face, (src_w, src_h))
    ass = build_ass(words if clip.get("captions", True) else [], start, end, hook, clip.get("style", "bounce"),
                    clip.get("highlight_words", []), hook_y, cap_y)
    badge = None
    if layout == "gameplay":    # the channel's name: the clip's own text, else the one made from its channel
        text = ((_badge_clean(clip.get("channel_label")) or _badge_clean(clip.get("channel_auto")))
                if clip.get("show_channel", True) else "")
        badge = (text, clip.get("channel_platform") or "")
    # The clip is made in a short work folder and moved into place afterwards. The caption step of ffmpeg cannot
    # open files whose full path is longer than 260 characters on Windows, which a data folder deep inside other
    # folders easily reaches. Inside the work folder every name is short and relative (a drive colon would also
    # break the filter syntax).
    work = Path(tempfile.mkdtemp(prefix="pk"))
    try:
        out = _render_in(work, src, out_dir, name, start, end, layout, fg_h, ass, face=face, src_size=(src_w, src_h),
                         badge=badge, cover_at=cover_at)
        out["note"] = note     # also clears an old note once the webcam is found
        return out
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _render_in(work: Path, src: Path, out_dir: Path, name: str, start, end, layout, fg_h, ass: str, face=None, src_size=None, badge=None, cover_at=None) -> dict:
    global _qsv_ok
    (work / f"{name}.ass").write_text(ass, encoding="utf-8")
    (work / "f").mkdir()
    for f in config.FONTS_DIR.glob("*.ttf"):
        shutil.copyfile(f, work / "f" / f.name)
    extra = []
    if badge is not None:       # the line and the channel's name, a picture laid over the clip
        make_badge(work / "badge.png", *badge)
        extra = ["-i", "badge.png"]
    fc = _filter(layout, fg_h, face, src_size, badge is not None) + f"[v0]ass={name}.ass:fontsdir=f[v]"
    base = ["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.2f}", "-t", f"{end - start:.2f}", "-i", str(src), *extra,
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
    if cover_at is not None and 0 <= cover_at < end - start - 0.1:
        cover_t = cover_at
    run(["ffmpeg", "-y", "-v", "error", "-ss", f"{cover_t:.2f}", "-i", f"{name}.mp4", "-frames:v", "1",
         "-q:v", "3", f"{name}.jpg"], cwd=work)
    for ext in ("mp4", "jpg", "ass"):
        shutil.move(str(work / f"{name}.{ext}"), str(out_dir / f"{name}.{ext}"))
        disk.flush(out_dir / f"{name}.{ext}")   # a clip finished just before the computer switches off is kept whole
    return {"file": f"{name}.mp4", "cover": f"{name}.jpg", "layout": layout}
