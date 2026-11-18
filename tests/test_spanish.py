"""Spanish streams: the language is told from the words, the AI is asked to write in Spanish, and captions are
not translated into English. The AI is not used: only the prompts and the helpers are looked at."""
import os, sys, tempfile
os.environ["PEAKCUT_DATA"] = tempfile.mkdtemp(prefix="pc-es-"); os.environ["DATABASE_URL"] = ""
os.environ.pop("PEAKCUT_LANGUAGE", None)
sys.path.insert(0, ".")
from studio import brain, config, creators
fails = []
def check(n, ok, x=""):
    print(("PASS " if ok else "FAIL ") + n + (f"  {x}" if x and not ok else "")); ok or fails.append(n)

ES = "Hola a todos, qué locura lo que acaba de pasar, no puede ser, es muy gracioso pero tengo que decir que esto es increíble"
EN = "So this is the thing that I was telling you about and it is going to be great for the team, you know what I mean"
check("Spanish is told from the words", brain.detect_lang(ES) == "es", brain.detect_lang(ES))
check("English is told from the words", brain.detect_lang(EN) == "en", brain.detect_lang(EN))
check("too little speech gives no answer", brain.detect_lang("hola qué tal") == "")
check("words shared by both languages give no answer", brain.detect_lang("no no no a a me me he he no no no a me") == "")
check("a list of words works too", brain.detect_lang([{"w": w, "s": 0, "e": 1} for w in ES.split()]) == "es")
check("the explicit choice comes first", brain.resolve_lang("en", ES) == "en" and brain.resolve_lang("es-419", EN) == "es")
check("auto reads the words", brain.resolve_lang("auto", ES) == "es" and brain.resolve_lang("", EN) == "en")
check("another language keeps the old behaviour", brain.resolve_lang("hi", ES) == "")
config.LANGUAGE = "es"
check("the server setting comes before the words", brain.resolve_lang("auto", EN) == "es")
config.LANGUAGE = "auto"

es = brain._rules("es", min_len=15, max_len=60, transcript="x")
en = brain._rules("en", min_len=15, max_len=60, transcript="x")
check("the Spanish prompt asks for Spanish", "natural, spoken Spanish" in es and "in English. If the video is in Hindi" not in es)
check("the English prompt keeps its old line", brain.WRITE_EN in en and "Spanish" not in en and "{write_in}" not in en)
check("the Spanish scouting notes exist", "jajaja" in brain.SPANISH_RULES and brain._lang_rules("en") == "")

seen = []
def fake(prompt, schema=None, purpose="", timeout=None):
    seen.append(prompt); return [], "m", 1
brain.ask_ai = fake
brain.describe(ES, "Stream", "Canal", lang=None)
check("describe writes in Spanish for Spanish speech", "natural, spoken Spanish" in seen[-1])
brain.describe(EN, "Stream", "Canal")
check("...and not for English", "natural, spoken Spanish" not in seen[-1])

check("an accent fix is the same word", brain._same_word("cancion", "canción") and brain._same_word("espanol", "español"))
check("another word is not", not brain._same_word("perro", "gato"))
check("Spanish captions are not translated", not brain.needs_translation("es") and not brain.needs_translation("es-419"))
check("other languages still are", brain.needs_translation("fr") and not brain.needs_translation("hi"))

words = [{"w": w, "s": i * 0.5, "e": i * 0.5 + 0.4} for i, w in enumerate(ES.split())]
brain.fix_caption_words(words, "t", "es")
check("the caption fix tells the AI to keep Spanish", "SPANISH" in seen[-1])
check("the fallback loudness pick is in Spanish", brain.fallback_by_loudness(__import__("numpy").array([1.0] * 5 + [90.0] * 5), words, 1, 10, 20, "es")[0]["title"] == "Momento ruidoso")

check("the creator can choose a language", creators._clean({"language": "es"})["language"] == "es")
check("anything else becomes auto", creators._clean({"language": "klingon"})["language"] == "auto")

from studio import live
check("someone saying clipea is heard", all(live.Session.CLIP_WORDS.search(brain._plain(x)) for x in ("clipea", "clipéalo", "clipealo", "clip")))

# ---- the post text for TikTok, Reels and Shorts in the clip window (needs node; skipped without it)
import json, re, shutil, subprocess
node = shutil.which("node")
if node:
    page = open("web/index.html", encoding="utf-8").read()
    m = re.search(r"const noLinks = .*?\n}\n(?=function openClip)", page, re.S)
    clip = {"title": "No puede ser, ¡qué locura!", "hook": "¿QUÉ ACABA DE PASAR?", "hashtags": ["gaming", "locura", "vamos", "clutch", "fail", "extra"],
            "description": "El streamer no lo cree. Mira https://x.com/a que gracioso. " + "Más texto. " * 30}
    js = m.group(0) + f"console.log(JSON.stringify(postDrafts({json.dumps(clip)}, 'Clipped by @canal')));"
    out = json.loads(subprocess.run([node, "-e", js], capture_output=True, text=True).stdout or "[]")
    names = [x[0] for x in out]
    check("three platforms get a suggestion", names == ["TikTok", "Instagram Reels", "YouTube Shorts"], names)
    if len(out) == 3:
        tt, ig, yt = (x[1] for x in out)
        check("no links in any suggestion", not any("http" in x for x in (tt, ig, yt)))
        check("TikTok: short caption, up to 5 hashtags", len(tt.split("\n\n")[0]) <= 150 and tt.count("#") == 5 and "#extra" not in tt)
        check("Reels opens with the hook", ig.startswith("¿QUÉ ACABA DE PASAR?") and ig.count("#") <= 5)
        check("Shorts: title first, #Shorts last tag", yt.startswith("No puede ser") and "#Shorts" in yt)
        check("the creator's footer line is added", all(x.endswith("Clipped by @canal") for x in (tt, ig, yt)))

print(f"\n{len(fails)} failed" if fails else "")
sys.exit(1 if fails else 0)
