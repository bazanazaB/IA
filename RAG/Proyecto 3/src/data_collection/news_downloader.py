# ======================================================
# REDDIT EXISTENTIAL DATASET DOWNLOADER (ENGLISH)
# ======================================================

import praw
import pandas as pd
from pathlib import Path
import re
import time

# ======================================================
# CONFIGURACIÓN DE RUTAS
# ======================================================

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "source"
DATA_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = DATA_DIR / "dataset_raw_reddit.csv"

# ======================================================
# REDDIT API (USA TU API KEY REAL)
# ======================================================

reddit = praw.Reddit(
    client_id="TU_CLIENT_ID",
    client_secret="TU_CLIENT_SECRET",
    user_agent="existential_rag_project"
)

# ======================================================
# SUBREDDITS BUENOS (FUNCIONAN DE VERDAD)
# ======================================================

SUBREDDITS = [
    "existentialism",
    "philosophy",
    "TrueOffMyChest",
    "changemyview",
    "LifeAdvice"
]

# ======================================================
# PALABRAS CLAVE EXISTENCIALES (INGLÉS)
# ======================================================

KEYWORDS = [
    "meaning", "purpose", "identity", "anxiety", "lonely",
    "existential", "future", "life", "death", "fear",
    "technology", "control", "autonomy", "void", "worth"
]

MIN_WORDS = 120
POSTS_PER_SUB = 300

# ======================================================
# FUNCIONES
# ======================================================

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def is_existential(text):
    t = text.lower()
    return sum(k in t for k in KEYWORDS) >= 2

# ======================================================
# DESCARGA
# ======================================================

print("\n==============================")
print("  DOWNLOADING REDDIT CORPUS")
print("==============================\n")

rows = []
post_id = 1

for sub in SUBREDDITS:
    print(f"📥 Subreddit: r/{sub}")
    subreddit = reddit.subreddit(sub)

    for post in subreddit.top(time_filter="year", limit=POSTS_PER_SUB):

        if post.selftext in ["", "[deleted]", "[removed]"]:
            continue

        text = clean_text(post.title + " " + post.selftext)

        if len(text.split()) < MIN_WORDS:
            continue

        if not is_existential(text):
            continue

        rows.append({
            "id": post_id,
            "fecha": pd.to_datetime(post.created_utc, unit="s"),
            "fuente": f"reddit/{sub}",
            "titulo": post.title,
            "texto": text,
            "url": f"https://reddit.com{post.permalink}"
        })

        post_id += 1

    time.sleep(2)

# ======================================================
# GUARDADO
# ======================================================

df = pd.DataFrame(rows)

if df.empty:
    print("❌ Dataset vacío (muy raro). Revisa API.")
else:
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print("\n✅ Dataset creado correctamente")
    print(f"📁 Guardado en: {OUTPUT_CSV}")
    print(f"📊 Textos obtenidos: {len(df)}")

print("\n================================\n")
