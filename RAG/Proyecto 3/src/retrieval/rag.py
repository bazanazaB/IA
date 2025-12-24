# ======================================================
# RAG – RETRIEVAL AUGMENTED GENERATION (DISCURSIVO)
# ======================================================
# Análisis sociológico / cultural de texto
# Proyecto escolar – versión estable
# ======================================================

import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

# ======================================================
# CONFIGURACIÓN
# ======================================================

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "source"

DATASET_PATH = DATA_DIR / "dataset_normalizado.csv"
TEXT_COL = "texto"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL = "HuggingFaceH4/zephyr-7b-beta"




TOP_K = 4
MAX_CHARS_PER_DOC = 450
MAX_NEW_TOKENS = 160

# ======================================================
# INICIO
# ======================================================

print("\nRAG listo. Escribe una pregunta (exit para salir)\n")

df = pd.read_csv(DATASET_PATH)

if TEXT_COL not in df.columns:
    raise ValueError(f"El dataset debe contener la columna '{TEXT_COL}'")

texts = df[TEXT_COL].astype(str).tolist()

# ======================================================
# MODELOS
# ======================================================

embedder = SentenceTransformer(EMBED_MODEL)
embeddings = embedder.encode(texts, show_progress_bar=True)

generator = pipeline(
    "text-generation",
    model=GEN_MODEL,
    device=-1  # CPU
)

# ======================================================
# FUNCIÓN DE RETRIEVAL
# ======================================================

def buscar_contexto(pregunta: str) -> str:
    """
    Recupera los fragmentos más relevantes del dataset
    según similitud semántica.
    """
    pregunta_emb = embedder.encode([pregunta])
    scores = cosine_similarity(pregunta_emb, embeddings)[0]

    top_idx = np.argsort(scores)[-TOP_K:][::-1]

    fragmentos = []
    for idx in top_idx:
        fragmentos.append(
            texts[idx][:MAX_CHARS_PER_DOC]
        )

    return "\n\n".join(fragmentos)

# ======================================================
# LOOP INTERACTIVO
# ======================================================

while True:
    pregunta = input("🧠 Pregunta > ")

    if pregunta.lower() == "exit":
        print("\nSesión finalizada.\n")
        break

    contexto = buscar_contexto(pregunta)

    # ==================================================
    # PROMPT ESTABLE (NO RESTRICTIVO)
    # ==================================================

    prompt = f"""
Tarea:
A partir del contexto, redacta una respuesta concisa en español.
Integra las ideas de forma coherente y reflexiva.

Contexto:
{contexto}

Pregunta:
{pregunta}

Respuesta:
"""

    salida = generator(
        prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=0.65,
        top_p=0.85
    )[0]["generated_text"]

    print("\n📘 Respuesta:\n")
    print(salida.strip())
    print("\n" + "-" * 90 + "\n")
