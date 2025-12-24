import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from pathlib import Path

# ============================================================
# RUTAS BASE
# ============================================================
BASE_DIR = Path(__file__).resolve().parents[2]   # raíz del proyecto
DATA_SOURCE = BASE_DIR / "data" / "source"
DATA_EMB = BASE_DIR / "data" / "embeddings"

INPUT_CSV = DATA_SOURCE / "dataset_normalizado.csv"
EMB_PATH = DATA_EMB / "embeddings.npy"
META_PATH = DATA_SOURCE / "metadata.csv"

DATA_EMB.mkdir(parents=True, exist_ok=True)

# ============================================================
# Evitar regenerar embeddings
# ============================================================
if EMB_PATH.exists() and META_PATH.exists():
    print("✓ Embeddings y metadata ya existen. Saltando generación.")
    exit()

# ============================================================
# Cargar dataset limpio
# ============================================================
df = pd.read_csv(INPUT_CSV)

model = SentenceTransformer("all-mpnet-base-v2")

batch_size = 64
embeddings = []

print("Generando embeddings...")

for i in tqdm(range(0, len(df), batch_size)):
    batch = df["texto"].iloc[i:i + batch_size].tolist()
    emb = model.encode(batch, show_progress_bar=False)
    embeddings.append(emb)

embeddings = np.vstack(embeddings)

# ============================================================
# Guardar resultados
# ============================================================
np.save(EMB_PATH, embeddings)
df.to_csv(META_PATH, index=False)

print(f"✓ Embeddings guardados en: {EMB_PATH}")
print(f"✓ Metadata guardada en: {META_PATH}")
