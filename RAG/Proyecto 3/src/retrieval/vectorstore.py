import faiss
import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================
# RUTAS BASE
# ============================================================
BASE_DIR = Path(__file__).resolve().parents[2]   # raíz del proyecto

DATA_SOURCE = BASE_DIR / "data" / "source"
DATA_EMB = BASE_DIR / "data" / "embeddings"

EMB_PATH = DATA_EMB / "embeddings.npy"
META_PATH = DATA_SOURCE / "metadata.csv"
FAISS_PATH = DATA_EMB / "vector_store.faiss"

# ============================================================
# Validaciones
# ============================================================
if not EMB_PATH.exists():
    raise FileNotFoundError(f"No se encontraron embeddings: {EMB_PATH}")

if not META_PATH.exists():
    raise FileNotFoundError(f"No se encontró metadata: {META_PATH}")

# ============================================================
# Cargar embeddings y metadata
# ============================================================
embeddings = np.load(EMB_PATH)
df = pd.read_csv(META_PATH)

# ============================================================
# Crear índice FAISS
# ============================================================
dimension = embeddings.shape[1]  # normalmente 768
index = faiss.IndexFlatL2(dimension)

# ============================================================
# Agregar embeddings
# ============================================================
index.add(embeddings)

# ============================================================
# Guardar vector store
# ============================================================
faiss.write_index(index, str(FAISS_PATH))

print(f"✓ Vector store creado y guardado en: {FAISS_PATH}")
print(f"✓ Metadata disponible en: {META_PATH}")
