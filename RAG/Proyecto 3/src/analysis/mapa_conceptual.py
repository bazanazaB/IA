# ======================================================
# GENERADOR DE MAPA CONCEPTUAL SEMÁNTICO
# ======================================================

import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# ======================================================
# CONFIGURACIÓN
# ======================================================

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "source"
OUTPUT_DIR = BASE_DIR / "data" / "analysis"

INPUT_CSV = DATA_DIR / "dataset_normalizado.csv"
OUTPUT_CSV = OUTPUT_DIR / "mapa_conceptual.csv"

TEXT_COL = "texto"     # 👈 tu dataset YA viene normalizado
N_CLUSTERS = 4         # puedes ajustar

# ======================================================
# CARGA DE DATOS
# ======================================================

print("\n==================================================")
print("  GENERANDO MAPA CONCEPTUAL SEMÁNTICO")
print("==================================================\n")

df = pd.read_csv(INPUT_CSV)

if TEXT_COL not in df.columns:
    raise ValueError(f"El CSV debe contener una columna llamada '{TEXT_COL}'")

texts = df[TEXT_COL].astype(str).tolist()

# ======================================================
# VECTORIZACIÓN TF-IDF
# ======================================================

vectorizer = TfidfVectorizer(
    max_features=500,
    min_df=1,
    max_df=0.95,
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(texts)

# ======================================================
# CLUSTERING
# ======================================================

kmeans = KMeans(
    n_clusters=N_CLUSTERS,
    random_state=42,
    n_init="auto"
)

clusters = kmeans.fit_predict(X)

df["cluster"] = clusters

# ======================================================
# TÉRMINOS REPRESENTATIVOS POR CLUSTER
# ======================================================

terms = vectorizer.get_feature_names_out()
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

conceptos = []

for i in range(N_CLUSTERS):
    top_terms = [terms[ind] for ind in order_centroids[i, :8]]
    conceptos.append(", ".join(top_terms))

cluster_df = pd.DataFrame({
    "cluster": range(N_CLUSTERS),
    "conceptos_clave": conceptos
})

# ======================================================
# GUARDADO
# ======================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df_salida = df.merge(cluster_df, on="cluster", how="left")

df_salida.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

print("✓ Mapa conceptual generado correctamente")
print(f"📁 Archivo: {OUTPUT_CSV}")
print(f"📊 Clusters: {N_CLUSTERS}")
print("\n=========================================\n")
