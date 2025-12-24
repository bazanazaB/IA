# visualizar_umap_clusters.py
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import umap
import hdbscan
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# Cargar
emb = np.load("embeddings.npy")        # shape (N, D)
df = pd.read_csv("metadata.csv")       # debe tener columna "texto"

# Reducir dimensionalidad con UMAP
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
emb_2d = reducer.fit_transform(emb)   # shape (N,2)

# Clustering con HDBSCAN
clusterer = hdbscan.HDBSCAN(min_cluster_size=15, metric='euclidean')
clusters = clusterer.fit_predict(emb_2d)
df["cluster"] = clusters

# Obtener top términos por cluster (TF-IDF)
vectorizer = TfidfVectorizer(max_features=5000, stop_words='spanish')
X_tfidf = vectorizer.fit_transform(df["texto"].astype(str))
terms = vectorizer.get_feature_names_out()

top_terms_by_cluster = {}
for c in sorted(set(clusters)):
    if c == -1:
        continue
    idx = (df["cluster"] == c).values
    if idx.sum() == 0:
        continue
    cluster_tfidf = X_tfidf[idx].mean(axis=0).A1
    top_indices = cluster_tfidf.argsort()[-8:][::-1]
    top_terms_by_cluster[c] = [terms[i] for i in top_indices]

# Guardar top terms a CSV
pd.DataFrame([
    {"cluster": c, "top_terms": ", ".join(top_terms_by_cluster.get(c, []))}
    for c in top_terms_by_cluster
]).to_csv("top_terms_by_cluster.csv", index=False, encoding="utf-8")

# Plot UMAP scatter (un solo gráfico)
plt.figure(figsize=(10,8))
# puntos por cluster; los ruidosos (-1) en gris
unique_clusters = sorted(set(clusters))
for c in unique_clusters:
    sel = clusters == c
    size = 8 if c != -1 else 4
    label = f"cluster {c}" if c != -1 else "ruido"
    plt.scatter(emb_2d[sel,0], emb_2d[sel,1], s=size, label=label, alpha=0.6)
plt.title("Mapa semántico UMAP + HDBSCAN")
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.legend(markerscale=2, fontsize=8)
plt.tight_layout()
plt.savefig("umap_clusters.png", dpi=300)
print("✓ Guardado: umap_clusters.png")
print("✓ Guardado: top_terms_by_cluster.csv")
