# wordcloud_clusters.py
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

df = pd.read_csv("metadata.csv")
if "cluster" not in df.columns:
    print("❗ No hay columna 'cluster'. Ejecuta primero visualizar_umap_clusters.py")
    exit()

clusters = df["cluster"].unique()
for c in clusters:
    if c == -1:
        continue
    text = " ".join(df[df["cluster"]==c]["texto"].astype(str).tolist())
    wc = WordCloud(width=800, height=400, collocations=False).generate(text)
    plt.figure(figsize=(10,5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"WordCloud cluster {c}")
    plt.tight_layout()
    outname = f"wordcloud_cluster_{c}.png"
    plt.savefig(outname, dpi=300)
    print(f"✓ Guardado: {outname}")
