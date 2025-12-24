import faiss
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Cargar FAISS y metadata
index = faiss.read_index("vector_store.faiss")
df = pd.read_csv("metadata.csv")

# Cargar embeddings del disco
embeddings = np.load("embeddings.npy")

# Reducir dimensiones para graficar
tsne = TSNE(n_components=2, perplexity=20)
emb_2d = tsne.fit_transform(embeddings)

# Crear grafo
G = nx.Graph()

# Añadir nodos
for i, texto in enumerate(df["texto"]):
    G.add_node(i, label=texto, pos=(emb_2d[i][0], emb_2d[i][1]))

# Añadir aristas por similitud
similarity_matrix = embeddings @ embeddings.T
threshold = 0.75

for i in range(len(embeddings)):
    for j in range(i+1, len(embeddings)):
        if similarity_matrix[i, j] > threshold:
            G.add_edge(i, j, weight=similarity_matrix[i, j])

# Dibujar
pos = nx.get_node_attributes(G, "pos")
nx.draw(G, pos, node_size=30, edge_color="gray")
plt.title("Mapa Semántico — Grafo de Similitud")
plt.show()
