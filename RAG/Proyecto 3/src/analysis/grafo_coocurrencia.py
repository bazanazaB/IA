# grafo_coocurrencia.py
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import networkx as nx
from pyvis.network import Network

# Cargar
df = pd.read_csv("metadata.csv")
texts = df["texto"].astype(str).tolist()

# Vectorizador de conteo (n-gram=1)
vectorizer = CountVectorizer(max_features=2000, stop_words='spanish', min_df=5)
X = vectorizer.fit_transform(texts)
terms = vectorizer.get_feature_names_out()

# Construir co-ocurrencia (term-term)
cooc = (X.T @ X).toarray()
import numpy as np
np.fill_diagonal(cooc, 0)

# Crear grafo y filtrar por umbral de fuerza
G = nx.Graph()
threshold = 20  # ajustar según tamaño del corpus
for i, term in enumerate(terms):
    G.add_node(term, size=int(X[:,i].sum()))

for i in range(len(terms)):
    for j in range(i+1, len(terms)):
        w = cooc[i,j]
        if w >= threshold:
            G.add_edge(terms[i], terms[j], weight=int(w))

# Crear visual interactiva con pyvis
net = Network(height="800px", width="100%", notebook=False, cdn_resources='in_line')
net.from_nx(G)
net.show("grafo_coocurrencia.html")
print("✓ Guardado: grafo_coocurrencia.html")
