# ======================================================
# CONSTRUCCIÓN DE DATASET DESDE TEXTO FILOSÓFICO
# ======================================================

import re
import pandas as pd
import unicodedata
from pathlib import Path

# ======================================================
# RUTAS
# ======================================================

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "source"

INPUT_FILE = DATA_DIR / "dataset_crudo.csv"
OUTPUT_FILE = DATA_DIR / "dataset_normalizado.csv"

# ======================================================
# FUNCIONES
# ======================================================

def quitar_acentos(texto):
    return "".join(
        c for c in unicodedata.normalize("NFD", texto)
        if unicodedata.category(c) != "Mn"
    )

def limpiar_texto(texto):
    texto = texto.lower()
    texto = quitar_acentos(texto)
    texto = re.sub(r"[\"“”‘’]", "", texto)
    texto = re.sub(r"[^a-z0-9\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto

def inferir_sentimiento(texto):
    negativos = [
        "ansiedad", "depresion", "vacio", "agotamiento",
        "burnout", "miedo", "soledad", "control"
    ]
    for palabra in negativos:
        if palabra in texto:
            return "negativo"
    return "neutral"

# ======================================================
# PARSEO DEL ARCHIVO
# ======================================================

rows = []
tema_actual = None
id_actual = 1

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for linea in f:
        linea = linea.strip()

        if not linea:
            continue

        # Detectar tema (líneas en MAYÚSCULAS sin comillas)
        if linea.isupper() and len(linea) > 10 and "TEMA" not in linea:
            tema_actual = limpiar_texto(linea)
            continue

        # Tomar solo textos entre comillas
        if linea.startswith('"') and linea.endswith('"') and tema_actual:
            texto_limpio = limpiar_texto(linea)

            if len(texto_limpio) < 30:
                continue

            rows.append({
                "id": id_actual,
                "texto": texto_limpio,
                "tema": tema_actual,
                "sentimiento": inferir_sentimiento(texto_limpio)
            })

            id_actual += 1

# ======================================================
# DATAFRAME FINAL
# ======================================================

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

print("✓ Dataset construido correctamente")
print(f"📁 Archivo: {OUTPUT_FILE}")
print(f"📊 Filas: {len(df)}")
print(df.head())
