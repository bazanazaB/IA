## Dataset
📥 Descarga de imágenes

Este proyecto no incluye imágenes dentro del repositorio para evitar que GitHub sea pesado y por temas de licencia.
En su lugar, las imágenes se descargan automáticamente ejecutando un script.
¿Cómo se descargan las imágenes?

El archivo:
src/data/downloader.py
se encarga de descargar imágenes desde Bing Images usando palabras clave.
Actualmente está configurado para descargar imágenes de:

🐢 tortugas
El script:
Descarga 150 imágenes
Guarda las imágenes en la carpeta data/raw/
Usa un filtro para evitar contenido adulto
No sobrescribe imágenes si ya existen
Ejemplo del código de descarga
from bing_image_downloader import downloader
import os

BASE_DIR = os.path.join("..", "..", "data", "raw")
QUERIES = ["tortuga"]

for q in QUERIES:
    downloader.download(
        q,
        limit=150,
        output_dir=BASE_DIR,
        adult_filter_off=True,
        force_replace=False,
        timeout=5,
        verbose=True
    )

🔁 Flujo de preparación del dataset

Para generar el dataset completo desde cero, solo hay que ejecutar los scripts en este orden:

python src/data/downloader.py   # Descarga las imágenes
python src/data/preprocess.py   # Preprocesa las imágenes
python src/data/split.py        # Divide en train / val / test


Al final, el dataset queda listo para entrenar el modelo en:

data/split/

ℹ️ Nota sobre las imágenes

Las imágenes se utilizan solo con fines educativos.
Si deseas usar otro tipo de imágenes o clases, basta con cambiar las palabras dentro de la lista QUERIES.

Orden de ejecución deseada:
1. Ejecutar `src/data/downloader.py`
2. Ejecutar `src/data/preprocess.py`
3. Ejecutar `src/data/split.py`
4. Entrenar con `src/training/train.py`
5. Test con `test/predict.py`
