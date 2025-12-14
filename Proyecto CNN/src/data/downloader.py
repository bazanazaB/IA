from bing_image_downloader import downloader
import os

BASE_DIR = os.path.join("..", "..", "data", "raw")

QUERIES = [
    "tortuga"
]

for q in QUERIES:
    downloader.download(
        q,
        limit=150,          # 🔥 pequeño = rápido
        output_dir=BASE_DIR,
        adult_filter_off=True,
        force_replace=False,
        timeout=5,
        verbose=True
    )

# # import kagglehub

# # # Download latest version
# # path = kagglehub.dataset_download("cookiemonsteryum/turtles")

# # print("Path to dataset files:", path)