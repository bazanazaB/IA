import os
import cv2

IMG_SIZE = 128

RAW_DIR = "../../data/raw"
PROC_DIR = "../../data/processed"

os.makedirs(PROC_DIR, exist_ok=True)

for label in os.listdir(RAW_DIR):
    src = os.path.join(RAW_DIR, label)
    dst = os.path.join(PROC_DIR, label)
    os.makedirs(dst, exist_ok=True)

    for img_name in os.listdir(src):
        img_path = os.path.join(src, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        cv2.imwrite(os.path.join(dst, img_name), img)

print("✅ Preprocess terminado")
