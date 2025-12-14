import os
import shutil
import random

SOURCE_DIR = "../../data/processed"
TARGET_DIR = "../../data/split"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)

for cls in os.listdir(SOURCE_DIR):
    cls_path = os.path.join(SOURCE_DIR, cls)

    # saltar carpetas ya creadas
    if cls in ["train", "val", "test"]:
        continue

    images = os.listdir(cls_path)
    random.shuffle(images)

    n = len(images)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:]
    }

    for split, files in splits.items():
        split_dir = os.path.join(TARGET_DIR, split, cls)
        os.makedirs(split_dir, exist_ok=True)

        for f in files:
            src = os.path.join(cls_path, f)
            dst = os.path.join(split_dir, f)
            shutil.copy(src, dst)

print("✅ Dataset dividido en train / val / test")
