import tensorflow as tf
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# -------------------------
# CONFIGURACIÓN
# -------------------------
MODEL_PATH = "../models/best_model.h5"
IMG_SIZE = 128

class_names = [
    "catarina",  # 0
    "gato",      # 1
    "hormiga",   # 2
    "perro",     # 3
    "tortuga"    # 4
]

# -------------------------
# CARGAR MODELO
# -------------------------
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Modelo cargado")

# -------------------------
# FUNCIÓN PREDICCIÓN
# -------------------------


def predict_image(path):
    img = cv2.imread(path)

    if img is None:
        return "Error", 0

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0]
    idx = np.argmax(pred)

    return class_names[idx], pred[idx]

# -------------------------
# FUNCIÓN BOTÓN
# -------------------------
def open_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Images", "*.jpg *.png *.jpeg")]
    )

    if not file_path:
        return

    # Mostrar imagen
    img = Image.open(file_path)
    img = img.resize((300, 300))
    img_tk = ImageTk.PhotoImage(img)

    img_label.config(image=img_tk)
    img_label.image = img_tk

    # Predicción
    pred_class, conf = predict_image(file_path)
    result_label.config(
        text=f"🧠 Predicción: {pred_class}\n🎯 Confianza: {conf*100:.2f}%"
    )

# -------------------------
# VENTANA
# -------------------------
root = tk.Tk()
root.title("CNN - Clasificador de Animales")
root.geometry("400x500")

title = tk.Label(root, text="Clasificador de Animales", font=("Arial", 16))
title.pack(pady=10)

btn = tk.Button(root, text="📂 Subir imagen", command=open_image, font=("Arial", 12))
btn.pack(pady=10)

img_label = tk.Label(root)
img_label.pack(pady=10)

result_label = tk.Label(root, text="", font=("Arial", 12))
result_label.pack(pady=20)

root.mainloop()
