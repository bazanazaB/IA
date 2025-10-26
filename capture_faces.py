import os
import numpy as np
import cv2 as cv

# === CONFIGURACIÓN INICIAL ===
# Pregunta qué emoción quieres capturar
emotion = input("👉 Ingresa la emoción a capturar (Normal, Confused, Angry, Smile, Sad): ").strip().capitalize()

# Verifica que sea una emoción válida
valid_emotions = ['Normal', 'Confused', 'Angry', 'Smile', 'Sad']
if emotion not in valid_emotions:
    print(f"❌ '{emotion}' no es válida. Usa una de: {valid_emotions}")
    exit(1)

# === CONFIGURA EL CLASIFICADOR DE ROSTROS ===
# Intentar localizar el XML del clasificador de varias formas:
# 1) archivo junto al script
# 2) en la carpeta detectaCaras dentro del proyecto
# 3) fallback a la carpeta de haarcascades instalada con OpenCV
script_dir = os.path.dirname(__file__)
candidates = [
    os.path.join(script_dir, 'haarcascade_frontalface_alt2.xml'),
    os.path.join(script_dir, 'detectaCaras', 'haarcascade_frontalface_alt2.xml'),
    os.path.join(cv.data.haarcascades, 'haarcascade_frontalface_alt2.xml')
]
cascade_path = None
for p in candidates:
    if os.path.exists(p):
        cascade_path = p
        break

if cascade_path is None:
    raise FileNotFoundError(
        "❌ No se encontró el archivo del clasificador 'haarcascade_frontalface_alt2.xml'.\n"
        f"Busqué en: {candidates}\n"
        "Puedes copiar el XML en el mismo directorio del script o instalar OpenCV correctamente."
    )

rostro = cv.CascadeClassifier(cascade_path)

# === ABRIR VIDEO O CÁMARA ===
video_path = os.path.join(os.path.dirname(__file__), 'korean.mp4')
if os.path.exists(video_path):
    cap = cv.VideoCapture(video_path)
else:
    cap = cv.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError('❌ No se pudo abrir la cámara ni el video.')

# === CREAR DIRECTORIO DE SALIDA ===
dataset_dir = os.path.join(os.path.dirname(__file__), 'sentimientos-dataset')
emotion_dir = os.path.join(dataset_dir, emotion.lower())
os.makedirs(emotion_dir, exist_ok=True)

print(f"✅ Guardando imágenes en: {emotion_dir}")
print("Presiona 'ESC' para detener la captura.\n")

# === BUCLE PRINCIPAL DE CAPTURA ===
i = 0
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    rostros = rostro.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in rostros:
        face = frame[y:y+h, x:x+w]
        if face.size == 0:
            continue

        face = cv.resize(face, (100, 100), interpolation=cv.INTER_AREA)

        # Guardar cada 2 frames (puedes cambiar a 1 si quieres más imágenes)
        if i % 2 == 0:
            filename = os.path.join(emotion_dir, f'{emotion.lower()}{i}.jpg')
            cv.imwrite(filename, face)
            cv.imshow('Rostro Detectado', face)

    cv.imshow('Detección en vivo', frame)
    i += 1

    # Presionar ESC para salir
    k = cv.waitKey(1) & 0xFF
    if k == 27:
        break

cap.release()
cv.destroyAllWindows()

print(f"\n📸 Captura finalizada. Se guardaron imágenes en: {emotion_dir}")
