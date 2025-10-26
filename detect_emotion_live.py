import cv2 as cv
import os
import sys

# === BUSCAR MODELO ENTRENADO ===
base_dir = os.path.dirname(__file__)
candidate_models = [
    os.path.join(base_dir, 'Eigenface-rostro.xml'),
    os.path.join(base_dir, 'Eigenface-sentimientos.xml'),
    os.path.join(base_dir, 'lbph-sentimientos.xml'),
    os.path.join(base_dir, 'Eigenface-sentimientos.xml'),
    os.path.join(base_dir, 'face_model.xml'),
    os.path.join(base_dir, 'model.xml'),
]
model_path = None
for p in candidate_models:
    if os.path.exists(p):
        model_path = p
        break

if model_path is None:
    print('❌ No se encontró un archivo de modelo entrenado en el directorio del script.')
    print('Busqué estos nombres comunes:', [os.path.basename(p) for p in candidate_models])
    print('Archivos presentes en el directorio:', os.listdir(base_dir))
    print('Coloca el archivo del modelo (por ejemplo face_model.xml) en:', base_dir)
    sys.exit(1)

print('✅ Usando modelo:', model_path)

# === CARGAR RECONOCEDOR ===
faceRecognizer = None
errors = []
try:
    if hasattr(cv, 'face') and hasattr(cv.face, 'EigenFaceRecognizer_create'):
        r = cv.face.EigenFaceRecognizer_create()
        r.read(model_path)
        faceRecognizer = r
except Exception as e:
    errors.append(('eigen', str(e)))

if faceRecognizer is None:
    try:
        if hasattr(cv, 'face') and hasattr(cv.face, 'LBPHFaceRecognizer_create'):
            r = cv.face.LBPHFaceRecognizer_create()
            r.read(model_path)
            faceRecognizer = r
    except Exception as e:
        errors.append(('lbph', str(e)))

if faceRecognizer is None:
    print('❌ No se pudo cargar el modelo con EigenFace ni LBPH. Errores:')
    for name, err in errors:
        print(name, err)
    print('Asegúrate de que el modelo fue generado por la misma técnica (EigenFace o LBPH).')
    sys.exit(1)

# === CLASES (en el mismo orden que tus carpetas del dataset) ===
faces = ['Normal', 'Confused', 'Angry', 'Smile', 'Sad']

# === CARGAR CLASIFICADOR DE ROSTROS ===
# Buscar el cascade en la instalación de OpenCV primero, luego en carpetas locales
cascade_candidates = [
    os.path.join(cv.data.haarcascades, 'haarcascade_frontalface_alt2.xml'),
    os.path.join(base_dir, 'haarcascade_frontalface_alt2.xml'),
    os.path.join(base_dir, 'detectaCaras', 'haarcascade_frontalface_alt2.xml'),
]
cascade_path = None
for p in cascade_candidates:
    if os.path.exists(p):
        cascade_path = p
        break
if cascade_path is None:
    print('❌ No se encontró el archivo del clasificador. Busqué en:', cascade_candidates)
    sys.exit(1)
rostro = cv.CascadeClassifier(cascade_path)

# === CARGAR MAPA DE ETIQUETAS SI EXISTE ===
labels_file = os.path.join(base_dir, 'labels.json')
if os.path.exists(labels_file):
    try:
        import json
        with open(labels_file, 'r', encoding='utf-8') as f:
            faces = json.load(f)
        print('✅ Cargadas etiquetas desde', labels_file)
    except Exception:
        print('⚠️ Error leyendo labels.json — usando la lista por defecto')

# === ABRIR CÁMARA ===
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print('Error: No se puede abrir la cámara')
    sys.exit(1)

# === RECONOCIMIENTO EN TIEMPO REAL ===
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    rostros = rostro.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3)

    for (x, y, w, h) in rostros:
        face_img = gray[y:y+h, x:x+w]
        if face_img.size == 0:
            continue
        face_resized = cv.resize(face_img, (100, 100), interpolation=cv.INTER_CUBIC)

        try:
            label, conf = faceRecognizer.predict(face_resized)
        except Exception:
            cv.putText(frame, 'Error predict', (x, y-20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            continue

        # === Mostrar etiqueta y confianza ===
        cv.putText(frame, f'{label} {conf:.1f}', (x, y-20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        # === Ajusta el umbral de confianza (menor = más seguro) ===
        if conf < 4000:
            name = faces[label] if 0 <= label < len(faces) else f'ID:{label}'
            cv.putText(frame, name, (x, y-40), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        else:
            cv.putText(frame, 'Desconocido', (x, y-40), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cv.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 2)

    cv.imshow('Reconocimiento Facial', frame)

    k = cv.waitKey(1) & 0xFF
    if k == 27:  # ESC
        break

cap.release()
cv.destroyAllWindows()
