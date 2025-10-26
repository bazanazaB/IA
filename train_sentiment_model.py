import cv2 as cv
import numpy as np
import os
import sys

# Dataset folder next to this script (sentimientos-dataset)
dataSet = os.path.join(os.path.dirname(__file__), 'sentimientos-dataset')
if not os.path.isdir(dataSet):
    print(f"Dataset folder not found: {dataSet}")
    print("Create the folder and place subfolders for each class (e.g. 'smile', 'neutral'),")
    print("each with their image files (.jpg/.png).")
    sys.exit(1)

labels = []
facesData = []
classes = []
label = 0

for class_name in sorted(os.listdir(dataSet)):
    class_path = os.path.join(dataSet, class_name)
    if not os.path.isdir(class_path):
        continue
    classes.append(class_name)
    for fname in os.listdir(class_path):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        fpath = os.path.join(class_path, fname)
        img = cv.imread(fpath, 0)  # load as grayscale
        if img is None:
            print(f"Warning: could not read image {fpath}, skipping.")
            continue
        # ensure consistent size (adjust to match your detection output size)
        img = cv.resize(img, (100, 100), interpolation=cv.INTER_AREA)
        facesData.append(img)
        labels.append(label)
    label += 1

if len(facesData) == 0:
    print("No training images found. Check your dataset folder structure and image files.")
    sys.exit(1)

print(f"Found {len(facesData)} images across {len(classes)} classes: {classes}")

# Choose a recognizer available in your OpenCV build
if hasattr(cv, 'face') and hasattr(cv.face, 'EigenFaceRecognizer_create'):
    recognizer = cv.face.EigenFaceRecognizer_create()
elif hasattr(cv, 'face') and hasattr(cv.face, 'LBPHFaceRecognizer_create'):
    recognizer = cv.face.LBPHFaceRecognizer_create()
else:
    print("OpenCV face module not found. Install opencv-contrib-python:")
    print("pip install opencv-contrib-python")
    sys.exit(1)

# Train (facesData is a list of 2D arrays)
try:
    recognizer.train(facesData, np.array(labels))
except Exception as e:
    print("Error during training:", e)
    sys.exit(1)

out_model = os.path.join(os.path.dirname(__file__), 'face_model.xml')
recognizer.write(out_model)
print("Model saved to", out_model)