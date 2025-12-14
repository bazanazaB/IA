import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import os

# ======================================================
# CONFIGURACIÓN
# ======================================================
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 35

TRAIN_DIR = "../../data/split/train"
VAL_DIR = "../../data/split/val"
MODEL_PATH = "../../models/best_model.h5"

# ======================================================
# GENERADORES DE IMÁGENES
# ======================================================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    zoom_range=0.3,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)


val_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# ======================================================
# MOSTRAR ORDEN REAL DE CLASES (CRÍTICO)
# ======================================================
print("\n📂 Clases y sus índices (ORDEN REAL):")
print(train_gen.class_indices)

# ======================================================
# MODELO CNN (MEJORADO)
# ======================================================
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(128,128,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Conv2D(256, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.6),

    Dense(train_gen.num_classes, activation="softmax")
])

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ======================================================
# CALLBACK: EARLY STOPPING
# ======================================================
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=6,
    restore_best_weights=True
)

# ======================================================
# ENTRENAMIENTO
# ======================================================
steps_per_epoch = train_gen.samples // BATCH_SIZE
validation_steps = val_gen.samples // BATCH_SIZE

history = model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_gen,
    validation_steps=validation_steps,
    epochs=EPOCHS,
    callbacks=[early_stop]
)

# ======================================================
# GUARDAR MODELO
# ======================================================
model.save(MODEL_PATH)
print(f"\n✅ Modelo guardado en: {MODEL_PATH}")
