import tensorflow as tf
import numpy as np
import os

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping


# ==========================================
# 1. BASIC SETTINGS
# ==========================================

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10

DATASET_PATH = "dataset"


# ==========================================
# 2. LOAD TRAINING DATA
# ==========================================

print("\nLoading training dataset...")

train_data = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,

    validation_split=0.2,

    subset="training",

    seed=123,

    image_size=(
        IMAGE_SIZE,
        IMAGE_SIZE
    ),

    batch_size=BATCH_SIZE
)


# ==========================================
# 3. LOAD VALIDATION DATA
# ==========================================

print("\nLoading validation dataset...")

validation_data = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,

    validation_split=0.2,

    subset="validation",

    seed=123,

    image_size=(
        IMAGE_SIZE,
        IMAGE_SIZE
    ),

    batch_size=BATCH_SIZE
)


# ==========================================
# 4. GET CLASS NAMES
# ==========================================

class_names = train_data.class_names

print("\nClasses found:")

for i, name in enumerate(class_names):

    print(
        i,
        "->",
        name
    )


NUM_CLASSES = len(class_names)


print(
    "\nTotal classes:",
    NUM_CLASSES
)


# ==========================================
# 5. DATA AUGMENTATION
# ==========================================

data_augmentation = tf.keras.Sequential([

    layers.RandomFlip(
        "horizontal"
    ),

    layers.RandomRotation(
        0.1
    ),

    layers.RandomZoom(
        0.1
    ),

])


# ==========================================
# 6. PREFETCH DATA
# ==========================================

AUTOTUNE = tf.data.AUTOTUNE

train_data = train_data.prefetch(
    buffer_size=AUTOTUNE
)

validation_data = validation_data.prefetch(
    buffer_size=AUTOTUNE
)


# ==========================================
# 7. EFFICIENTNET MODEL
# ==========================================

print("\nLoading EfficientNetB0...")

base_model = EfficientNetB0(

    include_top=False,

    weights="imagenet",

    input_shape=(
        IMAGE_SIZE,
        IMAGE_SIZE,
        3
    )

)


# Freeze pretrained layers

base_model.trainable = False


# ==========================================
# 8. BUILD MODEL
# ==========================================

model = models.Sequential([

    layers.Input(
        shape=(
            IMAGE_SIZE,
            IMAGE_SIZE,
            3
        )
    ),

    data_augmentation,

    base_model,

    layers.GlobalAveragePooling2D(),

    layers.Dropout(
        0.3
    ),

    layers.Dense(
        NUM_CLASSES,

        activation="softmax"
    )

])


# ==========================================
# 9. COMPILE MODEL
# ==========================================

model.compile(

    optimizer="adam",

    loss=
    "sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]

)


# ==========================================
# 10. SHOW MODEL
# ==========================================

print("\nModel summary:\n")

model.summary()


# ==========================================
# 11. EARLY STOPPING
# ==========================================

early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=3,

    restore_best_weights=True

)


# ==========================================
# 12. TRAIN MODEL
# ==========================================

print("\n================================")
print("STARTING MODEL TRAINING")
print("================================\n")


history = model.fit(

    train_data,

    validation_data=
    validation_data,

    epochs=EPOCHS,

    callbacks=[
        early_stopping
    ]

)


# ==========================================
# 13. CREATE MODEL FOLDER
# ==========================================

os.makedirs(
    "model",
    exist_ok=True
)


# ==========================================
# 14. SAVE MODEL
# ==========================================

model.save(
    "model/skin_disease_model.keras"
)


# ==========================================
# 15. SAVE CLASS NAMES
# ==========================================

with open(
    "model/classes.txt",
    "w"
) as file:

    for class_name in class_names:

        file.write(
            class_name + "\n"
        )


# ==========================================
# 16. FINAL MESSAGE
# ==========================================

print("\n================================")
print("TRAINING COMPLETE")
print("================================")

print(
    "\nModel saved at:"
)

print(
    "model/skin_disease_model.keras"
)

print(
    "\nClasses saved at:"
)

print(
    "model/classes.txt"
)