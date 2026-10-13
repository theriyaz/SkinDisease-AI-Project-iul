import tensorflow as tf
import numpy as np

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt


# ==========================================
# SETTINGS
# ==========================================

IMAGE_SIZE = 224
BATCH_SIZE = 32

DATASET_PATH = "dataset"


# ==========================================
# LOAD DATASET
# ==========================================

print("\nLoading dataset...")

dataset = tf.keras.utils.image_dataset_from_directory(

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
# GET CLASS NAMES
# ==========================================

class_names = dataset.class_names

print("\nClasses:")

for i, name in enumerate(class_names):

    print(
        i,
        "->",
        name
    )


# ==========================================
# LOAD MODEL
# ==========================================

print("\nLoading trained model...")

model = tf.keras.models.load_model(
    "model/skin_disease_model.keras"
)


print("Model loaded successfully!")


# ==========================================
# GET TRUE LABELS AND PREDICTIONS
# ==========================================

true_labels = []

predicted_labels = []


print("\nRunning evaluation...")


for images, labels in dataset:

    predictions = model.predict(
        images,
        verbose=0
    )

    predictions = np.argmax(
        predictions,
        axis=1
    )

    true_labels.extend(
        labels.numpy()
    )

    predicted_labels.extend(
        predictions
    )


# ==========================================
# ACCURACY
# ==========================================

true_labels = np.array(
    true_labels
)

predicted_labels = np.array(
    predicted_labels
)


accuracy = np.mean(
    true_labels == predicted_labels
)


print("\n================================")
print("MODEL ACCURACY")
print("================================")

print(
    "Accuracy:",
    round(
        accuracy * 100,
        2
    ),
    "%"
)


# ==========================================
# CLASSIFICATION REPORT
# ==========================================

print("\n================================")
print("CLASSIFICATION REPORT")
print("================================\n")


report = classification_report(

    true_labels,

    predicted_labels,

    target_names=class_names,

    zero_division=0

)


print(report)


# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(

    true_labels,

    predicted_labels

)


print("\n================================")
print("CONFUSION MATRIX")
print("================================")

print(cm)


# ==========================================
# CONFUSION MATRIX GRAPH
# ==========================================

plt.figure(
    figsize=(10, 8)
)

plt.imshow(cm)

plt.title(
    "Skin Disease Classification - Confusion Matrix"
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "True Label"
)

plt.xticks(
    range(len(class_names)),
    class_names,
    rotation=45
)

plt.yticks(
    range(len(class_names)),
    class_names
)


# Add numbers

for i in range(
    len(class_names)
):

    for j in range(
        len(class_names)
    ):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )


plt.tight_layout()

plt.savefig(
    "model/confusion_matrix.png"
)

plt.show()


print(
    "\nConfusion matrix saved:"
)

print(
    "model/confusion_matrix.png"
)