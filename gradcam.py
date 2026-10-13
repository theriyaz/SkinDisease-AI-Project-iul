import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

from PIL import Image


# ==========================================
# SETTINGS
# ==========================================

IMAGE_SIZE = 224

MODEL_PATH = "model/skin_disease_model.keras"

OUTPUT_PATH = "model/gradcam_result.jpg"


# ==========================================
# LOAD MODEL
# ==========================================

print("Loading model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully!")


# ==========================================
# CLASS NAMES
# ==========================================

class_names = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc"
]


disease_names = {

    "akiec": "Actinic Keratosis",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevus",
    "vasc": "Vascular Lesion"

}


# ==========================================
# IMAGE PATH
# ==========================================

image_path = input(
    "\nEnter image path: "
).strip().strip('"')


# ==========================================
# LOAD IMAGE
# ==========================================

original_image = Image.open(
    image_path
).convert("RGB")


original_image = original_image.resize(
    (IMAGE_SIZE, IMAGE_SIZE)
)


image_array = np.array(
    original_image
).astype("float32"
)


image_input = np.expand_dims(
    image_array,
    axis=0
)


# ==========================================
# GET MODEL LAYERS
# ==========================================

augmentation_model = model.layers[0]

efficientnet = model.layers[1]

gap_layer = model.layers[2]

dropout_layer = model.layers[3]

dense_layer = model.layers[4]


print(
    "\nEfficientNet:",
    efficientnet.name
)


# ==========================================
# PREDICTION + GRADIENTS
# ==========================================

with tf.GradientTape() as tape:

    # Apply augmentation layer
    augmented_image = augmentation_model(
        image_input,
        training=False
    )

    # Run EfficientNet
    conv_output = efficientnet(
        augmented_image,
        training=False
    )

    # Watch convolution output
    tape.watch(conv_output)

    # Classification head
    pooled_output = gap_layer(
        conv_output
    )

    dropped_output = dropout_layer(
        pooled_output,
        training=False
    )

    predictions = dense_layer(
        dropped_output
    )

    # Predicted class
    predicted_index = tf.argmax(
        predictions[0]
    )

    class_score = predictions[
        0,
        predicted_index
    ]


# ==========================================
# PREDICTION RESULT
# ==========================================

predictions_numpy = predictions.numpy()

predicted_index = int(
    np.argmax(
        predictions_numpy[0]
    )
)


predicted_class = class_names[
    predicted_index
]


confidence = (
    predictions_numpy[
        0,
        predicted_index
    ] * 100
)


print("\n================================")
print("PREDICTION")
print("================================")

print(
    "Disease:",
    disease_names[predicted_class]
)

print(
    "Class:",
    predicted_class
)

print(
    "Confidence:",
    round(
        confidence,
        2
    ),
    "%"
)


# ==========================================
# GET GRADIENTS
# ==========================================

gradients = tape.gradient(
    class_score,
    conv_output
)


if gradients is None:

    raise ValueError(
        "Gradients are None. "
        "Grad-CAM cannot be calculated."
    )


print(
    "\nGradients calculated successfully!"
)


# ==========================================
# GLOBAL AVERAGE POOLING
# ==========================================

pooled_gradients = tf.reduce_mean(
    gradients,
    axis=(0, 1, 2)
)


# ==========================================
# CREATE HEATMAP
# ==========================================

conv_output = conv_output[0]


heatmap = tf.reduce_sum(
    conv_output * pooled_gradients,
    axis=-1
)


heatmap = tf.maximum(
    heatmap,
    0
)


heatmap = heatmap / (
    tf.reduce_max(
        heatmap
    ) + 1e-8
)


heatmap = heatmap.numpy()


# ==========================================
# CREATE MODEL FOLDER
# ==========================================

os.makedirs(
    "model",
    exist_ok=True
)


# ==========================================
# DISPLAY RESULT
# ==========================================

plt.figure(
    figsize=(12, 5)
)


# ------------------------------------------
# ORIGINAL IMAGE
# ------------------------------------------

plt.subplot(
    1,
    2,
    1
)


plt.imshow(
    original_image
)


plt.title(
    "Original Dermoscopic Image"
)


plt.axis(
    "off"
)


# ------------------------------------------
# GRAD-CAM
# ------------------------------------------

plt.subplot(
    1,
    2,
    2
)


plt.imshow(
    original_image
)


plt.imshow(
    heatmap,
    cmap="jet",
    alpha=0.45
)


plt.title(
    "Grad-CAM Explainability"
)


plt.axis(
    "off"
)


plt.tight_layout()


# ==========================================
# SAVE
# ==========================================

plt.savefig(
    OUTPUT_PATH,
    dpi=200,
    bbox_inches="tight"
)


plt.show()


# ==========================================
# COMPLETE
# ==========================================

print("\n================================")
print("GRAD-CAM COMPLETE")
print("================================")

print(
    "Result saved at:"
)

print(
    OUTPUT_PATH
)