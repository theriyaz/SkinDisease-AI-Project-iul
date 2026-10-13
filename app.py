from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

IMAGE_SIZE = 224

MODEL_PATH = "model/skin_disease_model.keras"

UPLOAD_FOLDER = "uploads"

GRADCAM_FOLDER = "static/gradcam"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRADCAM_FOLDER, exist_ok=True)


# ==========================================
# LOAD MODEL
# ==========================================

print("Loading AI model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("AI model loaded successfully!")


# ==========================================
# MODEL LAYERS
# ==========================================

augmentation_model = model.layers[0]

efficientnet = model.layers[1]

gap_layer = model.layers[2]

dropout_layer = model.layers[3]

dense_layer = model.layers[4]


# ==========================================
# CLASSES
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
# HOME
# ==========================================

@app.route("/")
def home():

    return render_template("index.html")


# ==========================================
# GRAD-CAM FUNCTION
# ==========================================

def generate_gradcam(image_array, predicted_index):

    image_input = np.expand_dims(
        image_array,
        axis=0
    )

    with tf.GradientTape() as tape:

        augmented_image = augmentation_model(
            image_input,
            training=False
        )

        conv_output = efficientnet(
            augmented_image,
            training=False
        )

        tape.watch(conv_output)

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

        class_score = predictions[
            0,
            predicted_index
        ]

    gradients = tape.gradient(
        class_score,
        conv_output
    )

    if gradients is None:

        return None

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2)
    )

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
        tf.reduce_max(heatmap)
        + 1e-8
    )

    return heatmap.numpy()


# ==========================================
# PREDICT
# ==========================================

@app.route(
    "/predict",
    methods=["POST"]
)

def predict():

    if "image" not in request.files:

        return jsonify({
            "error": "No image uploaded"
        })


    file = request.files["image"]


    if file.filename == "":

        return jsonify({
            "error": "Please select an image"
        })


    # ======================================
    # SAVE UPLOADED IMAGE
    # ======================================

    image_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        file.filename
    )

    file.save(image_path)


    # ======================================
    # LOAD IMAGE
    # ======================================

    original_image = Image.open(
        image_path
    ).convert("RGB")


    original_image = original_image.resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    )


    image_array = np.array(
        original_image
    ).astype("float32")


    # ======================================
    # PREDICTION
    # ======================================

    image_input = np.expand_dims(
        image_array,
        axis=0
    )


    predictions = model.predict(
        image_input,
        verbose=0
    )


    predicted_index = int(
        np.argmax(predictions[0])
    )


    predicted_class = class_names[
        predicted_index
    ]


    confidence = (
        predictions[0][predicted_index]
        * 100
    )


    # ======================================
    # GRAD-CAM
    # ======================================

    heatmap = generate_gradcam(
        image_array,
        predicted_index
    )


    gradcam_url = None


    if heatmap is not None:

        # Convert heatmap to image

        heatmap_image = (
            heatmap * 255
        ).astype(
            np.uint8
        )


        heatmap_pil = Image.fromarray(
            heatmap_image
        )


        heatmap_pil = heatmap_pil.resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        )


        # Create colored heatmap using matplotlib

        import matplotlib

        matplotlib.use("Agg")

        import matplotlib.pyplot as plt


        plt.figure(
            figsize=(5, 5)
        )


        plt.imshow(
            original_image
        )


        plt.imshow(
            heatmap_pil,
            cmap="jet",
            alpha=0.45
        )


        plt.axis("off")


        gradcam_filename = (
            "gradcam_result.jpg"
        )


        gradcam_path = os.path.join(
            GRADCAM_FOLDER,
            gradcam_filename
        )


        plt.savefig(
            gradcam_path,
            bbox_inches="tight",
            pad_inches=0,
            dpi=150
        )


        plt.close()


        gradcam_url = (
            "/static/gradcam/"
            + gradcam_filename
            + "?v="
            + str(os.path.getmtime(
                gradcam_path
            ))
        )


    # ======================================
    # RETURN RESULT
    # ======================================

    return jsonify({

        "disease":
            disease_names[
                predicted_class
            ],

        "class":
            predicted_class,

        "confidence":
            round(
                float(confidence),
                2
            ),

        "gradcam":
            gradcam_url

    })


# ==========================================
# RUN
# ==========================================

if __name__ == "__main__":

   app.run(host="0.0.0.0", port=5000, debug=True)