import tensorflow as tf
import numpy as np
from PIL import Image


# ==============================
# LOAD TRAINED MODEL
# ==============================

model = tf.keras.models.load_model(
    "model/skin_disease_model.keras"
)


# ==============================
# LOAD CLASS NAMES
# ==============================

with open(
    "model/classes.txt",
    "r"
) as file:

    class_names = [
        line.strip()
        for line in file.readlines()
    ]


# ==============================
# DISEASE NAMES
# ==============================

disease_names = {

    "akiec": "Actinic Keratosis",

    "bcc": "Basal Cell Carcinoma",

    "bkl": "Benign Keratosis",

    "df": "Dermatofibroma",

    "mel": "Melanoma",

    "nv": "Melanocytic Nevus",

    "vasc": "Vascular Lesion"

}


# ==============================
# SHOW CLASSES
# ==============================

print("\nAvailable Classes:")

for class_name in class_names:

    print(
        class_name,
        "->",
        disease_names.get(
            class_name,
            class_name
        )
    )


# ==============================
# GET IMAGE PATH
# ==============================

image_path = input(
    "\nEnter image path: "
)


# ==============================
# LOAD IMAGE
# ==============================

image = Image.open(
    image_path
).convert("RGB")


image = image.resize(
    (224, 224)
)


# ==============================
# PREPARE IMAGE
# ==============================

image_array = np.array(
    image
).astype("float32")


image_array = np.expand_dims(
    image_array,
    axis=0
)


# ==============================
# PREDICT
# ==============================

predictions = model.predict(
    image_array,
    verbose=0
)


predicted_index = np.argmax(
    predictions[0]
)


predicted_class = class_names[
    predicted_index
]


confidence = (
    predictions[0][predicted_index]
    * 100
)


# ==============================
# RESULT
# ==============================

print("\n")
print("================================")
print("       PREDICTION RESULT")
print("================================")

print(
    "\nDisease:",
    disease_names.get(
        predicted_class,
        predicted_class
    )
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

print("\n================================")