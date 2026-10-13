import tensorflow as tf

model = tf.keras.models.load_model(
    "model/skin_disease_model.keras"
)

print("\nMODEL LAYERS:\n")

for i, layer in enumerate(model.layers):

    print(
        i,
        "->",
        layer.name,
        "->",
        type(layer).__name__
    )