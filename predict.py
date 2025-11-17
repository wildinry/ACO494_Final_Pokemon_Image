import tensorflow as tf
import numpy as np
from PIL import Image

# Load model + labels
model = tf.keras.models.load_model("pokemon_model.h5")
label_classes = np.load("label_classes.npy")

def predict_image(path):
    img = Image.open(path).convert("RGB").resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    pred_index = np.argmax(pred, axis=1)[0]
    pred_label = label_classes[pred_index]

    print(f"{path} → Predicted Type1: {pred_label}")
    return pred_label

# Example usage:
predict_image("images/pikachu.png")
