import os
import pandas as pd
from PIL import Image
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import pandas as pd

# -----------------------------
# Load CSV
# -----------------------------
df = pd.read_csv("pokemon.csv")
print("Loaded CSV rows:", len(df))
print(df.head())

# -----------------------------
# Setup image directory
# -----------------------------
image_dir = "images"

images = []
labels = []

label_set = list()
edge_crop = 20 # pixels
target_dim = 120
input_dim = 120
crop_box = (edge_crop, edge_crop, target_dim-edge_crop, target_dim-edge_crop)

# for the multiclass classification encoding
def get_encoding(lbls):
    encoding = np.zeros(len(label_set))
    for idx, lbl in enumerate(label_set):
        if (lbl in lbls):
            encoding[idx] = 1
    # normalize due to softmax output
    return encoding/len(lbls)

# -----------------------------
# Load images based on Name
# -----------------------------

for idx, row in df.iterrows():
    name = row["Name"].lower()       
    img_filename = name + ".png"        
    img_path = os.path.join(image_dir, img_filename)

    if not os.path.exists(img_path):
        print("Missing image:", img_path)
        continue

    img = Image.open(img_path).convert("RGBA")
    # img = img.crop(crop_box)

    flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
    img = np.array(img) 
    flipped_img = np.array(flipped_img)

    images.append(img)
    images.append(flipped_img)
    lbl = [row["Type1"]]

    # for the label_set, ignore
    if (not row["Type1"] in label_set):
        label_set.append(row["Type1"])

    # count the second type as well
    if type(row["Type2"]) is str:
        lbl.append(row["Type2"])
        if (not row["Type2"] in label_set):
            label_set.append(row["Type2"])
    lbl = get_encoding(lbl)
    labels.append(lbl)        
    labels.append(lbl)        

print("Loaded images:", len(images))

# Convert to arrays
# -----------------------------
# Label encode types
# -----------------------------
X = np.array(images)
y = np.array(labels)
# y = np.array(list(map(get_encoding, labels)))

print("Dataset shape:", X.shape, y.shape)
print(y)

# encoder = LabelEncoder()
# y_encoded = encoder.fit_transform(y)
# y_cat = to_categorical(y_encoded)

print("Number of classes:", len(label_set))

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=67, shuffle=True
)

# -----------------------------
# Simple CNN model
# -----------------------------
base = 64
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(base, (3,3), activation="relu", input_shape=(input_dim,input_dim,4)),
    tf.keras.layers.Conv2D(base, (3,3), activation="relu"),
    tf.keras.layers.Conv2D(base, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(base*2, (3,3), activation="relu"),
    tf.keras.layers.Conv2D(base*2, (3,3), activation="relu"),
    tf.keras.layers.Conv2D(base*2, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(base*2, (3,3), activation="relu"),
    tf.keras.layers.Conv2D(base*2, (3,3), activation="relu"),
    tf.keras.layers.Conv2D(base*2, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    # tf.keras.layers.AveragePooling2D(),


    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(2048, activation="relu"),
    tf.keras.layers.Dense(2048, activation="relu"),
    tf.keras.layers.Dense(len(label_set), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()
# -----------------------------
# Train model
# -----------------------------
history = model.fit(
    X_train, 
    y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=32
)

import matplotlib.pyplot as plt
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy'] # If you used validation data
epochs = range(1, len(train_accuracy) + 1)
plt.plot(epochs, train_accuracy, 'b', label='Training Accuracy')
plt.plot(epochs, val_accuracy, 'r', label='Validation Accuracy') # If you used validation data
plt.title('Training and Validation Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# -----------------------------
# Evaluate model on test set
# -----------------------------
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
# Save model + label encoder
# -----------------------------
# model.save("pokemon_model.h5")
# np.save("label_classes.npy", len(label_set))

print("Model and label classes saved!")
