import os
import pandas as pd
from PIL import Image
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D

# -----------------------------
# Setup M2 Configuration (Kept for clarity)
# -----------------------------
# We rely on your installed 'tensorflow-metal' dependencies here.
# Explicitly checking available devices to confirm a device is visible
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"TensorFlow found GPU device(s): {gpus}")
    else:
        print("TensorFlow did not find a GPU device. Running on CPU.")
except Exception as e:
    print(f"Error checking GPU devices: {e}")
    pass

# -----------------------------
# ... (Data Loading and Preprocessing remains the same)
# -----------------------------
df = pd.read_csv("pokemon.csv")
print("Loaded CSV rows:", len(df))

image_dir = "images copy"
images = []
labels = []

for idx, row in df.iterrows():
    name = row["Name"].lower()       
    img_filename = name + ".png"        
    img_path = os.path.join(image_dir, img_filename)

    if not os.path.exists(img_path):
        continue

    img = Image.open(img_path).convert("RGB").resize((224, 224))
    img = np.array(img) / 255.0      
    images.append(img)
    labels.append(row["Type1"])        

print("Loaded images:", len(images))

X = np.array(images)
y = np.array(labels)
print("Dataset shape:", X.shape, y.shape)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_cat = to_categorical(y_encoded)
NUM_CLASSES = len(encoder.classes_)
print("Number of classes:", NUM_CLASSES)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42, shuffle=True
)

# --- CRITICAL STABILITY FIX ---
# 1. Ensure input data is float32 (as before)
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

# 2. Convert NumPy arrays to TensorFlow Dataset objects.
# This optimizes data piping for the M2/MPS backend.
BATCH_SIZE = 32

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)) \
    .shuffle(buffer_size=1024) \
    .batch(BATCH_SIZE) \
    .prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)) \
    .batch(BATCH_SIZE) \
    .prefetch(tf.data.AUTOTUNE)

print("Data converted to TensorFlow Dataset objects.")
# ------------------------------

# -----------------------------
# VGG16 Transfer Learning Model (No Change)
# -----------------------------
base_model = VGG16(
    weights='imagenet', 
    include_top=False, 
    input_shape=(224, 224, 3)
)

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x) 
x = Dense(256, activation='relu')(x) 
x = Dropout(0.5)(x) 
predictions = Dense(NUM_CLASSES, activation='softmax')(x) 

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# -----------------------------
# Train model (Using the Dataset objects)
# -----------------------------
print("\nStarting training with TensorFlow Dataset stability fix...")
# Pass the dataset object directly instead of X_train and y_train
model.fit(
    train_dataset, 
    validation_data=test_dataset,
    epochs=30
)

# -----------------------------
# Evaluate model on test set
# -----------------------------
# Evaluate using the test dataset
test_loss, test_accuracy = model.evaluate(test_dataset, verbose=1)

print("\n--- FINAL RESULTS ---")
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
print("---------------------")

# -----------------------------
# Save model + label encoder
# -----------------------------
model.save("pokemon_vgg16_dataset_model.h5")
np.save("label_classes.npy", encoder.classes_)

print("Model and label classes saved!")