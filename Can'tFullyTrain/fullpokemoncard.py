import os
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf

# ============================================================
# Configuration & Setup
# ============================================================
# This is the directory containing your Pokemon card images
image_dir = "images" 
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.0005
VALID_EXTENSIONS = ('.png', '.jpg', '.jpeg')

images = []
labels = []

# ============================================================
# Load and Preprocess Images (Label extraction from filename)
# ============================================================
print("Starting to load and preprocess images...")

if not os.path.isdir(image_dir):
    print(f"Error: The directory '{image_dir}' was not found.")
    exit()

# Iterate over every file in the image directory
for filename in os.listdir(image_dir):
    # 1. Check for valid file extensions
    if not filename.lower().endswith(VALID_EXTENSIONS):
        continue

    # 2. Extract label (Pokemon name) from the filename
    # ASSUMPTION: The label (Pokemon name) is the part before the first underscore or period
    try:
        # Example: 'Charizard_Fire_Holo.png' -> 'charizard'
        base_name = filename.lower().split('.')[0]
        pokemon_name = base_name.split('_')[0] 
        
        if not pokemon_name:
             # Skip if the extracted name is empty
            continue

    except IndexError:
        print(f"Skipping file with unclear name format: {filename}")
        continue

    # 3. Construct the full file path
    path = os.path.join(image_dir, filename)

    # 4. Load and preprocess the image
    try:
        # Load, convert to RGB, and resize for MobileNetV2
        img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        # Normalize the pixel values to [0, 1]
        img = np.array(img) / 255.0  
        
        images.append(img)
        labels.append(pokemon_name) # Use the extracted name as the label
    except Exception as e:
        print(f"Error loading {filename}: {e}")

images = np.array(images)
labels = np.array(labels)

print("\n--- Data Summary ---")
print("Images loaded:", len(images))

if len(images) == 0:
    print("No images were successfully loaded. Cannot proceed with training.")
    exit()

print("Dataset shape:", images.shape)

# ============================================================
# Encode labels
# ============================================================
encoder = LabelEncoder()
y = encoder.fit_transform(labels)
num_classes = len(encoder.classes_)

# Convert integer labels to one-hot encoding
y_cat = tf.keras.utils.to_categorical(y, num_classes=num_classes)

print("Class count:", num_classes)
print("Classes:", encoder.classes_)

# ============================================================
# Train/val/test split
# ============================================================
# Split 70% Train, 15% Validation, 15% Test
X_train, X_temp, y_train, y_temp = train_test_split(
    images, y_cat, test_size=0.30, random_state=42, shuffle=True
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, shuffle=True
)

print("Train set shape:", X_train.shape)
print("Validation set shape:", X_val.shape)
print("Test set shape:", X_test.shape)

# ============================================================
# Data Augmentation
# ============================================================
data_aug = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.10),
    tf.keras.layers.RandomZoom(0.1),
])

# ============================================================
# Transfer Learning Model (MobileNetV2)
# ============================================================
# Load the MobileNetV2 base model, excluding the top classification layer
base = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)
base.trainable = False  # Freeze the pre-trained base layers

# Build the final model
model = tf.keras.Sequential([
    data_aug,
    base,
    tf.keras.layers.GlobalAveragePooling2D(), # Reduces spatial dimensions
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(num_classes, activation="softmax") # Final classification layer
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ============================================================
# Callbacks
# ============================================================
callbacks = [
    # Stop training if validation loss doesn't improve for 5 epochs
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    ),
    # Save the model only when validation loss is at its lowest
    tf.keras.callbacks.ModelCheckpoint(
        "pokemon_model_best.h5",
        save_best_only=True,
        monitor="val_loss"
    )
]

# ============================================================
# Train
# ============================================================
print("\n--- Starting Training ---")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks
)
print("Training complete.")

# ============================================================
# Evaluate on Test Set
# ============================================================
print("\n--- Evaluating on Test Set ---")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print("Test Accuracy:", test_acc)
print("Test Loss:", test_loss)

# ============================================================
# Save Final Model and Label Classes
# ============================================================
model.save("pokemon_model_final.h5")
np.save("label_classes.npy", encoder.classes_)
print("Model and label classes saved!")