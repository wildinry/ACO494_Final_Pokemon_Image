## ✨ Pokémon Type Classifier (CNN Model)

| 🎯 **Task** | **Model** | **Libraries** |
| :---: | :---: | :---: |
| Image Classification | Convolutional Neural Network ($\text{CNN}$) | TensorFlow, Keras, NumPy, Pandas |

This project trains a **Convolutional Neural Network (CNN)** to classify Pokémon based on their primary type ($\text{Type1}$) using image data.

---

### 🌟 Features

* 🖼️ **Image Preprocessing:** Efficiently loads and processes Pokémon sprite images for $\text{CNN}$ input.
* 🏷️ **Label Encoding:** Handles categorical $\text{Type1}$ labels for multi-class classification training.
* 🧠 **Deep Learning:** Trains a robust $\text{CNN}$ model built with **TensorFlow/Keras**.
* 💾 **Persistence:** Automatically saves the trained model (`pokemon_model.h5`) and the label mapping (`label_classes.npy`).
* 🔮 **Prediction Utility:** Includes a dedicated script (`predict.py`) for easy classification of new images.

---

### ⚙️ Getting Started

Follow these steps to set up your environment and prepare the training data.

#### 1. Clone the Repository

```bash
git clone [https://github.com/YOUR_USERNAME/YOUR_REPO.git](https://github.com/YOUR_USERNAME/YOUR_REPO.git)
cd YOUR_REPO
