import os
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"




MODEL_PATH = Path(r"saved_models\fish_disease_cnn_mobilenetv2.keras")
CLASS_NAMES_PATH = Path(r"saved_models\class_names.json")

IMAGE_PATH = Path(r"cotton-wool-disease.jpg")

IMG_SIZE = 224




def load_image_for_prediction(image_path):
    img = tf.keras.utils.load_img(
        image_path,
        target_size=(IMG_SIZE, IMG_SIZE)
    )

    img_array = tf.keras.utils.img_to_array(img)

    # Shape: 1, 224, 224, 3
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(f"Class names file not found: {CLASS_NAMES_PATH}")

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    model = keras.models.load_model(MODEL_PATH)

    image = load_image_for_prediction(IMAGE_PATH)

    probs = model.predict(image, verbose=0)[0]

    top_index = int(np.argmax(probs))
    top_class = class_names[top_index]
    top_confidence = float(probs[top_index])

    print("\nFish Disease Prediction")
    print("=======================")
    print(f"Predicted class : {top_class}")
    print(f"Confidence      : {top_confidence * 100:.2f}%")

    print("\nTop predictions:")
    sorted_indexes = np.argsort(probs)[::-1]

    for idx in sorted_indexes[:5]:
        print(f"{class_names[int(idx)]}: {float(probs[idx]) * 100:.2f}%")


if __name__ == "__main__":
    main()