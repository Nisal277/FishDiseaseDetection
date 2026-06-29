import os
import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix




os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)



DATASET_DIR = Path(r"C:\Users\nisal\Downloads\FishDiseaseDetection\cnn_model\dataset")

OUTPUT_DIR = Path("saved_models")
OUTPUT_DIR.mkdir(exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 16

INITIAL_EPOCHS = 40
FINE_TUNE_EPOCHS = 30

LEARNING_RATE = 1e-3
FINE_TUNE_LEARNING_RATE = 1e-5

MODEL_NAME = "fish_disease_cnn_mobilenetv2.keras"
CLASS_NAMES_FILE = OUTPUT_DIR / "class_names.json"




def get_dataset_paths(dataset_dir: Path):
    train_dir = dataset_dir / "train"

    valid_dir = dataset_dir / "valid"
    val_dir = dataset_dir / "val"

    test_dir = dataset_dir / "test"

    if not train_dir.exists():
        raise FileNotFoundError(f"Train folder not found: {train_dir}")

    if valid_dir.exists():
        validation_dir = valid_dir
    elif val_dir.exists():
        validation_dir = val_dir
    else:
        raise FileNotFoundError(
            "Validation folder not found. Expected either 'valid' or 'val'."
        )

    if not test_dir.exists():
        print("[WARNING] Test folder not found. Test evaluation will be skipped.")
        test_dir = None

    return train_dir, validation_dir, test_dir


def count_images_per_class(folder: Path):
    print(f"\n[INFO] Checking folder: {folder}")

    class_folders = sorted([p for p in folder.iterdir() if p.is_dir()])

    total = 0
    for class_folder in class_folders:
        images = list(class_folder.glob("*.*"))
        image_count = len([
            p for p in images
            if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
        ])
        total += image_count
        print(f"  {class_folder.name}: {image_count} images")

    print(f"  Total: {total} images")

    if total == 0:
        raise ValueError(f"No images found in {folder}")




def load_datasets(train_dir, validation_dir, test_dir):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="int",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        validation_dir,
        labels="inferred",
        label_mode="int",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_ds = None
    if test_dir is not None:
        test_ds = tf.keras.utils.image_dataset_from_directory(
            test_dir,
            labels="inferred",
            label_mode="int",
            image_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            shuffle=False
        )

    class_names = train_ds.class_names

    print("\n[INFO] Classes found:")
    for i, name in enumerate(class_names):
        print(f"  {i}: {name}")

    with open(CLASS_NAMES_FILE, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=4, ensure_ascii=False)

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(500).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    if test_ds is not None:
        test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names




def build_model(num_classes):
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.12),
            layers.RandomContrast(0.15),
        ],
        name="data_augmentation"
    )

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )

    base_model.trainable = False

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    x = data_augmentation(inputs)

    # MobileNetV2 expects pixels scaled to [-1, 1]
    x = layers.Rescaling(scale=1.0 / 127.5, offset=-1.0)(x)

    x = base_model(x, training=False)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.35)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="fish_disease_cnn_mobilenetv2")

    return model, base_model




def plot_training_history(history, output_path):
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, acc, label="Training Accuracy")
    plt.plot(epochs, val_acc, label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path / "accuracy_plot.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, loss, label="Training Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path / "loss_plot.png", dpi=200, bbox_inches="tight")
    plt.close()


def merge_histories(history1, history2):
    merged = {}

    for key in history1.history.keys():
        merged[key] = history1.history[key] + history2.history.get(key, [])

    class DummyHistory:
        pass

    dummy = DummyHistory()
    dummy.history = merged

    return dummy




def evaluate_and_save_report(model, dataset, class_names, split_name):
    if dataset is None:
        print(f"[WARNING] No {split_name} dataset found. Skipping.")
        return

    print(f"\n[INFO] Evaluating on {split_name} set...")

    loss, accuracy = model.evaluate(dataset, verbose=1)

    y_true = []
    y_pred = []

    for images, labels in dataset:
        probs = model.predict(images, verbose=0)
        preds = np.argmax(probs, axis=1)

        y_true.extend(labels.numpy())
        y_pred.extend(preds)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0
    )

    print(f"\n{split_name.upper()} Classification Report")
    print(report)

    report_path = OUTPUT_DIR / f"{split_name}_classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"{split_name.upper()} Loss: {loss:.6f}\n")
        f.write(f"{split_name.upper()} Accuracy: {accuracy:.6f}\n\n")
        f.write(report)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"{split_name.upper()} Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{split_name}_confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[INFO] {split_name} accuracy: {accuracy:.4f}")
    print(f"[INFO] Report saved to: {report_path}")




def main():
    print("[INFO] Starting CNN fish disease classification training...")

    train_dir, validation_dir, test_dir = get_dataset_paths(DATASET_DIR)

    count_images_per_class(train_dir)
    count_images_per_class(validation_dir)

    if test_dir is not None:
        count_images_per_class(test_dir)

    train_ds, val_ds, test_ds, class_names = load_datasets(
        train_dir,
        validation_dir,
        test_dir
    )

    num_classes = len(class_names)

    model, base_model = build_model(num_classes)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    best_model_path = OUTPUT_DIR / MODEL_NAME

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=best_model_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=12,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.CSVLogger(
            OUTPUT_DIR / "training_log.csv"
        )
    ]

    print("\n[INFO] Stage 1: Training classifier head...")
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=INITIAL_EPOCHS,
        callbacks=callbacks
    )

    print("\n[INFO] Stage 2: Fine-tuning last CNN layers...")

    base_model.trainable = True

    # Freeze most of MobileNetV2; train only last layers
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    # Keep BatchNorm stable during fine-tuning
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=FINE_TUNE_LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=FINE_TUNE_EPOCHS,
        callbacks=callbacks
    )

    full_history = merge_histories(history1, history2)
    plot_training_history(full_history, OUTPUT_DIR)

    print("\n[INFO] Loading best saved model...")
    best_model = keras.models.load_model(best_model_path)

    evaluate_and_save_report(best_model, val_ds, class_names, "validation")
    evaluate_and_save_report(best_model, test_ds, class_names, "test")

    print("\n[DONE] Training complete.")
    print(f"[INFO] Best model saved at: {best_model_path}")
    print(f"[INFO] Class names saved at: {CLASS_NAMES_FILE}")
    print(f"[INFO] Accuracy/loss plots saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()