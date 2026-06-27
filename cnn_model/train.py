import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
import numpy as np
import random
import json
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt



# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)




# Project Paths
DATASET_DIR = Path("dataset")
MODEL_DIR = Path("saved_models")
OUTPUT_DIR = Path("outputs")
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
BEST_MODEL_PATH = MODEL_DIR / "best_fish_disease_model.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"

# plot function
def plot_training_history(history, output_dir):

    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs = range(1, len(acc) + 1)

    # Accuracy Plot
    plt.figure(figsize=(8,5))
    plt.plot(epochs, acc, label="Training Accuracy")
    plt.plot(epochs, val_acc, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / "accuracy_plot.png")
    plt.close()

    # Loss Plot
    plt.figure(figsize=(8,5))
    plt.plot(epochs, loss, label="Training Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / "loss_plot.png")
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





# training confifuration
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 20





# Load datasets (training and validation)
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR / "train",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR / "valid",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_ds.class_names

print("\nClasses Found:")
print(class_names)

print("\nNumber of Classes:")
print(len(class_names))

# Save class names
with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as file:
    json.dump(class_names, file, indent=4)

print("Class names saved successfully.")





# Data augmentation
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])




# Load pretrained MobileNetV2
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)




# Freeze pretrained layers
base_model.trainable = False




# Build CNN Model Architecture
inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

x = data_augmentation(inputs)

x = layers.Rescaling(1.0 / 127.5, offset=-1)(x)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.3)(x)

outputs = layers.Dense(
    len(class_names),
    activation="softmax"
)(x)

model = keras.Model(inputs, outputs)

model.summary()





# training callbacks
callbacks = [

    keras.callbacks.ModelCheckpoint(
        filepath=BEST_MODEL_PATH,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1
    )

]




# stage 1 compile model
model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nModel compiled successfully.")

print("\nStarting training...")



# stage 1 - train Classification Head
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

print("\nTraining completed.")



# save stage 1 model
model.save(
    "saved_models/stage1_model.keras"
)
print("Model saved successfully.")





# stage 2 - fine tuning
print("\nStarting fine-tuning...")
base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

# Keep BatchNormalization layers frozen
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

# recompile model with a lower learning rate
model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.00001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)



# stage 2 - fine-tune MobileNetv2
history_finetune = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=callbacks
)


# save final fine tuned model
model.save(
    "saved_models/fish_disease_model.keras"
)

full_history = merge_histories(history, history_finetune)
plot_training_history(full_history, OUTPUT_DIR)
print("\nFine-tuned model saved successfully.")
print("Training graphs saved successfully.")

