import tensorflow as tf

DATASE_PATH = "dataset/train"

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASE_PATH,
    image_size=(224, 224),
    batch_size=16
)

print("\n classes found:")
print(train_ds.class_names)
print("\n number of classes:")
print(len(train_ds.class_names))
