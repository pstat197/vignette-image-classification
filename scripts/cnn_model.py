import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

import random # Ensure random is imported for the seed setup
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

tf.config.experimental.enable_op_determinism()
print("Deterministic ops enabled.")

print("Seed fixed:", SEED)
print("TensorFlow version:", tf.__version__)

# [MODIFIED] Configuration updated for full dataset
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32 
CHANNELS = 3
DATA_DIR = "../data/train" 

 
def create_dataframe(data_dir):
    """
    Creates a DataFrame with filenames and labels for the generator.
    This allows utilizing flow_from_dataframe for memory efficiency.
    """
    filenames = os.listdir(data_dir)
    categories = []
    valid_files = []

    for filename in filenames:
        if filename.lower().endswith(".jpg"):
            # Map 'dog' to 1 and 'cat' to 0 strings
            category = filename.split('.')[0].lower()
            if category == 'dog':
                categories.append("1") 
            else:
                categories.append("0")
            valid_files.append(filename)

    df = pd.DataFrame({
        'filename': valid_files,
        'category': categories
    })
    return df

 
def build_model():
    input_shape = (128, 128, 3)

    data_augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ],
        name="data_augmentation",
    )

    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            data_augmentation,

            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),

            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(1, activation="sigmoid"),  # binary output
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    
    return model

 
def plot_history(history_obj):
    acc = history_obj.history["accuracy"]
    val_acc = history_obj.history["val_accuracy"]
    loss = history_obj.history["loss"]
    val_loss = history_obj.history["val_loss"]
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, acc, "o-", label="Train acc")
    plt.plot(epochs, val_acc, "o-", label="Val acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, loss, "o-", label="Train loss")
    plt.plot(epochs, val_loss, "o-", label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

 
df = create_dataframe(DATA_DIR)
print(f"Total images found: {df.shape[0]}")
train_df, temp_df = train_test_split(df, test_size=0.4, stratify=df['category'], random_state=SEED)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['category'], random_state=SEED)

print(f"Train size: {train_df.shape[0]}")
print(f"Val size: {val_df.shape[0]}")
print(f"Test size: {test_df.shape[0]}")

 
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

 
train_generator = train_datagen.flow_from_dataframe(
    train_df, 
    DATA_DIR, 
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='binary',
    batch_size=16
)

 
validation_generator = val_datagen.flow_from_dataframe(
    val_df, 
    DATA_DIR, 
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='binary',
    batch_size=16
)

 
test_generator = test_datagen.flow_from_dataframe(
    test_df,
    DATA_DIR,
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='binary',
    batch_size=16,
    shuffle=False
)

 
model = build_model()
model.summary()

 
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    verbose=1,
)

 
history = model.fit(
    train_generator,
    epochs=5,
    validation_data=validation_generator,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

plot_history(history)

 
test_loss, test_acc = model.evaluate(test_generator, verbose=0)
print(f"\nTest loss: {test_loss:.4f}")
print(f"Test accuracy: {test_acc * 100:.2f}%")

 
y_true = test_generator.classes
y_prob = model.predict(test_generator).flatten()
y_pred = (y_prob >= 0.5).astype(int)

print("\nConfusion matrix (test):")
y_true = np.array(y_true).astype(int)
print(confusion_matrix(y_true, y_pred))
print("\nClassification report:")
print(classification_report(y_true, y_pred, target_names=["cat", "dog"]))


