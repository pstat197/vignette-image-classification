# ============================================================
# 0. Full reproducibility setup (like set.seed in R)
# ============================================================

import os, random, numpy as np, tensorflow as tf

SEED = 42  # fixed seed value

os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
    print("Deterministic ops enabled.")
except Exception as e:
    print("Deterministic ops not available:", e)

print("Seed fixed:", SEED)
print("TensorFlow version:", tf.__version__)


# ============================================================
# 1. Imports and helper functions
# ============================================================

import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras import layers, models


# -------------------------
# Image preprocessing
# -------------------------
def prepare_dl_data(data_path, target_size=(128, 128)):
    """
    Load JPGs from a folder and return X (images) and y (labels)
    Labels: cat -> 0, dog -> 1
    """
    filenames = [f for f in os.listdir(data_path) if f.lower().endswith(".jpg")]
    if not filenames:
        raise RuntimeError(f"No .jpg files found in {data_path}")

    X, y = [], []

    for filename in filenames:
        img_path = os.path.join(data_path, filename)
        img = Image.open(img_path).convert("RGB").resize(target_size)
        img_array = np.array(img).astype(np.float32) / 255.0

        prefix = filename.split(".")[0].lower()
        if prefix == "cat":
            label = 0
        elif prefix == "dog":
            label = 1
        else:
            continue  # skip unrecognized files

        X.append(img_array)
        y.append(label)

    X = np.stack(X)
    y = np.array(y, dtype=np.int64)
    return X, y


# ============================================================
# 2. Load and split data
# ============================================================

# Assuming notebook is in /notebooks and images in ../data_sample/
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
data_dir = os.path.join(PROJECT_ROOT, "data_sample")

print("Loading data from:", data_dir)
X_full, y_full = prepare_dl_data(data_dir, target_size=(128, 128))
print("Loaded:", X_full.shape, y_full.shape)
print("Class counts:", np.bincount(y_full))

# Split into train / val / test
X_train, X_temp, y_train, y_temp = train_test_split(
    X_full, y_full, test_size=0.4, stratify=y_full, random_state=SEED
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=SEED
)

print("Train size:", len(y_train))
print("Val size:", len(y_val))
print("Test size:", len(y_test))
print("Val label counts:", np.bincount(y_val))


# ============================================================
# 3. Build CNN model
# ============================================================

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

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)


# ============================================================
# 4. Train model (with early stopping)
# ============================================================

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
    X_train,
    y_train.astype(np.float32),
    epochs=30,
    batch_size=16,
    validation_data=(X_val, y_val.astype(np.float32)),
    callbacks=[early_stop, reduce_lr],
    verbose=1,
)


# ============================================================
# 5. Plot training vs validation accuracy/loss
# ============================================================

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


plot_history(history)


# ============================================================
# 6. Evaluate on test set
# ============================================================

test_loss, test_acc = model.evaluate(X_test, y_test.astype(np.float32), verbose=0)
print(f"\nTest loss: {test_loss:.4f}")
print(f"Test accuracy: {test_acc * 100:.2f}%")

# Predictions
y_prob = model.predict(X_test).flatten()
y_pred = (y_prob >= 0.5).astype(int)

print("\nConfusion matrix (test):")
print(confusion_matrix(y_test, y_pred))
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=["cat", "dog"]))


# ============================================================
# 7. Visualize predictions
# ============================================================

def show_examples(X, y_true, y_pred, y_prob, n=6):
    label_map = {0: "cat", 1: "dog"}
    n = min(n, len(X))
    indices = np.random.choice(len(X), n, replace=False)
    plt.figure(figsize=(12, 4))
    for i, idx in enumerate(indices):
        img = X[idx]
        plt.subplot(1, n, i + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(
            f"T: {label_map[int(y_true[idx])]} \nP: {label_map[int(y_pred[idx])]} \n{y_prob[idx]:.2f}"
        )
    plt.tight_layout()
    plt.show()

show_examples(X_test, y_test, y_pred, y_prob, n=6)
