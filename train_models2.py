# train_models2.py
import json, os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import matplotlib.pyplot as plt


# ---------- config ----------
IMG_SIZE = (500, 500)
BATCH = 8
SEED = 42
NUM_CLASSES = 3
DATA_DIR = "data"

# ---------- datasets ----------
train_dir = os.path.join(DATA_DIR, "Train")
val_dir   = os.path.join(DATA_DIR, "Validation")

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir, labels="inferred", label_mode="categorical",
    image_size=IMG_SIZE, batch_size=BATCH, seed=SEED
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir, labels="inferred", label_mode="categorical",  # keep categorical
    image_size=IMG_SIZE, batch_size=BATCH, seed=SEED
)



# cache/prefetch
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds   = val_ds.cache().prefetch(AUTOTUNE)

# preprocessing
data_augment = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.02),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.05),
])
rescale = layers.Rescaling(1./255)

# ---------- models ----------
def build_baseline(input_shape=(500,500,3), num_classes=3):
    inputs = keras.Input(shape=input_shape)
    x = rescale(inputs)
    x = data_augment(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x); x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x); x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(96,3, padding="same", activation="relu")(x); x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.45)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="baseline_cnn")

def build_stronger(input_shape=(500,500,3), num_classes=3, l2=1e-4):
    inputs = keras.Input(shape=input_shape)
    x = rescale(inputs)
    x = data_augment(x)

    def block(x, f):
        x = layers.Conv2D(f, 3, padding="same", kernel_regularizer=regularizers.l2(l2), use_bias=False)(x)
        x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)
        x = layers.Conv2D(f, 3, padding="same", kernel_regularizer=regularizers.l2(l2), use_bias=False)(x)
        x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.3)(x)
        return x

    x = block(x, 32)
    x = block(x, 64)
    x = block(x, 128)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="stronger_cnn")

# ---------- training helpers ----------
def compile_and_train(model, train_ds, val_ds, lr=1e-3, epochs=20, out_stub="baseline"):
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    os.makedirs("outputs/models", exist_ok=True)
    callbacks = [
        keras.callbacks.ModelCheckpoint(f"outputs/models/{out_stub}.keras",
                                        monitor="val_accuracy", save_best_only=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1),
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=2, callbacks=callbacks)
    with open(f"outputs/history_{out_stub}.json", "w") as f:
        import json; json.dump(history.history, f)
    return history

import matplotlib.pyplot as plt

def save_perf_plot(history_like, out_base):
    # Accept a dict or a History object
    h = history_like.history if hasattr(history_like, "history") else history_like

    # Handle different metric key names
    acc_key     = "accuracy" if "accuracy" in h else "categorical_accuracy"
    val_acc_key = "val_accuracy" if "val_accuracy" in h else "val_categorical_accuracy"

    # Accuracy
    plt.figure()
    plt.plot(h[acc_key], label="train_acc")
    plt.plot(h[val_acc_key], label="val_acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Accuracy"); plt.legend()
    plt.savefig(out_base.replace(".png", "_acc.png"), bbox_inches="tight"); plt.close()

    # Loss
    plt.figure()
    plt.plot(h["loss"], label="train_loss")
    plt.plot(h["val_loss"], label="val_loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss"); plt.legend()
    plt.savefig(out_base.replace(".png", "_loss.png"), bbox_inches="tight"); plt.close()
import json, os

# Load and plot baseline history (exists)
if os.path.exists("outputs/history_baseline.json"):
    with open("outputs/history_baseline.json") as f:
        h_a = json.load(f)
    save_perf_plot(h_a, "outputs/perf_baseline.png")

# Load and plot stronger history if present
if os.path.exists("outputs/history_baseline.json"):
    with open("outputs/history_baseline.json") as f:
        h_b = json.load(f)
    save_perf_plot(h_b, "outputs/perf_baseline.png")

# ---------- main ----------
if __name__ == "__main__":
    baseline = build_baseline()
    hist_a = compile_and_train(baseline, train_ds, val_ds, lr=1e-3, epochs=20, out_stub="baseline")
    save_perf_plot(hist_a.history, "outputs/perf_baseline.png")
    stronger = build_stronger()
    hist_b = compile_and_train(stronger, train_ds, val_ds, lr=7e-4, epochs=25, out_stub="stronger")
    save_perf_plot(hist_b.history, "outputs/perf_stronger.png")