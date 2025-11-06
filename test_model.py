import os, numpy as np
import tensorflow as tf
from tensorflow import keras

IMG_SIZE = (500, 500)
MODEL_PATH = "outputs/models/baseline.keras"   
TRAIN_DIR = os.path.join("data", "Train")      

# --- get class order EXACTLY as used in training ---
_tmp = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR, image_size=IMG_SIZE, batch_size=1, shuffle=False, label_mode="categorical"
)
CLASS_NAMES = _tmp.class_names
del _tmp
print("CLASS_NAMES detected from Train folder:", CLASS_NAMES)

def load_and_prep(img_path):
 
    img = keras.utils.load_img(img_path, target_size=IMG_SIZE)
    x = keras.utils.img_to_array(img)   # 0..255
    x = np.expand_dims(x, axis=0)
    return x

def predict_image(model, img_path):
    x = load_and_prep(img_path)
    probs = model.predict(x, verbose=0)[0]        # softmax output
    idx = int(np.argmax(probs))
    return CLASS_NAMES[idx], float(probs[idx]), probs

if __name__ == "__main__":
    model = keras.models.load_model(MODEL_PATH)

    tests = [
        "data/Test/crack/test_crack.jpg",
        "data/Test/missing-head/test_missinghead.jpg",
        "data/Test/paint-off/test_paintoff.jpg",
    ]

    for p in tests:
        label, conf, probs = predict_image(model, p)
        print(f"{os.path.basename(p)} -> {label} ({conf:.2%}) | probs={np.round(probs,3)}")
