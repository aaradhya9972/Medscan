import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load models only once
BASE_DIR = os.path.dirname(__file__)
MODELS = {
    "pneumonia": tf.keras.models.load_model(os.path.join(BASE_DIR, "saved_models", "pneumonia_model.h5")),
    "breast_cancer": tf.keras.models.load_model(os.path.join(BASE_DIR, "saved_models", "breast_cancer_model.h5")),
    # "dental_cavity": tf.keras.models.load_model(os.path.join(BASE_DIR, "saved_models", "dental_cavity_model.h5")),  # later
}

IMG_SIZE = (224, 224)

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

def predict_disease(model_name, img_path):
    if model_name not in MODELS:
        raise ValueError(f"❌ Model '{model_name}' not available.")

    model = MODELS[model_name]
    processed_img = preprocess_image(img_path)

    prediction = model.predict(processed_img)[0][0]
    probability = float(prediction)

    # Convert probability → label
    if probability >= 0.5:
        label = "Positive"
    else:
        label = "Negative"

    return label, probability
