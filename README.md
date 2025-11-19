# MedScan — Diagnosis ML Models

Brief project README for MedScan, a small medical image diagnosis utility using TensorFlow Keras models.

## Project structure (relevant)
- a:\Python\medscan\
  - diagnosis\ml_model\predict.py        # prediction helper that loads models at import
  - diagnosis\ml_model\saved_models\     # place model .h5 files here (e.g. pneumonia_model.h5)

## Purpose
This project loads pre-trained Keras models and provides a simple function to predict disease labels (Positive/Negative) and probabilities from input images.

## Requirements
- Python 3.8+
- TensorFlow (compatible version for your models, e.g. tensorflow>=2.4)
- numpy
- pillow

Install basic deps:
```
pip install tensorflow numpy pillow
```

## How it works
- Models are loaded once at module import from the `saved_models` folder.
- Images are resized to 224×224 and normalized to [0,1].
- Prediction returns a label (`"Positive"` or `"Negative"`) and a probability (float).

## Usage example (Python)
```python
from diagnosis.ml_model import predict

label, prob = predict.predict_disease("pneumonia", r"C:\path\to\image.jpg")
print(label, prob)
```

Notes:
- Valid model names by default: `pneumonia`, `breast_cancer`. To add more, place a `.h5` model into `diagnosis\ml_model\saved_models` and register it in the `MODELS` dict in `predict.py`.
- Because models are loaded at import time, ensure model files exist before importing the module to avoid import-time errors.

## Adding new models
1. Save your model as `your_model_name_model.h5` into:
   `a:\Python\medscan\diagnosis\ml_model\saved_models\`
2. Edit `predict.py` and add an entry to the `MODELS` dict mapping a friendly name to the loaded model file:
```python
# ...existing code...
MODELS = {
    "pneumonia": tf.keras.models.load_model(os.path.join(BASE_DIR, "saved_models", "pneumonia_model.h5")),
    "breast_cancer": tf.keras.models.load_model(os.path.join(BASE_DIR, "saved_models", "breast_cancer_model.h5")),
    "your_model_name": tf.keras.models.load_model(os.path.join(BASE_DIR, "saved_models", "your_model_name_model.h5")),
}
# ...existing code...
```

## Troubleshooting
- Import errors on startup usually mean a model file is missing or incompatible — verify model path and TensorFlow version.
- For large models, consider lazy-loading models instead of importing them at module import to reduce startup memory.

License: add your project's license as needed.
