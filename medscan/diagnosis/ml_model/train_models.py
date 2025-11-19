import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# Common params
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5   # keep small first for testing
SEED = 42

# Datasets paths (you need to download datasets into these folders)
DATASETS = {
    "pneumonia": "datasets/pneumonia",
    "breast_cancer": "datasets/breast_cancer",
    "dental_cavity": "datasets/dental_cavity"
}

def build_model(input_shape=(224, 224, 3)):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2,2),

        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),

        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(name="auc")])
    return model

def load_data(base_path):
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_gen = datagen.flow_from_directory(
        os.path.join(base_path, "train"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset="training",
        seed=SEED
    )

    val_gen = datagen.flow_from_directory(
        os.path.join(base_path, "train"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset="validation",
        seed=SEED
    )

    test_gen = datagen.flow_from_directory(
        os.path.join(base_path, "test"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        seed=SEED
    )

    return train_gen, val_gen, test_gen


if __name__ == "__main__":
    os.makedirs("diagnosis/ml_model/saved_models", exist_ok=True)

    for disease, path in DATASETS.items():
        print(f"\n🔹 Training model for {disease.upper()}...\n")

        train_gen, val_gen, test_gen = load_data(path)
        model = build_model()

        history = model.fit(
            train_gen,
            epochs=EPOCHS,
            validation_data=val_gen
        )

        print("\n✅ Evaluating on test set...")
        loss, acc, auc = model.evaluate(test_gen)
        print(f"Test Accuracy: {acc:.2f}, AUC: {auc:.2f}")

        # Save model
        model.save(f"diagnosis/ml_model/saved_models/{disease}_model.h5")
        print(f"💾 Saved {disease} model to diagnosis/ml_model/saved_models/{disease}_model.h5")

callbacks = [
    EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    ModelCheckpoint(f"diagnosis/ml_model/saved_models/{disease}_best.keras", save_best_only=True)
]

history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=callbacks
)
