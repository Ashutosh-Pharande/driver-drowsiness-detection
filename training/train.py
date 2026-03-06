from pathlib import Path
import tensorflow as tf
from training.model import build_model


def main():

    project_root = Path(__file__).resolve().parents[1]

    train_dir = project_root / "dataset" / "train"
    test_dir = project_root / "dataset" / "test"

    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "cnn_model.h5"

    batch_size = 32
    img_height, img_width = 24, 24
    epochs = 25   # Increased epochs

    # Improved augmentation
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=15,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=[0.7, 1.3],
        horizontal_flip=True,
    )

    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255.0
    )

    train_generator = train_datagen.flow_from_directory(
        directory=str(train_dir),
        target_size=(img_height, img_width),
        color_mode="grayscale",
        batch_size=batch_size,
        class_mode="binary",
        shuffle=True,
    )

    validation_generator = test_datagen.flow_from_directory(
        directory=str(test_dir),
        target_size=(img_height, img_width),
        color_mode="grayscale",
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False,
    )

    model = build_model()

    model.summary()

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
    )

    print("Final Training Accuracy:", history.history["accuracy"][-1])
    print("Final Validation Accuracy:", history.history["val_accuracy"][-1])

    model.save(str(model_path))
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()