"""Train the waveform-image classifier using the local fake/ and real/ folders."""

import argparse
from pathlib import Path


IMAGE_HEIGHT = 640
IMAGE_WIDTH = 480


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).parent)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'deepfake_model.h5')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    if not all((data_dir / label).is_dir() for label in ('fake', 'real')):
        raise SystemExit(f'Dataset must contain {data_dir / "fake"} and {data_dir / "real"}.')

    from tensorflow import keras
    from tensorflow.keras import layers

    train_data = keras.utils.image_dataset_from_directory(
        data_dir,
        labels='inferred',
        label_mode='categorical',
        class_names=['fake', 'real'],
        image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        batch_size=args.batch_size,
        validation_split=0.2,
        subset='training',
        seed=42,
    )
    validation_data = keras.utils.image_dataset_from_directory(
        data_dir,
        labels='inferred',
        label_mode='categorical',
        class_names=['fake', 'real'],
        image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        batch_size=args.batch_size,
        validation_split=0.2,
        subset='validation',
        seed=42,
    )

    model = keras.Sequential([
        layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)),
        layers.Rescaling(1.0 / 255),
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(0.1),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation='relu'),
        layers.Dense(2, activation='softmax'),
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_data, validation_data=validation_data, epochs=args.epochs)
    model.save(args.output)
    print(f'Saved trained model to {args.output}')


if __name__ == '__main__':
    main()