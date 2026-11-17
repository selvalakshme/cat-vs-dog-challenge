# Cats vs. Dogs Image Classifier (CNN)

A convolutional neural network built with Keras that classifies an image as a **cat** or a **dog**.

## Overview

The script builds a small CNN, trains it on a folder of labelled cat and dog images with on-the-fly data augmentation, and then predicts the class of a single new image.

## Dataset

The script reads images from a folder structure with one sub-folder per class. The dataset itself is not included, and the script does not name its source; any cats-vs-dogs image set organised this way works.

```
dataset/
  training_set/
    cats/
    dogs/
  test_set/
    cats/
    dogs/
  Dog or Cat ? Predict/
    cat_or_dog_1.jpg      # single image for prediction
```

Keras assigns class indices alphabetically, so `cats = 0` and `dogs = 1`.

## Model

| Step | Layer |
|---|---|
| 1 | `Conv2D(32, 3x3, ReLU)` on 64x64x3 input |
| 2 | `MaxPooling2D(2x2)` |
| 3 | `Conv2D(32, 3x3, ReLU)` |
| 4 | `MaxPooling2D(2x2)` |
| 5 | `Flatten` |
| 6 | `Dense(128, ReLU)` |
| 7 | `Dense(1, sigmoid)` - probability that the image is a dog |

- **Loss:** binary cross-entropy
- **Optimizer:** Adam
- **Metric:** accuracy

## Data Preparation

- **Training images:** rescaled by 1/255 with augmentation (shear 0.2, zoom 0.2, horizontal flip).
- **Test images:** rescaled by 1/255 only.
- All images are resized to 64x64, loaded in batches of 32 with `class_mode='binary'`.

## Training

The model is trained for 25 epochs with `fit_generator`, validating on the test set. The script does not print or save accuracy results, so run it to see the training and validation accuracy.

## Prediction

A single image is loaded at 64x64, converted to an array, given a batch dimension, and passed to `classifier.predict`. A result of 1 is labelled **dog**, otherwise **cat**. The image is then displayed with `IPython.display`.

## Getting Started

Requirements: Python 3, TensorFlow/Keras, NumPy, Pillow, Jupyter (for the image display).

```bash
pip install tensorflow numpy pillow jupyter
```

1. Organise your images in the folder structure above.
2. Replace the `'....../dataset/...'` placeholder paths in the script with your real paths.
3. Run the script (or open it as a notebook) from top to bottom.

## Notes

- **Syntax error:** one line joins two statements (`training_data_path = '....../dataset/training_set'training_set = train_datagen.flow_from_directory(...)`). Put `training_set = ...` on its own line before running.
- **Steps per epoch:** `steps_per_epoch = 8000` and `validation_steps = 2000` are counts of *batches*, not images. With a batch size of 32 this is 256,000 and 64,000 images per epoch. Use `training_set.samples // 32` and `test_set.samples // 32` instead.
- **Deprecated API:** `fit_generator` and `keras.preprocessing.image` are deprecated in recent TensorFlow versions; use `model.fit` and `tf.keras.utils` equivalents.
- **Prediction check:** the sigmoid output is a probability, so `result[0][0] == 1` is rarely true. Compare it with a threshold such as `> 0.5` instead.
