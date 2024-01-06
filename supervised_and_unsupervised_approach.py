from tqdm import tqdm
import os
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np


def create_sub_folders(direc, dataset_classes, is_test):
    """
    Create subfolders of all classes

    Args:
        direc (list) List of images.
        dataset_classes (string) name of the new folder containing all subfolders
        is_test (bool) It's a test or not
    Returns:
        None
    """
    # check whether directory already exists
    if not os.path.exists(dataset_classes):
        os.mkdir(dataset_classes)
        print("Folder %s created!" % dataset_classes)

        for i in tqdm(direc):
            path = "image_test" + "/" + i
            slash_split = path.split("/")
            underscore_split = slash_split[1].split("_")
            if not is_test:
                class_detected = underscore_split[0] + "_" + underscore_split[1]
            else:
                class_detected = underscore_split[0]

            if not os.path.exists(dataset_classes + "/" + class_detected):
                os.mkdir(dataset_classes + "/" + class_detected)

            shutil.copy(path, dataset_classes + "/" + class_detected)
    else:
        print("Folder %s already exists" % dataset_classes)


def create_keras_dataset(direc):
    """
    Create training and evaluation datasets

    Args:
        direc (list) List of images.
    Returns:
        train_ds (tf dataset) training dataset
        val_ds (tf dataset) validation dataset
        total_classes (int) number of classes
    """
    batch_size = 10
    img_height = 180
    img_width = 180

    train_ds = tf.keras.utils.image_dataset_from_directory(
        direc,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        direc,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
    )

    class_names = train_ds.class_names
    total_classes = len(class_names)

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return total_classes, train_ds, val_ds


def create_model(num_classes):
    """
    Create keras sequential model

    Args:
        num_classes (int) Number of classes required to classify.
    Returns:
        model (keras model) A keras model
    """

    # The model consists of 3 Convolution Layers with 2 Fully Connected with a maximum pool layer in each of them.

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Rescaling(1.0 / 255))

    # Feature Learning block:
    model.add(
        tf.keras.layers.Conv2D(
            16,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
            input_shape=(180, 180, 3),
        )
    ),
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(3, 3))),
    model.add(
        tf.keras.layers.Conv2D(
            32, kernel_size=(3, 3), padding="same", activation="relu"
        )
    ),
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(3, 3))),
    model.add(
        tf.keras.layers.Conv2D(
            64, kernel_size=(3, 3), padding="same", activation="relu"
        )
    ),
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2))),

    # Classification block:
    model.add(tf.keras.layers.Flatten()),
    model.add(tf.keras.layers.Dense(units=512, activation="relu")),
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(units=num_classes))

    return model


def image_feature(direc):
    """
    Extract features of all images using InceptionV3

    Args:
        direc (list) List of images.
    Returns:
        features (list) list containing features of images
        img_name (list) list containing image names
    """
    model = tf.keras.models.load_model("my_model.keras")
    model.summary()
    features = []
    img_name = []
    for i in tqdm(direc):
        fname = "image_test" + "/" + i
        img = image.load_img(fname, target_size=(180, 180))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        feat = model.predict(x)
        feat = feat.flatten()
        features.append(feat)
        img_name.append(i)

    print(f"features shape = {np.shape(features)}")
    print(f"img_name shape = {np.shape(img_name)}")
    return features, img_name
