from tqdm import tqdm
import os
import shutil
import tensorflow as tf


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

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Rescaling(1.0 / 255),
            tf.keras.layers.Conv2D(32, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(num_classes),
        ]
    )

    return model
