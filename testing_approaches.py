import tensorflow as tf
from sklearn.cluster import KMeans
import pandas as pd
import os
import shutil
import time

import classical_cv_approach
import ml_supervised_approach
import supervised_and_unsupervised_approach
import dominant_colours


def main(option):
    # Dataset path
    current_path = os.getcwd()
    images_path = "image_test"
    path = os.path.join(current_path, images_path)
    img_path = os.listdir(path)

    # 1. Attempt to extract color of the car using classical computer vision approach
    if option == 1:
        # Extract color of cars
        classical_cv_approach.image_colour_extractor(img_path)

    # 2. Deep Learning/ML (Creating a custom NN to classify 200 types of cars available in the dataset)
    if option == 2:
        dataset_classes = "./dataset_classes_test"
        # Create 200 subfolders (200 classes)
        ml_supervised_approach.create_sub_folders(img_path, dataset_classes, True)
        # Create train and val datasets required by Keras
        number_classes, train_ds, val_ds = ml_supervised_approach.create_keras_dataset(
            dataset_classes
        )
        # Create Sequential Keras model
        model = ml_supervised_approach.create_model(number_classes)

        # Optimizer, loss and metrics used during training
        model.compile(
            optimizer="adam",
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

        # Train the model
        model.fit(train_ds, validation_data=val_ds, epochs=3)
        # Save Keras model
        model.save("my_model.keras")

    # 3. Deep Learning (supervised) + Clustering (unsupervised) >>> Using a faster NN to create features. These features are used to create 200 clusters
    if option == 3:
        dataset_classes = "./dataset_classes_test"
        # Create 200 subfolders (200 classes)
        supervised_and_unsupervised_approach.create_sub_folders(
            img_path, dataset_classes, True
        )
        # Create train and val datasets required by Keras
        (
            number_classes,
            train_ds,
            val_ds,
        ) = supervised_and_unsupervised_approach.create_keras_dataset(dataset_classes)
        # Create Sequential Keras model
        model = supervised_and_unsupervised_approach.create_model(number_classes)

        # Optimizer, loss and metrics used during training
        model.compile(
            optimizer=tf.keras.optimizers.Adadelta(learning_rate=1, name="Adadelta"),
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

        # Train the model
        model.fit(train_ds, validation_data=val_ds, epochs=20)
        # Save Keras model
        model.save("my_model.keras")

        # Using feature extractors
        img_features, img_name = supervised_and_unsupervised_approach.image_feature(
            img_path
        )
        # Creating Clusters
        k = 200
        clusters = KMeans(k, random_state=40)
        clusters.fit(img_features)
        image_cluster = pd.DataFrame(img_name, columns=["image"])
        image_cluster[
            "clusterid"
        ] = clusters.labels_  # To mention which image belong to which cluster
        print(image_cluster)

        # Made folder to separate images
        os.mkdir("clusters_detected/")
        for i in range(k):
            os.mkdir("clusters_detected/" + str(i))

        # Images will be separated according to cluster they belong
        for i in range(len(image_cluster)):
            for j in range(k):
                if image_cluster["clusterid"][i] == j:
                    shutil.copy(
                        os.path.join("image_test", image_cluster["image"][i]),
                        "clusters_detected/" + str(j),
                    )
                    continue

    # 4. Clustering by colors available (yellow, orange, black, blue, white, green, red, and gray )
    if option == 4:
        dom_colours, img_name = dominant_colours.find_dominant_colours(img_path)
        # Creating Clusters
        start_time = time.time()
        k = 8
        clusters = KMeans(k, random_state=40)
        clusters.fit(dom_colours)
        image_cluster = pd.DataFrame(img_name, columns=["image"])
        image_cluster[
            "clusterid"
        ] = clusters.labels_  # To mention which image belong to which cluster
        print(image_cluster)
        print("--- %s seconds ---" % (time.time() - start_time))

        # Made folder to separate images
        os.mkdir("clusters_detected/")
        for i in range(k):
            os.mkdir("clusters_detected/" + str(i))

        # Images will be separated according to cluster they belong
        for i in range(len(image_cluster)):
            for j in range(k):
                if image_cluster["clusterid"][i] == j:
                    shutil.copy(
                        os.path.join("image_test", image_cluster["image"][i]),
                        "clusters_detected/" + str(j),
                    )
                    continue


if __name__ == "__main__":
    # Choose an approach
    # 1. Classical Computer Vision
    # 2. Deep Learning (supervised)
    # 3. Deep Learning (supervised) + Clustering (unsupervised)
    # 4. Dominant colours + 5 colors image segmentation

    main(3)
