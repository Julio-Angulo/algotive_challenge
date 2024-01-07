import tensorflow as tf
from sklearn.cluster import KMeans
import pandas as pd
import os
import shutil
import time

import clustering_approach
import dominant_colours_segmentation
import clustering_own_NN_approach


def main(option):
    # Dataset path
    current_path = os.getcwd()
    images_path = "image_test"
    path = os.path.join(current_path, images_path)
    img_path = os.listdir(path)

    # 1. Clustering by green color available
    if option == 1:
        dom_colours, img_name = dominant_colours_segmentation.find_dominant_colours(
            img_path
        )
        # Creating Clusters
        start_time = time.time()
        k = 2
        clusters = KMeans(k, random_state=40)
        clusters.fit(dom_colours)
        image_cluster = pd.DataFrame(img_name, columns=["image"])
        image_cluster[
            "clusterid"
        ] = clusters.labels_  # To mention which image belong to which cluster
        print(image_cluster)
        print("--- %s seconds ---" % (time.time() - start_time))

        # Made folder to separate images
        os.mkdir("clusters_detected_2/")
        for i in range(k):
            os.mkdir("clusters_detected_2/" + str(i))

        # Images will be separated according to cluster they belong
        for i in range(len(image_cluster)):
            for j in range(k):
                if image_cluster["clusterid"][i] == j:
                    shutil.copy(
                        os.path.join("image_test", image_cluster["image"][i]),
                        "clusters_detected_2/" + str(j),
                    )
                    continue

    # 2. Clustering all available classes in the dataset (1677 classes) by using inceptionv3
    if option == 2:
        # Using feature extractors
        img_features, img_name = clustering_approach.image_feature(img_path)
        # Creating Clusters
        start_time = time.time()
        k = 1677
        clusters = KMeans(k, random_state=40)
        clusters.fit(img_features)
        image_cluster = pd.DataFrame(img_name, columns=["image"])
        image_cluster[
            "clusterid"
        ] = clusters.labels_  # To mention which image belong to which cluster
        print(image_cluster)
        print("--- %s seconds ---" % (time.time() - start_time))

        # Made folder to separate images
        os.mkdir("clusters_detected_1677_inceptionv3/")
        for i in range(k):
            os.mkdir("clusters_detected_1677_inceptionv3/" + str(i))

        # Images will be separated according to cluster they belong
        for i in range(len(image_cluster)):
            for j in range(k):
                if image_cluster["clusterid"][i] == j:
                    shutil.copy(
                        os.path.join("image_test", image_cluster["image"][i]),
                        "clusters_detected_1677_inceptionv3/" + str(j),
                    )
                    continue

    # 3. Clustering 200 types of vehicles available in the dataset (200 classes) by using a faster model
    if option == 3:
        dataset_classes = "./dataset_classes_200"
        # Create 200 subfolders (200 classes)
        clustering_own_NN_approach.create_sub_folders(img_path, dataset_classes, 200)
        # Create train and val datasets required by Keras
        (
            number_classes,
            train_ds,
            val_ds,
        ) = clustering_own_NN_approach.create_keras_dataset(dataset_classes)
        # Create Sequential Keras model
        model = clustering_own_NN_approach.create_model(number_classes)

        # Optimizer, loss and metrics used during training
        model.compile(
            optimizer=tf.keras.optimizers.Adadelta(learning_rate=1, name="Adadelta"),
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

        # Train the model
        model.fit(train_ds, validation_data=val_ds, epochs=20)

        # Save Keras model
        model.save("model_c200.keras")

        # Using feature extractors
        img_features, img_name = clustering_own_NN_approach.image_feature(img_path, 200)
        # Creating Clusters
        start_time = time.time()
        k = 200
        clusters = KMeans(k, random_state=40)
        clusters.fit(img_features)
        image_cluster = pd.DataFrame(img_name, columns=["image"])
        image_cluster[
            "clusterid"
        ] = clusters.labels_  # To mention which image belong to which cluster
        print(image_cluster)
        print("--- %s seconds ---" % (time.time() - start_time))

        # Made folder to separate images
        os.mkdir("clusters_detected_200/")
        for i in range(k):
            os.mkdir("clusters_detected_200/" + str(i))

        # Images will be separated according to cluster they belong
        for i in range(len(image_cluster)):
            for j in range(k):
                if image_cluster["clusterid"][i] == j:
                    shutil.copy(
                        os.path.join("image_test", image_cluster["image"][i]),
                        "clusters_detected_200/" + str(j),
                    )
                    continue

    # 4. Clustering all available classes in the dataset (1677 classes) by using a faster model
    if option == 4:
        dataset_classes = "./dataset_classes_1677"
        # Create 1677 subfolders (1677 classes)
        clustering_own_NN_approach.create_sub_folders(img_path, dataset_classes, 1677)
        # Create train and val datasets required by Keras
        (
            number_classes,
            train_ds,
            val_ds,
        ) = clustering_own_NN_approach.create_keras_dataset(dataset_classes)
        # Create Sequential Keras model
        model = clustering_own_NN_approach.create_model(number_classes)

        # Optimizer, loss and metrics used during training
        model.compile(
            optimizer=tf.keras.optimizers.Adadelta(learning_rate=1, name="Adadelta"),
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

        # Train the model
        model.fit(train_ds, validation_data=val_ds, epochs=20)

        # Save Keras model
        model.save("model_c1677.keras")

        # Using feature extractors
        img_features, img_name = clustering_own_NN_approach.image_feature(
            img_path, 1677
        )
        # Creating Clusters
        start_time = time.time()
        k = 1677
        clusters = KMeans(k, random_state=40)
        clusters.fit(img_features)
        image_cluster = pd.DataFrame(img_name, columns=["image"])
        image_cluster[
            "clusterid"
        ] = clusters.labels_  # To mention which image belong to which cluster
        print(image_cluster)
        print("--- %s seconds ---" % (time.time() - start_time))

        # Made folder to separate images
        os.mkdir("clusters_detected_1677_custom/")
        for i in range(k):
            os.mkdir("clusters_detected_1677_custom/" + str(i))

        # Images will be separated according to cluster they belong
        for i in range(len(image_cluster)):
            for j in range(k):
                if image_cluster["clusterid"][i] == j:
                    shutil.copy(
                        os.path.join("image_test", image_cluster["image"][i]),
                        "clusters_detected_1677_custom/" + str(j),
                    )
                    continue


if __name__ == "__main__":
    # Choose an approach
    # 1. Dominant colors + green color segmentation
    # 2. Clustering (unsupervised) using inceptionv3 model (1677 classes)
    # 3. Clustering using my own NN model (200 classes)
    # 4. Clustering using my own NN model (1677 classes)

    main(4)
