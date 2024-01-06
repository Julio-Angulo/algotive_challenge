import tensorflow as tf
from sklearn.cluster import KMeans
import pandas as pd
import os
import shutil
import time

import classical_cv_approach
import clustering_approach
import ml_supervised_approach
import supervised_and_unsupervised_approach
import dominant_colours
import dominant_colours_segmentation


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

    # 2. Clustering all available classes in the dataset (1677 classes)
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
    # 1. Dominant colors + green color segmentation
    # 2. Clustering (unsupervised)

    main(2)
