# Algotive Challenge

## Description of the problem: 

You have a dataset of car images, taken from surveillance cameras, you need to develop a system that generates groups of similar images (you are free to decide which feature of the cars you will use). The system must automatically generate groups, where the elements inside each group share the feature selected.

## Project Structure.

Under this project you will find the next files. Next, I will explain their purposes:

- main.py (python script required to run my final implementations)
- testing_approaches.py (python script where I tested several ideas)
- clustering_approach.py (module required to implement image clustering using InceptionV3 + kmeans)
- dominant_colours_segmentation.py (module required to implement clustering using dominant colors, green color segmentation, and kmeans)
- clustering_own_NN_approach.py (module required to implement image clustering using my own model + kmeans)
- classical_cv_approach.py (Approach to extract mean color of the vehicle)
- ml_supervised_approach.py (Approach to train a NN model. This model can detect all type of vehicle available in the dataset, 200 )
- supervised_and_unsupervised_approach.py (Approach to improve a custom NN model by using less weights. This model is used to extract features from the image and these features are used by Kmeans to create 200 clusters)
- dominant_colours.py (module required to implement clustering using dominant colors, image color segmentation, and kmeans)

## Explanation

First, I explored several approaches/ideas in order to choose the appropiate one. These ideas contemplates different approaches and/or combination of them, such as, 
 - Using Classical Computer Vision.
 - Using pretrained NN models.
 - Using my own NN models adapted to the dataset.
 - Different size of clusters.

 Then, after I got ideas, I decided to select two approaches:
 1. Extract dominant color of images and use this information to create clusters by using kmeans.
 2. Extract features using a pretrained NN (InceptionV3). Then, those features are used to create clusters by using kmeans.
 3. Extract features using my own NN. Then, those features are used to create clusters by using kmeans.


 Aditionally, I decided to use three size/type of clusters:
 - The smallest possible number of clusters, 2. Whether it's a car of green color or not. 
 - The greatest possible number of clusters, 1677. All classes found in the dataset where each class is a specific car in a specific orientation. 
 - A middle number of clusters, 200. Each class represent an specific car.

## How to run my implementations?
1. Download dataset and unzip it in the root path. Use the default folder name.
2. Run main.py script (Choose 1, 2, 3, or 4 option)

## Results
Depending on the option selected, images are grouped in a folder called "clusters_detected_2", "clusters_detected_1677_inceptionv3",  "clusters_detected_200", or "clusters_detected_1677_custom". This folder is created automatically.

In the case of the 1rst approach, detecting 2 clusters. Folder 1 contains vehicles of non-green colors, folder 2 contains vehicles of green colors.

In the case of the 2nd approach, detect 1677 clusters. Each folder contains an specific car in an specific orientation. For example, folder #8 contains back-views of a truck. The time required to group all dataset in clusters using inceptionV3 model is ~90 minutos.

For the third approach, detecting 200 clusters. Each folder contains an specific car. I implemented it to demonstrate the NN model I trained is able to create clusters by specific car, 200 types of cars.

Finally, the fourth option, detecting 1677 clusters by using my own NN model. I implemented it in order to optimize the model and have better performance.The time required to group all dataset in clusters using my own model is ~53 seconds.

## Performance improvements
Using a pretrained model (inceptionV3 model) required ~90 minutos to group 1677 classes. So, I decided to explore another NN architecture in order to decrease this time. 

Previously, I used [1] NN model with great performance results. Therefore, I give it other chance for this task.

So, taking into account the results described above of 2nd vs 4th approaches:

- Speedup = 5400/53 = ~102X.

Additionally, inceptionV3 has ~24 million of parameters. Whereas, my NN model has ~4 million of parameters.

## References
1. Falaschetti L, Manoni L, Di Leo D, Pau D, Tomaselli V, Turchetti C. A CNN-based image detector for plant leaf diseases classification. HardwareX. 2022 Sep 27;12:e00363.






