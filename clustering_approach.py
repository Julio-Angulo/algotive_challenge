import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array


def image_feature(direc):
    """
    Extract features of all images using InceptionV3

    Args:
        direc (list) List of images.
    Returns:
        features (list) list containing features of images
        img_name (list) list containing image names
    """
    model = InceptionV3(weights="imagenet", include_top=False)
    # model.summary()
    features = []
    img_name = []
    for i in tqdm(direc):
        # fname = "image_test" + "/" + i
        fname = "image_test" + "/" + i
        img = image.load_img(fname, target_size=(224, 224))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        feat = model.predict(x)
        feat = feat.flatten()
        print(f"feat shape = {np.shape(feat)}")
        features.append(feat)
        img_name.append(i)

    return features, img_name
