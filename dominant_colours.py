from tqdm import tqdm
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
from skimage import io


def find_dominant_colours(direc):
    """
    Find 5 dominant colors present in the image

    Args:
        direc (list) List of images.
    Returns:
        dominant_colours (list) flatten palett containing 5 dominant colors present in the image
        img_name (list) list containing image names
    """
    dominant_colours = []
    img_name = []
    for i in tqdm(direc):
        fname = "image_test" + "/" + i
        img = cv2.imread(fname)

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

        lab = cv2.cvtColor(
            img, cv2.COLOR_BGR2LAB
        )  # convert from BGR to LAB color space
        l, a, b = cv2.split(lab)  # split on 3 different channels

        l2 = clahe.apply(l)  # apply CLAHE to the L-channel

        lab = cv2.merge((l2, a, b))  # merge channels
        img2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR

        pixels = np.float32(img2.reshape(-1, 3))

        n_colors = 5
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
        flags = cv2.KMEANS_RANDOM_CENTERS

        _, _, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)

        sorted_palette = palette[palette[:, 2].argsort()]

        palette_1d = sorted_palette.flatten()

        dominant_colours.append(palette_1d)
        img_name.append(i)

    return dominant_colours, img_name
