from tqdm import tqdm
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
from skimage import io


def find_dominant_colours(direc):
    """
    Find 2 dominant colors present in the image using color segmentation (green and black)

    Args:
        direc (list) List of images.
    Returns:
        dominant_colours (list) flatten palett containing 2 dominant colors present in the image
        img_name (list) list containing image names
    """
    dominant_colours = []
    img_name = []
    for i in tqdm(direc):
        fname = "image_test" + "/" + i
        img = cv2.imread(fname)

        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # green mask
        light_green = (70, 100, 85)
        dark_green = (50, 90, 75)
        green_mask = cv2.inRange(hsv_img, dark_green, light_green)
        green_result = cv2.bitwise_and(img, img, mask=green_mask)
        green_result[np.where((green_result != [0, 0, 0]).all(axis=2))] = [0, 255, 0]

        pixels = np.float32(green_result.reshape(-1, 3))

        n_colors = 2
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
        flags = cv2.KMEANS_RANDOM_CENTERS

        _, _, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)

        sorted_palette = palette[palette[:, 1].argsort()]

        palette_1d = sorted_palette.flatten()

        dominant_colours.append(palette_1d)
        img_name.append(i)

    return dominant_colours, img_name
