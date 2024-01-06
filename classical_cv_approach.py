import cv2
from tqdm import tqdm
import numpy as np


def image_colour_extractor(direc):
    """
    Extract color of the car

    Args:
        direc (list) List of images.
    Returns:
        None
    """
    for i in tqdm(direc):
        fname = "image_test" + "/" + i
        image = cv2.imread(fname, cv2.IMREAD_UNCHANGED)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        _, mask = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)
        mask = cv2.erode(mask, np.ones((3, 3), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        largest_item = sorted_contours[0]

        # For visualization purposes
        cv2.drawContours(image, largest_item, -1, (255, 0, 0), 10)
        cv2.imshow("Largest Object", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # TODO: Adapt this piece of code to the car detected (largest_item):
        # filtered_contours = []
        # df_mean_color = pd.DataFrame()
        # for idx, contour in enumerate(contours):
        #     area = int(cv2.contourArea(contour))

        #     # if area is higher than 3000:
        #     if area > 3000:
        #         filtered_contours.append(contour)
        #         # get mean color of contour:
        #         masked = np.zeros_like(
        #             image[:, :, 0]
        #         )  # This mask is used to get the mean color of the specific bead (contour), for kmeans
        #         cv2.drawContours(masked, [contour], 0, 255, -1)

        #         B_mean, G_mean, R_mean, _ = cv2.mean(image, mask=masked)
        #         df = pd.DataFrame(
        #             {"B_mean": B_mean, "G_mean": G_mean, "R_mean": R_mean}, index=[idx]
        #         )
        #         df_mean_color = pd.concat([df_mean_color, df])
