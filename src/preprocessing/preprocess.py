import cv2
import numpy as np


def preprocess_image(image_input):

    # =====================================
    # FILE PATH INPUT
    # =====================================

    if isinstance(image_input, str):

        img = cv2.imread(image_input)

    # =====================================
    # NUMPY ARRAY INPUT
    # =====================================

    else:

        img = image_input

    # =====================================
    # VALIDATION
    # =====================================

    if img is None:
        return None

    # =====================================
    # RGB TO GRAY
    # =====================================

    if len(img.shape) == 3:

        img = cv2.cvtColor(
            img,
            cv2.COLOR_RGB2GRAY
        )

    # =====================================
    # RESIZE
    # =====================================

    img = cv2.resize(
        img,
        (128, 128)
    )

    # =====================================
    # BLUR
    # =====================================

    img = cv2.GaussianBlur(
        img,
        (5, 5),
        0
    )

    return img