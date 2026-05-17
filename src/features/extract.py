import numpy as np

from skimage.feature import hog


def extract_features(image):

    hog_features = hog(

        image,

        orientations=9,

        pixels_per_cell=(8, 8),

        cells_per_block=(2, 2),

        visualize=False,

        feature_vector=True
    )

    mean = np.mean(image)

    std = np.std(image)

    contrast = np.max(image) - np.min(image)

    return np.hstack([

        hog_features,

        mean,

        std,

        contrast
    ])