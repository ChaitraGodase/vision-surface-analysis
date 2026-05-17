import numpy as np
from skimage.feature import hog

def extract_features(image):
    features = hog(image)

    mean = np.mean(image)
    std = np.std(image)

    return np.hstack([features, mean, std])