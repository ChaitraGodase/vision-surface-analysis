import cv2

def preprocess_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (128, 128))
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)

    return blurred