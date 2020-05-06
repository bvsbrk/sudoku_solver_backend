from keras.models import load_model
import cv2
import numpy as np
from pytesseract import image_to_string


def temp(area):
    l = []
    while area > 1:
        ps = int(area ** 0.5) ** 2
        l.append(ps)
        area -= ps
    if area == 1:
        l.append(1)
    return l


def extract_number(image):
    im = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    im = cv2.resize(im, (28, 28))
    im = cv2.threshold(im, 120, 255, cv2.THRESH_BINARY)[1]
    if np.count_nonzero(im) == 784:
        return '.'

    gray = im.reshape((1, 28, 28, 1))

    # normalize image
    gray = gray / 255

    model = load_model("test_model.h5")

    # predict digit
    prediction = model.predict(gray)
    return prediction.argmax()
