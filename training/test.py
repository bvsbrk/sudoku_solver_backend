from keras.models import load_model
import os
import cv2
from matplotlib import pyplot as plt
import numpy as np

model = load_model("printed_digit.h5")

for image in os.listdir("../cells"):
    im = cv2.imread("../cells/" + image)
    im = cv2.resize(im, (40, 40))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = cv2.threshold(im, 120, 255, cv2.THRESH_BINARY)[1]
    if np.count_nonzero(im) == 1600:
        print(image, '.')
    else:
        im = im / 255
        model = load_model("printed_digit.h5")
        im = im.reshape((1, 40, 40, 1))
        print(image, model.predict(im).argmax())
