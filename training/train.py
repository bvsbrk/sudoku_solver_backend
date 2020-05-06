import sys
from matplotlib import pyplot as plt
import numpy as np
import cv2

im = cv2.imread('train.png')
im3 = im.copy()

gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY)[1]

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

samples = np.empty((0, 1600))
responses = []
keys = [i for i in range(48, 58)]
l = []
for cnt in contours:
    l.append(cv2.contourArea(cnt))

print(sorted(l))

for cnt in contours:
    if cv2.contourArea(cnt) >= 1000:
        [x, y, w, h] = cv2.boundingRect(cnt)

        if h > 28:
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 2)
            roi = thresh[y:y + h, x:x + w]
            roismall = cv2.resize(roi, (40, 40))
            cv2.imshow('norm', im)
            key = cv2.waitKey(0)

            if key == 27:  # (escape to quit)
                sys.exit()
            elif key in keys:
                responses.append(int(chr(key)))
                sample = roismall.reshape((1, 1600))
                samples = np.append(samples, sample, 0)
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)

responses = np.array(responses, np.float32)
responses = responses.reshape((responses.size, 1))
print("training complete")

np.savetxt('generalresponses2.data', responses)
np.savetxt('generalsamples2.data', samples)
