import cv2
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict
from collections import deque





def draw_hough(threshold):
    sudoku = cv2.imread('sudoku3.png')
    return cv2.Canny(threshold, 50, 150, apertureSize=3)
