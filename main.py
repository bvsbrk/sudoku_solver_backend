import cv2
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict
from collections import deque
from time import time
import os, sys
from extract_number import extract_number

sudoku = cv2.imread('sudoku3.png')


def showImage(title, img):
    cv2.imshow(title, img)
    cv2.waitKey()

def invert(full_img):
    full_img[full_img == 255] = 254
    full_img[full_img == 0] = 255
    full_img[full_img == 254] = 0

def fill_edges_with_black(image):
    def is_valid_neighbor(x, y):
        return 0 <= x < len(image) and 0 <= y < len(image[0]) and image[x][y] == 255

    l = [(0, i) for i in range(image.shape[0])]
    l += [(image.shape[0] - 1, i) for i in range(image.shape[0])]
    l += [(i, 0) for i in range(image.shape[1])]
    l += [(i, image.shape[1] - 1) for i in range(image.shape[1])]
    que = deque(l)
    while len(que) > 0:
        p, q = que.popleft()
        if is_valid_neighbor(p, q):
            image[p, q] = 0
        if is_valid_neighbor(p - 1, q):
            que.append((p - 1, q))
            image[p - 1][q] = 0
        if is_valid_neighbor(p - 1, q - 1):
            que.append((p - 1, q - 1))
            image[p - 1][q - 1] = 0
        if is_valid_neighbor(p, q - 1):
            que.append((p, q - 1))
            image[p][q - 1] = 0
        if is_valid_neighbor(p + 1, q - 1):
            que.append((p + 1, q - 1))
            image[p + 1][q - 1] = 0
        if is_valid_neighbor(p + 1, q):
            que.append((p + 1, q))
            image[p + 1][q] = 0
        if is_valid_neighbor(p + 1, q + 1):
            que.append((p + 1, q + 1))
            image[p + 1][q + 1] = 0
        if is_valid_neighbor(p, q + 1):
            que.append((p, q + 1))
            image[p][q + 1] = 0
        if is_valid_neighbor(p - 1, q + 1):
            que.append((p - 1, q + 1))
            image[p - 1][q + 1] = 0


def gray_blur_threshold(img, blur_dim):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # showImage("gray", gray)

    blur = cv2.GaussianBlur(gray, (blur_dim, blur_dim), 0)
    # showImage("blur", blur)

    threshold = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
    # showImage("threshold", threshold)
    return threshold

def get_cropped_cell(image):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    img_height = image.shape[0]
    if len(contours) == 0:
        return image
    biggest_contour = 0
    idx = -1
    for i in range(len(contours)):
        cnt = contours[i]
        x, y, w, h = cv2.boundingRect(cnt)
        if h != img_height and cv2.contourArea(cnt) > biggest_contour:
            biggest_contour = cv2.contourArea(cnt)
            idx = i
    if idx == -1:
        return image
    else:
        x, y, w, h = cv2.boundingRect(contours[idx])
        return image[y:y + h, x:x + w]



def floodFill(image, min_pixels):
    def is_valid_neighbor(x, y, color):
        return 0 <= x < len(image) and 0 <= y < len(image[0]) and image[x][y] == color and visited[x][y] == 0

    visited = np.zeros(image.shape)
    img_idx = 0
    for i in range(len(image)):
        for j in range(len(image[i])):
            if i == 500 and j == 500:
                print(i, j)
            if visited[i][j] == 0:
                cur = []
                que = deque([(i, j)])
                while len(que) > 0:
                    p, q = que.popleft()
                    clr = image[p][q]
                    visited[p, q] = 1
                    cur.append((p, q))
                    if is_valid_neighbor(p - 1, q, clr):
                        que.append((p - 1, q))
                        visited[p - 1][q] = 1
                    if is_valid_neighbor(p - 1, q - 1, clr):
                        que.append((p - 1, q - 1))
                        visited[p - 1][q - 1] = 1
                    if is_valid_neighbor(p, q - 1, clr):
                        que.append((p, q - 1))
                        visited[p][q - 1] = 1
                    if is_valid_neighbor(p + 1, q - 1, clr):
                        que.append((p + 1, q - 1))
                        visited[p + 1][q - 1] = 1
                    if is_valid_neighbor(p + 1, q, clr):
                        que.append((p + 1, q))
                        visited[p + 1][q] = 1
                    if is_valid_neighbor(p + 1, q + 1, clr):
                        que.append((p + 1, q + 1))
                        visited[p + 1][q + 1] = 1
                    if is_valid_neighbor(p, q + 1, clr):
                        que.append((p, q + 1))
                        visited[p][q + 1] = 1
                    if is_valid_neighbor(p - 1, q + 1, clr):
                        que.append((p - 1, q + 1))
                        visited[p - 1][q + 1] = 1
                if len(cur) <= min_pixels:
                    replace = 0 if image[cur[0][0], cur[0][1]] == 255 else 255
                    for pixel in cur:
                        image[pixel[0], pixel[1]] = replace


def drawHough(gray):
    # filter = False
    # img = cv2.imread('sudoku3.png')
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 90, 150, apertureSize=3)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.erode(edges, kernel, iterations=1)
    cv2.imwrite('canny.jpg', edges)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 120)

    if lines is None or not lines.any():
        print('No lines were found')
        sys.exit()

    if filter:
        rho_threshold = 15
        theta_threshold = 0.1

        # how many lines are similar to a given one
        similar_lines = {i: [] for i in range(len(lines))}
        for i in range(len(lines)):
            for j in range(len(lines)):
                if i == j:
                    continue

                rho_i, theta_i = lines[i][0]
                rho_j, theta_j = lines[j][0]
                if abs(rho_i - rho_j) < rho_threshold and abs(theta_i - theta_j) < theta_threshold:
                    similar_lines[i].append(j)

        # ordering the INDECES of the lines by how many are similar to them
        indices = [i for i in range(len(lines))]
        indices.sort(key=lambda x: len(similar_lines[x]))

        # line flags is the base for the filtering
        line_flags = len(lines) * [True]
        for i in range(len(lines) - 1):
            if not line_flags[indices[
                i]]:  # if we already disregarded the ith element in the ordered list then we don't care (we will not delete anything based on it and we will never reconsider using this line again)
                continue

            for j in range(i + 1, len(lines)):  # we are only considering those elements that had less similar line
                if not line_flags[indices[j]]:  # and only if we have not disregarded them already
                    continue

                rho_i, theta_i = lines[indices[i]][0]
                rho_j, theta_j = lines[indices[j]][0]
                if abs(rho_i - rho_j) < rho_threshold and abs(theta_i - theta_j) < theta_threshold:
                    line_flags[
                        indices[j]] = False  # if it is similar and have not been disregarded yet then drop it now

    print('number of Hough lines:', len(lines))

    filtered_lines = []

    if filter:
        for i in range(len(lines)):  # filtering
            if line_flags[i]:
                filtered_lines.append(lines[i])

        print('Number of filtered lines:', len(filtered_lines))
    else:
        filtered_lines = lines

    for line in filtered_lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(gray, (x1, y1), (x2, y2), 255, 2)

    return gray


def change_persepctive(image, lu, ld, rd, ru, image_size):
    size = image_size

    dest = np.array([
        [0, 0],
        [size, 0],
        [size, size],
        [0, size]],
        dtype="float32")
    source = np.array([lu, ru, rd, ld], dtype="float32")
    transf = cv2.getPerspectiveTransform(source, dest)
    warped = cv2.warpPerspective(image, transf, (size, size))

    return warped


def draw_white_outline(full_img):
    ones = np.array([255] * full_img.shape[0])
    full_img[:, -1] = ones
    full_img[:, -1] = ones
    full_img[:, 0] = ones
    full_img[:, 1] = ones
    full_img[0, :] = ones
    full_img[1, :] = ones
    full_img[-1, :] = ones
    full_img[-2, :] = ones


if __name__ == '__main__':
    print(time())
    blur_dim = max(sudoku.shape) // 100
    blur_dim -= 0 if blur_dim & 1 else 1
    # showImage("sudoku", image)

    threshold = gray_blur_threshold(sudoku, blur_dim)

    # showImage("first threshold", threshold)

    contours, hierarchy = cv2.findContours(threshold, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    final_sudoku = np.zeros(threshold.shape)

    # Find biggest contour
    max_area = 0
    best_contour_idx = 0
    for idx in range(len(contours)):
        if cv2.contourArea(contours[idx]) > max_area:
            max_area = cv2.contourArea(contours[idx])
            best_contour_idx = idx

    cv2.drawContours(final_sudoku, contours, best_contour_idx, 255, -1)

    outline = cv2.approxPolyDP(contours[best_contour_idx], 0.009 * cv2.arcLength(contours[best_contour_idx], True),
                               True)
    vertices = outline.ravel().reshape(4, 2)

    lu, ld, rd, ru = vertices

    only_sudoku = change_persepctive(sudoku, lu, ld, rd, ru, 500)

    # showImage("only sudoku", only_sudoku)

    threshold = gray_blur_threshold(only_sudoku, blur_dim)

    floodFill(threshold, 30)

    showImage("before", threshold)
    full_img = drawHough(threshold)
    draw_white_outline(full_img)

    showImage("after", full_img)

    print(time())
    x_coordinates = []
    for i in range(full_img.shape[1]):
        j = 0
        while j < full_img.shape[0] and full_img[i, j] == 255:
            j += 1
        if j >= 0.8 * full_img.shape[1]:
            if len(x_coordinates) == 0:
                x_coordinates.append(i)
            elif i - x_coordinates[-1] > 10:
                x_coordinates.append(i)
    print(time())
    print(x_coordinates)

    y_coordinates = []
    for i in range(full_img.shape[0]):
        j = 0
        while j < full_img.shape[1] and full_img[i, j] == 255:
            j += 1
        if j >= 0.8 * full_img.shape[0]:
            if len(y_coordinates) == 0:
                y_coordinates.append(i)
            elif i - y_coordinates[-1] > 10:
                y_coordinates.append(i)

    print(y_coordinates)

    for i in range(1, len(x_coordinates)):
        temp = full_img[:, x_coordinates[i - 1]: x_coordinates[i]]

        for j in range(1, len(y_coordinates)):
            temp2 = temp[y_coordinates[j - 1]: y_coordinates[j], :]
            fill_edges_with_black(temp2)
            invert(temp2)
            temp2 = get_cropped_cell(temp2)
            cv2.imwrite("cells/{}_{}.jpg".format(j - 1, i - 1), temp2)

    for image in sorted(os.listdir("cells")):
        print(image, extract_number("cells/{}".format(image)))


