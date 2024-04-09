import cv2
import numpy as np


def gaussian_blur(image):
    return cv2.GaussianBlur(image, (5, 5), 0)


def gradient_edge_detection(image):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    gradient = cv2.magnitude(sobelx, sobely)
    return gradient

image = cv2.imread('real_strawberry_3.jpg')

screen_res = 1280, 720
scale_width = screen_res[0] / image.shape[1]
scale_height = screen_res[1] / image.shape[0]
scale = min(scale_width, scale_height)

window_width = int(image.shape[1] * scale)
window_height = int(image.shape[0] * scale)

resized = cv2.resize(image, (window_width, window_height), interpolation=cv2.INTER_AREA)
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)


blurred_gray = gaussian_blur(gray)
gradient_gray = gradient_edge_detection(blurred_gray)


gradient_gray_normalized = cv2.normalize(gradient_gray, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

cv2.imshow('resized',resized)
cv2.imshow('Blurred Gray', blurred_gray)
cv2.imshow('Gradient Gray', gradient_gray_normalized)
cv2.waitKey(0)
cv2.destroyAllWindows()
