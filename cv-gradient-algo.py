import cv2
import numpy as np

moon = cv2.imread('D:/McMaster/Advanced/Computer_Vision/train_0001.jpg', 0)
row, column = moon.shape
moon_f = moon.astype("float")

gradient = np.zeros((row, column))

for x in range(row - 1):
    for y in range(column - 1):
        gx = abs(moon_f[x + 1, y] - moon_f[x, y])
        gy = abs(moon_f[x, y + 1] - moon_f[x, y])
        gradient[x, y] = gx + gy

sharp = moon_f + gradient
sharp = np.where(sharp < 0, 0, np.where(sharp > 255, 255, sharp))

gradient = gradient.astype("uint8")
sharp = sharp.astype("uint8")

screen_res = 1280, 720
scale_width = screen_res[0] / moon.shape[1]
scale_height = screen_res[1] / moon.shape[0]
scale = min(scale_width, scale_height)

window_width = int(moon.shape[1] * scale)
window_height = int(moon.shape[0] * scale)

resized_moon = cv2.resize(moon, (window_width, window_height), interpolation=cv2.INTER_AREA)
resized_gradient = cv2.resize(gradient, (window_width, window_height), interpolation=cv2.INTER_AREA)
resized_sharp = cv2.resize(sharp, (window_width, window_height), interpolation=cv2.INTER_AREA)

cv2.imshow("moon", resized_moon)
cv2.imshow("gradient", resized_gradient)
cv2.imshow("sharp", resized_sharp)
cv2.waitKey(0)
cv2.destroyAllWindows()
