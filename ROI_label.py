import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import numpy as np

from PIL import Image

# Replace 'path_to_image.jpg' with the path to your actual image
image_path = 'D:/McMaster/4OI6/Train_Strawberry_Dataset_Final_Version/strawberries/train_0001.jpg'
image = np.array(Image.open(image_path).convert("RGB"))

fig, ax = plt.subplots()
ax.imshow(image)

def onselect(eclick, erelease):
    'Function to handle the selection event'
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata

    # Ensure we have x1 < x2 and y1 < y2
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])

    # Calculate the normalized coordinates
    width = image.shape[1]
    height = image.shape[0]
    x_center = (x1 + x2) / (2 * width)
    y_center = (y1 + y2) / (2 * height)
    roi_width = (x2 - x1) / width
    roi_height = (y2 - y1) / height

    # The class index is set to 1 as an example, change it if needed
    class_index = 1

    print(f"{class_index} {x_center:.6f} {y_center:.6f} {roi_width:.6f} {roi_height:.6f}")

# Create a RectangleSelector widget
toggle_selector = RectangleSelector(ax, onselect, useblit=True,
                                    button=[1], minspanx=5, minspany=5,
                                    spancoords='pixels', interactive=True)

plt.show()
