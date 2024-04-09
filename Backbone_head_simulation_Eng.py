import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import matplotlib.patches as patches

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))

class YOLOBackbone(nn.Module):
    def __init__(self):
        super(YOLOBackbone, self).__init__()
        self.layer1 = ConvBlock(3, 32, kernel_size=3, stride=1, padding=1)
        self.layer2 = ConvBlock(32, 64, kernel_size=3, stride=2, padding=1)
        self.layer3 = ConvBlock(64, 128, kernel_size=3, stride=1, padding=1)
        self.layer4 = ConvBlock(128, 256, kernel_size=3, stride=2, padding=1)
        # self.layer5 = ConvBlock(256, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # x = self.layer5(x)
        return x

def visualize_feature_maps(feature_maps, num_columns=6, scale_factor=1.5,file_name="feature_maps.png"):

    feature_maps = feature_maps[0]

    num_feature_maps = feature_maps.shape[0]  

    num_rows = int(np.ceil(num_feature_maps / num_columns))

    fig, axes = plt.subplots(num_rows, num_columns, figsize=(num_columns * scale_factor, num_rows * scale_factor))
    axes = axes.flatten()

    for i in range(num_feature_maps):
        fm = feature_maps[i].detach().numpy()
        fm = (fm - fm.min()) / (fm.max() - fm.min())

        axes[i].imshow(fm, cmap='gray')
        axes[i].axis('off')
        if i >= num_feature_maps - 1:
            break

    plt.tight_layout()
    #plt.show()
    plt.savefig(file_name)
    plt.close(fig)

yolo_backbone = YOLOBackbone()
print(yolo_backbone)


#########################################################################################################


image_path = 'D:/McMaster/4OI6/real_strawberry_3.jpg'

image = Image.open(image_path)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

transformed_image = transform(image)

dummy_input = transformed_image.unsqueeze(0)

print(f"转换后图像张量的形状: {dummy_input.shape}")

backbone_output  = yolo_backbone(dummy_input)

visualize_feature_maps(backbone_output , scale_factor=5, file_name="D:/McMaster/4OI6/Yolo_Simulate_Code/my_backbone_feature_maps_real_strawberry_3.png")  

print(f"主干网络的输出形状: {backbone_output .shape}")

##########################################

class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(DetectionHead, self).__init__()

        self.box_pred = nn.Conv2d(in_channels, 4, kernel_size=1)


        self.class_pred = nn.Conv2d(in_channels, num_classes, kernel_size=1)

        self.conf_pred = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):

        bboxes = self.box_pred(x)
        class_scores = self.class_pred(x)
        confidences = self.conf_pred(x)

        return bboxes, class_scores, confidences

num_classes = 1  
head = DetectionHead(256, num_classes)


bboxes, class_scores, confidences = head(backbone_output)

def visualize_output(original_img, bboxes, class_scores, confidences):

    img_np = original_img.permute(1, 2, 0).numpy()
    plt.imshow(img_np)

    bboxes_np = bboxes.detach().numpy() 
    for i in range(bboxes_np.shape[2]):  
        for j in range(bboxes_np.shape[3]):

            x_center, y_center, width, height = bboxes_np[0, :, i, j]

            x = x_center - width / 2
            y = y_center - height / 2

            rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')
            plt.gca().add_patch(rect)

    plt.show()


    class_scores_np = class_scores.detach().numpy() 
    confidences_np = confidences.detach().numpy()  
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(class_scores_np[0, 0, :, :], cmap='hot', interpolation='nearest')
    ax[0].set_title('Class Scores')
    ax[1].imshow(confidences_np[0, 0, :, :], cmap='hot', interpolation='nearest')
    ax[1].set_title('Confidences')
    plt.show()

num_classes = 3 
head = DetectionHead(256, num_classes)

bboxes, class_scores, confidences = head(backbone_output)

visualize_output(transformed_image, bboxes, class_scores, confidences)
