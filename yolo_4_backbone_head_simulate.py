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



###############################################################################################################

'''

# 可替换
# Suppose we have an image of size 256x256 with 3 channels (RGB)

dummy_input = torch.randn(1, 3, 256, 256)

# Convert the tensor to numpy array for visualization
# Squeeze removes the batch dimension, transpose rearranges dimensions from (C, H, W) to (H, W, C)
image = dummy_input.squeeze().numpy().transpose(1, 2, 0)

# Normalize the image for display
image = (image - image.min()) / (image.max() - image.min())

# Plot the image
plt.imshow(image)
plt.title("Randomly Generated Dummy Image")
plt.axis('off')
plt.show()

'''


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



'''
输入图像的形状是 [1, 3, 256, 256]，输出特征图的形状是 [1, 512, 64, 64]

特征值是从原始数据中提取信息或表示（表示为边缘，纹理，模式等等）

特征值变多（即通道数增加）有以下几个好处：

特征提取能力增强：随着网络深度的增加，可以学习更多、更复杂的特征，从基础的边缘和纹理到更高级的形状和模式。
表征能力提升：更多的特征通道意味着网络能够更好地区分不同的物体和图像内容，提高分类和检测的精确度。
增强了网络的非线性: 深层网络能够通过非线性激活函数如LeakyReLU来组合低层特征, 从而识别高层次的抽象概念。
而像素变少（即空间维度减小）通常有以下影响：

减少计算量：随着像素的减少，后续层需要处理的数据量显著降低，从而减轻计算负担，提高处理速度。
增加感受野：通过减小空间分辨率，每个卷积核覆盖的输入区域（感受野）变大，能够捕捉更广泛的上下文信息，这对于理解整体结构是有帮助的。
可能丢失细节：减小空间分辨率会损失一些细节信息，这对于一些需要高分辨率来识别细小特征的任务可能是不利的。

'''


'''
在神经网络中,特别是在卷积神经网络(CNN)中,逐渐增加卷积层的输出通道数量(在这个例子中从3增加到512)是一种常见的做法。这种设计背后有几个关键原因：

特征层次化：网络中的初始层通常捕捉到更简单和更通用的特征（如边缘和颜色）。随着网络的深入，后面的层能够从前面层的输出中学习更复杂和更抽象的特征。这意味着在更深的层次，需要更多的神经元（或通道）来表示这些复杂的特征。

感受野的增加：随着网络层的加深，每个神经元的“感受野”（即它能观察到的输入区域的大小）也在增加。这意味着更深层次的神经元能够捕捉到输入数据中更大区域的模式和关系。为了有效地处理这些大范围的信息，需要更多的通道来学习不同种类的特征。

参数数量的平衡:在卷积网络中,随着层的深入,通常会减小特征图的空间维度(例如,使用步长为2的卷积或池化来减半特征图的高度和宽度)。为了保持或增加网络的容量(即其学习复杂功能的能力)，我们增加了通道数量。

网络容量的增加：随着更多通道的增加，网络能够学习并存储更多的信息。这是必要的，因为在处理复杂任务（如图像识别、物体检测）时，需要大量的特征来理解和区分不同的类别和对象。
'''


'''
是的，根据您提供的 `YOLOBackbone` 网络和 `visualize_feature_maps` 函数，最终输出的特征图数量取决于网络最后一个卷积层的输出通道数。

在您的示例中，最后一层 `ConvBlock` 将输出通道数设置为了 512,这意味着该层的输出是一个具有 512 个不同通道的特征图集合。

每个特征图可以视为原始输入（在这种情况下是图像）的不同特征的表示。

这些特征是通过网络中的卷积、批量归一化和激活函数学习到的。每个通道的特征图突出显示了输入数据中某种特定模式或特征的响应。

当您调用 `visualize_feature_maps` 函数时，它会为每个通道生成一个子图。

因此，如果输出特征图有 512 个通道，您的可视化将包括 512 张不同的图像，每张图像代表一个通道的特征图。需要注意的是，这些图像表示的是高度抽象化的特征，可能不会像普通的图像那样直观。
'''

class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(DetectionHead, self).__init__()
        # 预测边界框坐标
        self.box_pred = nn.Conv2d(in_channels, 4, kernel_size=1)

        # 预测物体类别
        self.class_pred = nn.Conv2d(in_channels, num_classes, kernel_size=1)

        # 预测置信度（边界框内是否存在物体）
        self.conf_pred = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        # x 是来自特征提取网络的特征图

        # 分别进行边界框、类别和置信度的预测
        bboxes = self.box_pred(x)
        class_scores = self.class_pred(x)
        confidences = self.conf_pred(x)

        return bboxes, class_scores, confidences

# 创建检测头网络
num_classes = 1  # 假如检测图像中的车辆、行人和交通信号灯，那么 num_classes 应该设置为 3。
head = DetectionHead(256, num_classes)

# 前向传播
bboxes, class_scores, confidences = head(backbone_output)

def visualize_output(original_img, bboxes, class_scores, confidences):
    # 将原始图像转换为numpy数组并绘制
    img_np = original_img.permute(1, 2, 0).numpy()
    plt.imshow(img_np)

    # 添加边界框 - 假设bboxes是中心点坐标和宽高形式
    bboxes_np = bboxes.detach().numpy()  # 分离并转换为numpy数组
    for i in range(bboxes_np.shape[2]):  # 遍历每个位置
        for j in range(bboxes_np.shape[3]):
            # 获取边界框坐标
            x_center, y_center, width, height = bboxes_np[0, :, i, j]
            # 转换为左上角坐标
            x = x_center - width / 2
            y = y_center - height / 2
            # 绘制边界框
            rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')
            plt.gca().add_patch(rect)

    plt.show()

    # 可视化类别得分和置信度 - 使用热图
    class_scores_np = class_scores.detach().numpy()  # 分离并转换为numpy数组
    confidences_np = confidences.detach().numpy()  # 分离并转换为numpy数组
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(class_scores_np[0, 0, :, :], cmap='hot', interpolation='nearest')
    ax[0].set_title('Class Scores')
    ax[1].imshow(confidences_np[0, 0, :, :], cmap='hot', interpolation='nearest')
    ax[1].set_title('Confidences')
    plt.show()


# 继续使用您之前定义的 DetectionHead 类
# ...

# 创建检测头网络
num_classes = 3  # 假设有3个类别，如车辆、行人和交通信号灯
head = DetectionHead(256, num_classes)

# 前向传播 - 使用YOLO主干网络的输出
bboxes, class_scores, confidences = head(backbone_output)


# 这里您可以使用 visualize_output 函数来可视化结果
# 注意，这需要您的原始输入图像进行可视化对比
visualize_output(transformed_image, bboxes, class_scores, confidences)
