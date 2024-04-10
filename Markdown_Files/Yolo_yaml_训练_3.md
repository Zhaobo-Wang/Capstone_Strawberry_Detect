通过这个方法训练，得到最后的训练图像pic

from ultralytics import YOLO

# Create a new YOLO model from scratch

model = YOLO('yolov8n.yaml')

# Load a pretrained YOLO model (recommended for training)

model = YOLO('yolov8n.pt')

# Train the model using the 'coco128.yaml' dataset for 3 epochs

results = model.train(data='D:/McMaster/4OI6/strawberry.yaml', epochs=3)

# Evaluate the model's performance on the validation set

results = model.val()

# Perform object detection on an image using the model

results = model('https://ultralytics.com/images/bus.jpg')

# Export the model to ONNX format

success = model.export(format='onnx')



训练过程截图



![04407fd5-0242-4739-a694-2c0d7704a728](file:///D:/TypeDown_Screenshot/04407fd5-0242-4739-a694-2c0d7704a728.png)


