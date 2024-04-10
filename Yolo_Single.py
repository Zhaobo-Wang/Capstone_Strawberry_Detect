from ultralytics import YOLO

model = YOLO('yolov8n.yaml')

model = YOLO('yolov8n.pt')

results = model.train(data='D:/McMaster/4OI6/strawberry.yaml', epochs=3)

results = model.val()

results = model('https://ultralytics.com/images/bus.jpg')

success = model.export(format='onnx')



