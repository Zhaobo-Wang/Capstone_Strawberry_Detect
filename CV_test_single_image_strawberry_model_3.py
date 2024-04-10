import numpy as np
import tensorflow as tf  
from PIL import Image
import serial
import time

# 加载 TFLite 模型并分配张量（tensors）
interpreter = tf.lite.Interpreter(model_path="D:/McMaster/4OI6/strawberry_model.tflite")
interpreter.allocate_tensors()

# 获取输入和输出张量
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 测试模型
image = Image.open('D:/McMaster/4OI6/testModelDataset/real_strawberry_1.jpg').resize(
    (input_details[0]['shape'][2], input_details[0]['shape'][1]))
image = np.expand_dims(image, axis=0)  # 增加批量维度
image = np.array(image, dtype=np.float32)  # 确保输入数据类型匹配

# 将测试图像输入模型
interpreter.set_tensor(input_details[0]['index'], image)

# 运行推理
interpreter.invoke()

# 获取推理结果
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)  # 输出结果，根据模型的不同，这里可以是类别、概率等


"""

# 替换'COM6'Arduino的串行端口。'115200'是波特率，应与Arduino代码中设置的相匹配。
# timeout=1表示如果1秒内没有数据到来，则停止等待。

arduino = serial.Serial('COM6', 115200, timeout=1)
time.sleep(2)  

#send_to_arduino(data)是一个函数，用于将字符串数据发送到Arduino。
# data.encode()将字符串转换为字节，因为串行传输是以字节形式进行的。

try:
    while True:
        # 发送数据到Arduino
        arduino.write(b'H') 
        print(f"Sent value is {b'H'}")
        time.sleep(1)       
        arduino.write(b'L')
        print(f"Sent value is {b'L'}")
        time.sleep(1)       
except KeyboardInterrupt:
    arduino.close()

"""
