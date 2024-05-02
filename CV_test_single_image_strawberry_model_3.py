import numpy as np
import tensorflow as tf  
from PIL import Image
import serial
import time

interpreter = tf.lite.Interpreter(model_path="D:/McMaster/4OI6/strawberry_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


image = Image.open('D:/McMaster/4OI6/testModelDataset/real_strawberry_1.jpg').resize(
    (input_details[0]['shape'][2], input_details[0]['shape'][1]))
image = np.expand_dims(image, axis=0)  
image = np.array(image, dtype=np.float32)  

interpreter.set_tensor(input_details[0]['index'], image)

interpreter.invoke()


output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)  

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
