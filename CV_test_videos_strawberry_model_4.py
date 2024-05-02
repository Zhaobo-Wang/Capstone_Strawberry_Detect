import cv2  
import numpy as np
import tensorflow as tf
import time 
import serial


interpreter = tf.lite.Interpreter(model_path="D:/McMaster/4OI6/strawberry_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


cap = cv2.VideoCapture(0)  


'''
arduino = serial.Serial('COM6', 115200, timeout=1)
time.sleep(2)
'''
  

'''
# ESP32-CAM的IP地址和端口号
ip_address = 'ESP32_CAM_IP_ADDRESS'
port = 'ESP32_CAM_PORT'
 
# 视频流的URL
stream_url = f'http://{ip_address}:{port}/stream'
'''

try:
    while True:
        time.sleep(0.5) 
        ret, frame = cap.read()
        if not ret:
            break  

        image = cv2.resize(frame, (input_details[0]['shape'][2], input_details[0]['shape'][1]))
        image = np.expand_dims(image, axis=0).astype(np.float32)

        interpreter.set_tensor(input_details[0]['index'], image)
        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details[0]['index'])

        if output_data <= 0.25:
            #arduino.write(b'H') 
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 6)
            print(f"Sent value is {b'H'}, MODEL data value is {output_data}")
        else:       
            #arduino.write(b'L')
            print(f"Sent value is {b'L'}, Model data value is {output_data}")
                            
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  
            break

except KeyboardInterrupt:
    pass

cap.release()
cv2.destroyAllWindows()
