import cv2
import numpy as np
import time 
import serial
from flask import Flask, render_template, Response
from inference_sdk import InferenceHTTPClient
from flask_socketio import SocketIO, emit

app = Flask(__name__)
socketio = SocketIO(app)
cap = cv2.VideoCapture(0)
#ip_address = "192.168.43.173"
#port = "81"
#stream_url = f'http://{ip_address}:{port}/stream'
#cap = cv2.VideoCapture(stream_url)  

#arduino = serial.Serial('COM6', 115200, timeout=1)
#time.sleep(2)
strawberry_threshold = 0.6


def detect_strawberries(image):

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_ripe = np.array([0, 100, 50]) 
    upper_ripe = np.array([10, 255, 255])  

    ripe_mask = cv2.inRange(hsv_image, lower_ripe, upper_ripe)

    kernel = np.ones((3,3), np.uint8)  
    ripe_mask = cv2.morphologyEx(ripe_mask, cv2.MORPH_OPEN, kernel)
    ripe_mask = cv2.morphologyEx(ripe_mask, cv2.MORPH_CLOSE, kernel)

    ripe_contours, _ = cv2.findContours(ripe_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_strawberries = []

    for contour in ripe_contours:
        area = cv2.contourArea(contour)
        if 100 < area < 2000: 
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            if 0.5 < aspect_ratio < 1.5:  
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, 255, -1)
                _, std_dev = cv2.meanStdDev(image, mask=mask)
                std_dev_avg = np.mean(std_dev)
                if std_dev_avg > 10:  
                    detected_strawberries.append(((x, y, w, h), "Strawberry"))
    return detected_strawberries

CLIENT = InferenceHTTPClient(
 api_url="https://detect.roboflow.com",
 api_key="dAvqulL8VqM1B2uZvArr"
)

def generate_frames():
    while True:
        success, frame = cap.read()  
        if not success:
            break


        try:
            result = CLIENT.infer(frame, model_id="strawberries-unlimited/3")

            if result:
                print("Call successful and data received.")
                # Emit the result to the frontend
                socketio.emit('model_result', {'data': result})
            else:
                print("Call was successful but no data was returned.")
        except Exception as e:
            print(f"call failed: {e}")

        if 'predictions' in result:
            for pred in result['predictions']:
                x = int(pred['x'] - pred['width']/2)  
                y = int(pred['y'] - pred['height']/2)
                w = int(pred['width'])
                h = int(pred['height'])
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                label = f"{pred['class']} {pred['confidence']:.2%}"  

                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                health_label = pred['class']
                health_level = pred['confidence']
                label_text = f"{health_label} {health_level:.2%}"  

                cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        '''
        if health_level <= 0.8:
            #arduino.write(b'H')  
            print("Strawberries not healthy")
           
        else:
            #arduino.write(b'L')  
            print("Strawberries healthy")
        '''

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/model')
def model():
    return render_template('model.html') 

@app.route('/team')
def team():
    return render_template('team.html') 

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)


