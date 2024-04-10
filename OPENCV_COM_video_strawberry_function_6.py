import cv2
import numpy as np
import time 
import serial
from flask import Flask, render_template, Response
from inference_sdk import InferenceHTTPClient
from flask_socketio import SocketIO, emit

app = Flask(__name__)
socketio = SocketIO(app)
cap = cv2.VideoCapture(1)
#ip_address = "192.168.43.173"
#port = "81"
#stream_url = f'http://{ip_address}:{port}/stream'
#cap = cv2.VideoCapture(stream_url)  

#arduino = serial.Serial('COM6', 115200, bytesize=8, parity='N', stopbits=1, timeout=1)

strawberry_threshold = 0.6

CLIENT = InferenceHTTPClient(
 api_url="https://detect.roboflow.com",
 api_key="dAvqulL8VqM1B2uZvArr"
)

def generate_frames():
    while True:
        success, frame = cap.read()  
        if not success:
            break

        mold_detected = False  

        try:
            result = CLIENT.infer(frame, model_id="strawberries-unlimited/3")
            if not result:
                print("Call was successful but no data was returned.")
                send_data_to_arduino(b'L') 
                continue

            socketio.emit('model_result', {'data': result})

            for pred in result.get('predictions', []):
                x = int(pred['x'] - pred['width'] / 2)
                y = int(pred['y'] - pred['height'] / 2)
                w = int(pred['width'])
                h = int(pred['height'])
                health_label = pred['class']
                health_level = pred['confidence']
                label_text = f"{health_label} {health_level:.2%}"

                if health_label == "mold":
                    mold_detected = True
                    rect_color = (255, 0, 0)  # Red
                    text_color = (255, 0, 0)
                    print("Mold detected: Sending HIGH signal")
                else:
                    rect_color = (0, 255, 0)  # Green
                    text_color = (36, 255, 12)

                cv2.rectangle(frame, (x, y), (x + w, y + h), rect_color, 2)
                cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)

            
            if mold_detected:
                send_data_to_arduino(b'H')
            else:
                send_data_to_arduino(b'L')

        except Exception as e:
            print(f"call failed: {e}")
            send_data_to_arduino(b'L')  
            continue

        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            

def send_data_to_arduino(data):
    try:
        arduino = serial.Serial('COM6', 115200, timeout = 1)
        arduino.write(data)
        arduino.close()
    except Exception as e:
        print("Error in connection:", e)



@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/model')
def model():
    return render_template('model.html') 

@app.route('/algorithm')
def algorithm():
    return render_template('algorithm.html') 

@app.route('/team')
def team():
    return render_template('team.html') 

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)






