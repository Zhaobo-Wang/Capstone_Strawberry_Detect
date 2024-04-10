视频流返回Web



把每一帧生成视频流返回Web端口

```
        # 处理帧，编码为JPEG格式
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # 生成MJPEG流
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
```

Use Flask框架处理后端视频

```
@app.route('/')
def index():
    return render_template('index.html')  # 返回主页面

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

```


