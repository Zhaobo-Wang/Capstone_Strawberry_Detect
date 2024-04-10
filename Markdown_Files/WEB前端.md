前端注意事项

1.render template 把html文件都放在template folder里面
对应后端get接口为

@app.route('/')
def index():
    return render_template('index.html')  # 返回主页面

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/model')
def model():
    return render_template('model.html') 


2.图片等静态文件放在static folder里面

3.css有手写+bootstrap两种格式