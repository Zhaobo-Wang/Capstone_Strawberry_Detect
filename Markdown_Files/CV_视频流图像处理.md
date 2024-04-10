**CV 视频流图像识别**

本质和单张图像识别没有区别

因为它是流媒体截获每一帧的图像，以下是流媒体截获的代码

```
cap = cv2.VideoCapture(0)  # 0是默认摄像头的索引

while True:
        time.sleep(0.5)  # 每0.5秒处理一次
        ret, frame = cap.read()
        if not ret:
            break  # 如果没有捕获到帧，跳出循环

        # 预处理帧
        image = cv2.resize(frame, (input_details[0]['shape'][2], input_details[0]['shape'][1]))
        image = np.expand_dims(image, axis=0).astype(np.float32)


```

通过以上代码，可以将流媒体中的视频转变成一帧帧的图像

再把图像取出来，对比模型，做图像识别的分析


