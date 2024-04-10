#### Module:

**Computer Vision and Real-Time Monitoring Website Module**

This module harnesses the power of computer vision to detect strawberries in a real-time video stream. Utilizing a camera, it employs advanced image processing techniques for accurate identification and mapping of target objects. Key tools used in this module include VS Code, Python, OpenCV, Flask, Socket.IO, NumPy, JavaScript, CSS, HTML, and React.

**Part 1: Real-Time Video Processing and Object Labeling (25% Weight)**

In this section, the system captures live video and collaborates seamlessly with the camera hardware. Each frame undergoes processing to identify strawberries, label them, and draw bounding boxes. The model analyzes the visual data, outputting results in JSON format, which includes identifiers, confidence scores, and spatial coordinates (e.g., `{id: "1", confidence: "95%", x: "height", y: "width"}`). This data is critical for accurately outlining specific areas within each frame.

**Part 2: Data Compression and Backend Integration (25% Weight)**

Focusing on data efficiency, this part uses JPEG compression to optimize video stream bandwidth. The backend, powered by Python Flask, handles the management of both the model’s output data and the compressed video feeds. It ensures smooth handling and integration of various data streams.

**Part 3: API Development and Frontend Interaction (50% Weight)**

Our backend infrastructure includes multiple API endpoints such as `app.route("/video")`, `app.route("/model")`, and `app.route("/")`, all facilitated by Flask. Real-time data communication to the frontend is maintained via WebSocket.io. The frontend design, developed with React.js, includes dynamic routing for URL paths and integrates JavaScript for fetching and displaying weather API data. The combination of CSS and JavaScript in the frontend design focuses on delivering a user-friendly and responsive monitoring interface.



**Computer Vision and Real-Time Monitoring Website Module**



Video Demo:

[Computer_Vision_Real_Time_Monitoring_Part1 - YouTube](https://youtu.be/uEbbFWbtMcs)

[Algorithm_Math_Part_Video_Part2 - YouTube](https://youtu.be/Nz6r4xxmAvw)

Github Code: 

[GitHub - Zhaobo-Wang/Capstone_Strawberry_Detect](https://github.com/Zhaobo-Wang/Capstone_Strawberry_Detect)





**Provided work hours and percentage of completion for each item:**



Computer Vision Module Integration and Understanding 80 hrs

Traditional Methods for Computer Vision 80 hrs

Web Architect Design 120 hrs

Web Build ( Including Backend and Frontend ) 180 hrs

Web Design(Responsive Design) 20hrs



**Describe what you learned from the item/module**



Prior to embarking on this capstone project, my experience with computer vision was quite limited. However, the journey through this project has been enlightening and rich with learning. I've gained practical skills in video stream processing using Python, tackling the complexities inherent in video data by capturing and analyzing each frame. Furthermore, I've developed an understanding of hardware communication, such as between Arduino and a computer, or a camera and a computer. This includes various communication methods like COM serial communication and wireless approaches, including socket.io and WiFi communication using an Arduino board.

My skills in Python have been further enhanced through its use in serial communication with Arduino, and I've delved into traditional computer vision techniques for object detection using gradients, Sobel filters, and Canny edge detection.

Another significant area of learning has been in utilizing TensorFlow for model development, managing model's JSON data, and data compression techniques. Additionally, I've learned to integrate Python back-end processes with front-end development, creating dynamic URL paths for websites and enabling device accessibility through IP addresses. This comprehensive experience has not only broadened my technical expertise but also strengthened my problem-solving skills and adaptability in the field of computer vision.



**Provide the problems you encountered and the solutions you applied to solve the problems**



Initially, in the API Development and Frontend Interaction part of the module, I encountered a significant bottleneck while transmitting model results to the frontend. The method used was the Fetch API, which, despite its simplicity and wide use, proved to be inefficient for our needs. The primary issue was the latency and the occasional loss of data continuity in transmitting real-time video analysis results. This inefficiency was particularly problematic due to the real-time nature of our application, where every millisecond counts for accuracy and user experience.

**Implemented Solution: Transition to WebSocket.IO for Enhanced Real-Time Data Transmission**

To resolve this issue, a strategic shift was made from the traditional Fetch API to using WebSocket.IO for data transmission. Here's how this solution was implemented and refined:

1. **Integration of WebSocket.IO:** WebSocket.IO was integrated into our system as it provides full-duplex communication channels over a single TCP connection. This meant that data could be sent back and forth simultaneously without the need for multiple HTTP requests, significantly reducing latency.

2. **Optimization of Data Packets:** I optimized the size and frequency of data packets transmitted via WebSocket.IO. By sending smaller, more manageable packets at a more frequent rate, we ensured that the data was up-to-date and reflective of the real-time situation.

3. **Frontend and Backend Synchronization:** Adjustments were made in both the backend (Python Flask) and frontend (React.js) to ensure they were fully compatible with WebSocket.IO communication. Special attention was given to the handling of WebSocket connections to maintain stability and efficiency.

4. **Testing and Validation:** Rigorous testing was conducted to validate the effectiveness of WebSocket.IO in our application. This included stress testing under various network conditions to ensure consistency and reliability.

5. **Continuous Monitoring and Adjustment:** Following the implementation, continuous monitoring was set up to observe the system’s performance. This allowed for real-time adjustments and optimizations to be made, ensuring the system's responsiveness and accuracy.

As a result of these changes, the data transmission from the backend to the frontend became significantly more efficient and reliable. The real-time video analysis results were now being transmitted seamlessly, greatly enhancing the module's performance and the overall user experience.



**Provide the reasons for and the weight of uncompleted items**



**Uncompleted Item: Wireless Transmission Between Computer and Arduino**

1. **Risk Management:** Implementing wireless transmission introduced potential risks, such as data security concerns and increased chances of interference or connection instability. These risks needed to be thoroughly assessed and mitigated, which would have required additional time and resources.

2. **Project Scope and Focus:** The primary focus of the module was on computer vision and real-time data processing. While wireless transmission between the computer and Arduino would be a valuable addition, it was not essential for achieving the primary objectives of the project.


