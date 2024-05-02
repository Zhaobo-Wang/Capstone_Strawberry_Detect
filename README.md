# AI-Driven Agricultural Monitoring System

### Description:

This project leverages computer vision and deep learning technologies to develop a comprehensive real-time monitoring and health assessment system for strawberries. Using a YOLOv8 model, the system not only detects strawberries but also classifies their health status to automate irrigation and pesticide application, ensuring optimal agricultural management through a sophisticated web interface.


### Monitoring Interface:

![image1 description](md_Image/Image5.png)
Implemented a high-performance real-time video processing system for strawberry detection; 
Utilizes advanced image processing techniques with OpenCV and Python to detect strawberries in live video feeds;
Optimized video stream bandwidth through advanced data compression techniques;
Developed a responsive front-end interface, integrating WebSocket communication to ensure seamless real-time data transfer between the backend and frontend, facilitating immediate updates and interactive user engagement

![image1 description](md_Image/Image6.png)
Real-Time data (JSON format) return to user monitoring screen;
Features multiple API endpoints managed via Flask to handle real-time data transactions and system commands;
Offers a dynamic and user-friendly front-end, built with React.js, which provides real-time data visualization and interaction through WebSocket communication;

### Model Algorithm Page:

![image1 description](md_Image/Image4.png)
Leveraged a YOLOv8 deep learning model to differentiate between healthy / moldy strawberry;

### Model Training Shown:

![image1 description](md_Image/Image2.png)
Training on a dataset of 3225 annotated images to ensure high precision and recall across multiple classes;
Deploys the trained model for real-time and batch inference, enhancing the capability of monitoring systems to detect and react to health issues immediately

![image1 description](md_Image/Image3.png)
The system automatically initiates real-time irrigation based on health detection via COM serial communication to Arduino;
Integrates COM serial communication to automate irrigation and pesticide spraying based on the detected health condition of the strawberries, significantly improving response times and effectiveness of treatment.

### Project Impact:

This system transforms traditional strawberry farming by integrating artificial intelligence with agricultural practices, leading to increased efficiency, reduced waste, and improved crop health through timely interventions. It stands as a testament to the potential of AI in revolutionizing farming and sustainability practices.
