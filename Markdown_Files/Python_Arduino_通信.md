**Python 传值给 Arduino**

1）使用 pyserial
确保连接Arduino的端口号
Python 用以下方式表示：

```
import serial
import time

# 替换'COM6'Arduino的串行端口。'115200'是波特率，应与Arduino代码中设置的相匹配。
# timeout=1表示如果1秒内没有数据到来，则停止等待。

arduino = serial.Serial('COM6', 115200, timeout=1)
time.sleep(2)  

#send_to_arduino(data)是一个函数，用于将字符串数据发送到Arduino。
# data.encode()将字符串转换为字节，因为串行传输是以字节形式进行的。

def send_to_arduino(data):
    arduino.write(data.encode())
    time.sleep(0.5)  

try:
    while True:
        # 发送数据到Arduino
        arduino.write(b'H')  # 发送高电平信号
        time.sleep(1)       # 等待一秒
        arduino.write(b'L')  # 发送低电平信号
        time.sleep(1)       # 等待一秒
except KeyboardInterrupt:
    # 如果有Ctrl+C被按下，则关闭串行连接
    arduino.close()
```

2）Arduino使用以下方式表示：

```
void setup() {
  // initialize serial communication:
  Serial.begin(115200);
}

void loop{
   if (Serial.available() > 0) {
    char incomingByte = Serial.read();
    // 接收到python的值
  }  
} 
```

3）运行逻辑是先把Arduino的code load到板子中，在run python，让python不停的往Arduino中传值

Arduino IDE 无法开启Serial Port，因为已经被Python占用，只能用Python去监听并且打印传送的值

（**关闭 Arduino IDE**：如果您刚刚使用 Arduino IDE 上传了程序，那么在运行 Python 脚本之前，请关闭 Arduino IDE。Arduino IDE 有可能保持对串口的占用，即使上传完成后。）

1. **断开然后重新连接 Arduino**：有时，简单地将 Arduino 断开再重新连接到电脑可以解决串口占用问题。
2. 或者重启计算机

-----------------------------------------

使用 Python 的 `pyserial` 库与 Arduino 通信时，您需要按照以下顺序进行操作：

1. **编写并上传 Arduino 程序**: 首先，您需要在 Arduino IDE 中编写 Arduino 程序（sketch）。这个程序应当能够处理从 Python 发送的数据，并做出相应的反应。完成编写后，您需要将这个程序上传到 Arduino 开发板上。

2. **编写 Python 脚本**: 接下来，在您的电脑上编写 Python 脚本。这个脚本将使用 `pyserial` 库与 Arduino 通信。在脚本中，您需要指定正确的串口（COM 端口），这是 Arduino 开发板连接到电脑的接口。

3. **运行 Python 脚本**: 确保 Arduino 开发板已经连接到电脑并正确配置了串口后，运行您的 Python 脚本。这时，Python 脚本将通过指定的 COM 端口向 Arduino 发送数据。
   
   ----------------------------------------
   
   
   
   

**COM串口通信的基础知识**

**什么是COM串口通信？**

  COM串口通信（Communication Port）是一种用于计算机和外部设备之间进行数据传输的技术。它使用一个可编程的串行接口，可以实现计算机和外部设备之间的数据传输。

**COM串口通信的工作原理**
  COM串口通信是一种串行接口，它使用一组可编程的信号线来传输数据。它的工作原理是：计算机将数据以字节的形式发送到外部设备，外部设备接收到数据之后，再将数据以字节的形式发送给计算机。



**COM串口通信的优点
COM串口通信具有以下优点：**

它可以实现计算机和外部设备之间的高速数据传输；
它可以实现计算机和外部设备之间的双向通信；
它可以实现多种不同设备之间的通信；
它可以实现计算机和外部设备之间的简单连接；
它可以实现计算机和外部设备之间的可靠通信。



另外一种通讯方式是Socket通讯

通过python Socket传值给arduino
