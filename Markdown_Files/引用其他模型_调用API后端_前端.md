引用他人模型可以用

先看后端
from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
 api_url="https://detect.roboflow.com",
 api_key="dAvqulL8VqM1B2uZvArr"
)


result = CLIENT.infer(frame, model_id="strawberries-unlimited/3")

这样print results 直接返回json数据

将后端数据传到前端

socketio = SocketIO(app)

# Emit the result to the frontend
socketio.emit('model_result', {'data': result})

前端接收,这样socket传result的值可以被接收到

<script src="//cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.0/socket.io.js"></script>
<script type="text/javascript">
    var socket = io.connect('http://' + document.domain + ':' + location.port);

    socket.on('model_result', function (msg) {
        var resultsDiv = document.getElementById('modelResults');
        resultsDiv.innerHTML = formatModelResults(msg.data);
    });