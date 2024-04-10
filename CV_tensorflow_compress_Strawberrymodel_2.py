import numpy as np
import tensorflow as tf

# 加载TFLite模型并分配张量（tensor）
interpreter = tf.lite.Interpreter(model_path="D:/McMaster/4OI6/strawberry_model.tflite")
interpreter.allocate_tensors()

# 获取输入和输出张量的详细信息
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 测试模型
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# 获取预测结果
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)  # 根据模型和任务，解析这些数据
