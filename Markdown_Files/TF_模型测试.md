在经历tensorflow模型训练与压缩后，已经得到模型 strawberry_model.tflite

接下来, 可以加载模型在.py文件下面, 代码如下

```
# 加载TFLite模型
interpreter = tf.lite.Interpreter(model_path="D:/McMaster/4OI6/strawberry_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
```

这样,Input_details 和 output_details现在都是关于模型的具体值，可以进行进一步的模型应用

和推理结果

```
        # 应用模型
        # 在这里image是通过视频或者图片传回的图像
        # 由此，可以判断image是否符合模型特征
        interpreter.set_tensor(input_details[0]['index'], image)
        interpreter.invoke()

        # 获取推理结果
        output_data = interpreter.get_tensor(output_details[0]['index'])
```

假设说，使用一个草莓模型，那么如果image很像草莓，那么output_data会返回一个非常小的值（小于0.05）

如果，image不像草莓，那么output_data会返回一个大于0.05的值


