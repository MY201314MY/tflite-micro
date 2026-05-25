# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# =============================================================================
"""CNN model evaluation for MNIST recognition - Ultra Light Version with Prob"""
import os
import sys

from absl import app
from absl import flags
from absl import logging
import numpy as np
from PIL import Image

from tflite_micro.python.tflite_micro import runtime

FLAGS = flags.FLAGS
flags.DEFINE_string("model_path", "/tmp/lstm_trained_model/mnist_cnn_quant.tflite", "model path.")
flags.DEFINE_string("img_path", "", "image path.")

def main(_):
  if not os.path.exists(FLAGS.model_path) or not os.path.exists(FLAGS.img_path):
    raise ValueError("Model or Image file does not exist.")

  # 1. 加载模型并获取输入/输出量化参数
  interpreter = runtime.Interpreter.from_file(FLAGS.model_path)
  
  in_details = interpreter.get_input_details(0)
  in_scale, in_zero_point = in_details["quantization_parameters"]["scales"][0], in_details["quantization_parameters"]["zero_points"][0]
  
  out_details = interpreter.get_output_details(0)
  out_scale, out_zero_point = out_details["quantization_parameters"]["scales"][0], out_details["quantization_parameters"]["zero_points"][0]

  # 2. 读取图片并处理为 4D 形状 (1, 28, 28, 1)
  image = Image.open(FLAGS.img_path).convert('L')
  data = np.asarray(image, dtype=np.float32) / 255.0
  data = data.reshape((1, 28, 28, 1))

  # 3. 输入数据量化为 INT8
  data = (data / in_scale + in_zero_point).astype(np.int8)

  # 4. 推理
  interpreter.set_input(data, 0)
  interpreter.invoke()
  raw_output = interpreter.get_output(0)[0]

  # 5. 反量化输出以获取准确概率
  probabilities = out_scale * (raw_output.astype(np.float32) - out_zero_point)

  # 6. 输出结果
  predicted_category = np.argmax(probabilities)
  logging.info("Model predicts the image as %i with probability %.2f", 
               predicted_category, probabilities[predicted_category])

if __name__ == "__main__":
  app.run(main)