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

# 顺从官方设计：直接导入标准的 runtime
from tflite_micro.python.tflite_micro import runtime

FLAGS = flags.FLAGS
flags.DEFINE_string("model_path", "/tmp/lstm_trained_model/mnist_cnn_quant.tflite", "model path.")
flags.DEFINE_string("img_path", "", "image path.")

def main(_):
  # 路径增强检查
  if not os.path.exists(FLAGS.model_path) or not os.path.exists(FLAGS.img_path):
    print(f"\n❌ [路径错误] 无法访问输入文件！")
    print(f"尝试读取的 model_path: {FLAGS.model_path}")
    print(f"尝试读取的 img_path: {FLAGS.img_path}")
    raise ValueError("Model or Image file does not exist.")

  # 官方原装加载方式
  interpreter = runtime.Interpreter.from_file(FLAGS.model_path)
  
  # 1. 获取输入/输出量化参数
  in_details = interpreter.get_input_details(0)
  in_scale, in_zero_point = in_details["quantization_parameters"]["scales"][0], in_details["quantization_parameters"]["zero_points"][0]
  
  out_details = interpreter.get_output_details(0)
  out_scale, out_zero_point = out_details["quantization_parameters"]["scales"][0], out_details["quantization_parameters"]["zero_points"][0]

  # 2. 读取图片并强制重采样为 28x28（🎯 核心修复：干掉 size 100 报错）
  image = Image.open(FLAGS.img_path).convert('L')
  if image.size != (28, 28):
    # 使用抗锯齿高质量缩放为标准的 28x28
    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    
  raw_val = np.asarray(image, dtype=np.float32)

  # 3. 使用与 C++ 完全相同的合并乘法系数和四舍五入逻辑
  inv_scale_255 = 1.0 / (255.0 * in_scale)
  zero_point_f = float(in_zero_point)
  
  # 一步到位计算量化浮点值
  quantized_float = (raw_val * inv_scale_255) + zero_point_f
  
  # 严格执行 C++ 的 std::round() 和 std::max/min 边界裁剪
  quantized_float = np.clip(quantized_float, -128.0, 127.0)
  quantized_int8 = np.round(quantized_float).astype(np.int8)

  # 4. 变形为单样本 4D 形状 (1, 28, 28, 1) —— 此时必为 784 像素，绝对不会再报错
  data = quantized_int8.reshape((1, 28, 28, 1))

  # 5. 推理
  interpreter.set_input(data, 0)
  interpreter.invoke()
  raw_output = interpreter.get_output(0)[0]

  # 6. 反量化输出以获取准确概率
  probabilities = out_scale * (raw_output.astype(np.float32) - out_zero_point)
  probabilities = np.clip(probabilities, 0.0, 1.0)

  # 7. 打印每个数字的详情
  print("\n=== Model Output Details (Python 官方标准版) ===")
  for i in range(10):
    marker = " ***" if probabilities[i] > 0.1 else ""
    print(f"Digit {i}: raw={raw_output[i]:4d} -> prob={probabilities[i]:.4f}{marker}")
  print(f"Probability sum: {np.sum(probabilities):.4f}")

  # 8. 输出最终结果
  predicted_category = np.argmax(probabilities)
  logging.info("🎯 Model predicts the image as %i with probability %.2f%%", 
               predicted_category, probabilities[predicted_category] * 100)

if __name__ == "__main__":
  app.run(main)