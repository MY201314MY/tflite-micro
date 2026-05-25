import os
import numpy as np
import tensorflow as tf
from absl import app
from absl import flags
from absl import logging

FLAGS = flags.FLAGS

flags.DEFINE_integer("epochs", 20, "number of epochs to train the model.")
flags.DEFINE_string("save_dir", "/tmp/lstm_trained_model", "the directory to save.")
flags.DEFINE_boolean("quantize", True, "convert and save the full integer (int8) quantized model.")

def create_cnn_model():
  """创建一个专为 MCU 优化、无硬展开死重的微型 CNN 模型"""
  model = tf.keras.models.Sequential([
      # 输入层 (Batch, 28, 28, 1)
      tf.keras.layers.Input(batch_shape=(1, 28, 28, 1), name="input"),
      # 极其轻量的卷积核，参数量极低
      tf.keras.layers.Conv2D(8, (3, 3), activation='relu', name="conv1"),
      tf.keras.layers.MaxPooling2D((2, 2), name="pool1"),
      tf.keras.layers.Flatten(name="flatten"),
      # 最终分类
      tf.keras.layers.Dense(10, activation='softmax', name="output")
  ])
  model.compile(optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"])
  model.summary()
  return model

def get_train_data():
  (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
  x_train = x_train / 255.0
  # 扩展出通道维 (28, 28) -> (28, 28, 1)
  x_train = np.expand_dims(x_train, axis=-1).astype(np.float32)
  return (x_train, y_train)

def main(_):
  x_train, y_train = get_train_data()
  
  # 1. 建立并训练模型 (因 Batch 固定为 1，直接用单样本或通过 tf.data 训练，这里演示单样本适配)
  model = create_cnn_model()
  model.fit(x_train, y_train, epochs=FLAGS.epochs, batch_size=1, validation_split=0.2)

  if not os.path.exists(FLAGS.save_dir):
    os.makedirs(FLAGS.save_dir)

  # 2. 转换 Float32 版本
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  tflite_model = converter.convert()
  with open(os.path.join(FLAGS.save_dir, "mnist_cnn.tflite"), "wb") as f:
    f.write(tflite_model)

  # 3. 转换真正的全整型 INT8 瘦身版本
  if FLAGS.quantize:
    def representative_dataset_gen():
      # 提供 100 个样本用于激活值校准
      for data in x_train[:100]:
        yield [np.expand_dims(data, axis=0)]

    quant_converter = tf.lite.TFLiteConverter.from_keras_model(model)
    quant_converter.optimizations = [tf.lite.Optimize.DEFAULT]
    quant_converter.representative_dataset = representative_dataset_gen
    # 强制全整型转换，适合 TFLite Micro 部署
    quant_converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    quant_converter.inference_input_type = tf.int8
    quant_converter.inference_output_type = tf.int8
    
    quant_tflite_model = quant_converter.convert()
    with open(os.path.join(FLAGS.save_dir, "mnist_cnn_quant.tflite"), "wb") as f:
      f.write(quant_tflite_model)
      
    logging.info("极致全量化 CNN 模型已保存在: %s/mnist_cnn_quant.tflite", FLAGS.save_dir)

if __name__ == "__main__":
  app.run(main)