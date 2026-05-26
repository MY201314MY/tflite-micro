import os
import numpy as np
import tensorflow as tf
from absl import app
from absl import flags
from absl import logging

FLAGS = flags.FLAGS

flags.DEFINE_integer("epochs", 20, "number of epochs to train the model.")
flags.DEFINE_string("save_dir", "/tmp/cnn_trained_model", "the directory to save.")
flags.DEFINE_boolean("quantize", True, "convert and save the full integer (int8) quantized model.")

def create_cnn_model():
    """创建一个专为 C++ / TFLite Micro 优化、从根源杜绝动态 SHAPE 算子的静态 CNN 模型"""
    model = tf.keras.models.Sequential([
        # 💡 核心锁死：使用标准 InputLayer 并锁定 batch_size=1
        tf.keras.layers.InputLayer(input_shape=(28, 28, 1), batch_size=1, name="input"),
        
        # 极其轻量的卷积与池化
        tf.keras.layers.Conv2D(8, (3, 3), activation='relu', name="conv1"),
        tf.keras.layers.MaxPooling2D((2, 2), name="pool1"),
        
        # 💡 绝对不用 Flatten()，直接用明确的整型元组 (1352,) 写死 Reshape
        tf.keras.layers.Reshape((1352,), name="reshape_node"),
        
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
    
    # 引入 EarlyStopping 防止无意义挂机
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    
    # 1. 建立并训练模型
    model = create_cnn_model()
    model.fit(x_train, y_train, 
              epochs=FLAGS.epochs, 
              batch_size=1, 
              validation_split=0.2,
              callbacks=[early_stop])

    if not os.path.exists(FLAGS.save_dir):
        os.makedirs(FLAGS.save_dir)

    # =====================================================================
    # 💡 核心改进：提取完全死锁形状的 Concrete Function（具象函数）
    # =====================================================================
    run_model = tf.function(lambda x: model(x))
    # 使用包含固定 Batch Size 维度的 TensorSpec 进行绑定：[1, 28, 28, 1]
    concrete_func = run_model.get_concrete_function(
        tf.TensorSpec([1, 28, 28, 1], model.inputs[0].dtype)
    )

    # 2. 转换 Float32 版本 (基于静态 Concrete Function)
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func], model)
    
    # 开启 MLIR 静态折叠强力开关
    converter.experimental_new_converter = True
    converter._experimental_lower_tensor_list_ops = True
    
    tflite_model = converter.convert()
    with open(os.path.join(FLAGS.save_dir, "mnist_cnn.tflite"), "wb") as f:
        f.write(tflite_model)

    # 3. 转换真正的全整型 INT8 瘦身版本
    if FLAGS.quantize:
        def representative_dataset_gen():
            # 提供 100 个样本用于激活值校准
            for data in x_train[:100]:
                # 严格对齐 (1, 28, 28, 1)
                yield [np.expand_dims(data, axis=0)]

        # 同样基于静态 Concrete Function 建立量化转换器
        quant_converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func], model)
        quant_converter.optimizations = [tf.lite.Optimize.DEFAULT]
        quant_converter.representative_dataset = representative_dataset_gen
        
        # 强制全整型转换，确保无 float 残留
        quant_converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        quant_converter.inference_input_type = tf.int8
        quant_converter.inference_output_type = tf.int8
        
        # 开启 MLIR 静态折叠强力开关
        quant_converter.experimental_new_converter = True
        quant_converter._experimental_lower_tensor_list_ops = True
        
        quant_tflite_model = quant_converter.convert()
        
        output_path = os.path.join(FLAGS.save_dir, "mnist_cnn_quant.tflite")
        with open(output_path, "wb") as f:
            f.write(quant_tflite_model)
            
        logging.info("🔥 完美杜绝 SHAPE 算子的纯静态全量化 CNN 模型已保存在: %s", output_path)

if __name__ == "__main__":
    app.run(main)