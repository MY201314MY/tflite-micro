#include <iostream>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <chrono>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "mnist_cnn_quant_model_data.h"
#include "test_input_digital.h"


uint8_t *p_sample_data = sample9_data;

// 分配 64KB 保证内存安全
constexpr int kTensorArenaSize = 64 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

// --- 调试函数保留 ---
void print_image_stats(uint8_t* image, int size, const char* name) {
    std::cout << "\n=== " << name << " Statistics ===" << std::endl;
    int min_val = 255, max_val = 0;
    long sum = 0;
    int non_zero = 0;
    for (int i = 0; i < size; ++i) {
        uint8_t val = image[i];
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
        sum += val;
        if (val > 50) non_zero++;
    }
    float mean = static_cast<float>(sum) / size;
    float density = static_cast<float>(non_zero) / size * 100.0f;
    std::cout << "  Min value: " << min_val << std::endl;
    std::cout << "  Max value: " << max_val << std::endl;
    std::cout << "  Mean value: " << mean << std::endl;
    std::cout << "  Non-dark pixels (>50): " << non_zero << " / " << size << " (" << density << "%)" << std::endl;
}

void print_image_ascii(uint8_t* image, int width, int height) {
    std::cout << "\n=== Image ASCII Representation ===" << std::endl;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            uint8_t pixel = image[y * width + x];
            if (pixel > 200) std::cout << "██";
            else if (pixel > 150) std::cout << "▓▓";
            else if (pixel > 100) std::cout << "▒▒";
            else if (pixel > 50) std::cout << "░░";
            else std::cout << "  ";
        }
        std::cout << std::endl;
    }
}

void print_quantization_debug(const uint8_t* raw_image, const int8_t* quantized_image, float scale, int zero_point, int size) {
    std::cout << "\n=== Quantization Debug (First 10 Non-Zero Pixels) ===" << std::endl;
    std::cout << "Quant params: scale=" << scale << ", zero_point=" << zero_point << std::endl;
    int printed_count = 0;
    for (int i = 0; i < size && printed_count < 10; ++i) {
        if (raw_image[i] == 0) continue; 
        
        float raw_val = static_cast<float>(raw_image[i]);
        float normalized = raw_val / 255.0f;
        float quantized_float = (normalized / scale) + zero_point;
        int8_t calc_quant = static_cast<int8_t>(std::max(-128.0f, std::min(127.0f, std::round(quantized_float))));
        
        std::cout << "  Idx[" << i << "] Raw: " << std::setw(3) << (int)raw_image[i] 
                  << " -> Norm: " << std::fixed << std::setprecision(3) << normalized 
                  << " -> Calc int8: " << std::setw(4) << (int)calc_quant 
                  << " | Actual Buffer: " << std::setw(4) << (int)quantized_image[i] << std::endl;
        printed_count++;
    }
}

void print_output_details(int8_t* output_buffer, float scale, int zero_point) {
    std::cout << "\n=== Model Output Details ===" << std::endl;
    std::cout << "Output quant params: scale=" << scale << ", zero_point=" << zero_point << std::endl;
    float sum = 0.0f;
    for (int i = 0; i < 10; ++i) {
        float probability = (static_cast<float>(output_buffer[i]) - zero_point) * scale;
        probability = std::max(0.0f, std::min(1.0f, probability));
        sum += probability;
        std::cout << "Digit " << i << ": raw=" << std::setw(4) << (int)output_buffer[i] 
                  << " -> prob=" << std::fixed << std::setprecision(4) << probability;
        if (probability > 0.1) std::cout << " ***";
        std::cout << std::endl;
    }
    std::cout << "Probability sum: " << sum << " (should be ~1.0)" << std::endl;
}

int main(int argc, char* argv[]) {
    tflite::InitializeTarget();

    std::cout << "\n========================================" << std::endl;
    std::cout << "=== MNIST CNN Model Debug Version ===" << std::endl;
    std::cout << "========================================\n" << std::endl;

    const tflite::Model* model = tflite::GetModel(_mnist_cnn_quant_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        std::cerr << "Model schema version mismatch!" << std::endl;
        return 1;
    }
    std::cout << "✓ Model loaded successfully" << std::endl;

    // 💡 将容量扩展至 8，并补上 AddMul() 算子
    static tflite::MicroMutableOpResolver<8> micro_op_resolver;
    micro_op_resolver.AddConv2D();
    micro_op_resolver.AddMaxPool2D();
    micro_op_resolver.AddReshape();
    micro_op_resolver.AddFullyConnected();
    micro_op_resolver.AddSoftmax();
    micro_op_resolver.AddAdd();        // 兜底量化 Bias/Rescale 加法
    micro_op_resolver.AddQuantize();   // 兜底输入硬量化转换
    micro_op_resolver.AddMul();        // 💡 核心新增：解决 "Didn't find op for builtin opcode 'MUL'" 报错

    static tflite::MicroInterpreter interpreter(
        model, micro_op_resolver, tensor_arena, kTensorArenaSize);
    
    if (interpreter.AllocateTensors() != kTfLiteOk) {
        std::cerr << "Failed to allocate tensor memory!" << std::endl;
        return 1;
    }

    TfLiteTensor* input = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);
    
    float input_scale = input->params.scale;
    int input_zero_point = input->params.zero_point;
    float output_scale = output->params.scale;
    int output_zero_point = output->params.zero_point;
    
    // 1. 打印原始图像统计
    print_image_stats(p_sample_data, 784, "Raw Input Image");
    print_image_ascii(p_sample_data, 28, 28);
    
    // 2. 预处理量化
    int8_t* input_buffer = tflite::GetTensorData<int8_t>(input);
    int input_size = input->bytes;

    const float inv_scale_255 = 1.0f / (255.0f * input_scale);
    const float zero_point_f = static_cast<float>(input_zero_point);
    
    for (int i = 0; i < input_size; ++i) {
        float raw_val = static_cast<float>(p_sample_data[i]);
        
        // 高效量化公式
        float quantized_float = (raw_val * inv_scale_255) + zero_point_f;
        
        quantized_float = std::max(-128.0f, std::min(127.0f, quantized_float));
        input_buffer[i] = static_cast<int8_t>(std::round(quantized_float));
    }
    
    // 3. 打印量化过程参数
    print_quantization_debug(p_sample_data, input_buffer, input_scale, input_zero_point, input_size);
    
    // ===== 模型推理 =====
    TfLiteStatus invoke_status = interpreter.Invoke();
    if (invoke_status != kTfLiteOk) {
        std::cerr << "ERROR: Model inference failed!" << std::endl;
        return 1;
    }
    
    // ===== 输出详情 =====
    int8_t* output_buffer = tflite::GetTensorData<int8_t>(output);
    print_output_details(output_buffer, output_scale, output_zero_point);
    
    int max_digit = 0;
    float max_probability = -1.0f;
    for (int i = 0; i < 10; ++i) {
        float prob = (static_cast<float>(output_buffer[i]) - output_zero_point) * output_scale;
        prob = std::max(0.0f, std::min(1.0f, prob));
        if (prob > max_probability) {
            max_probability = prob;
            max_digit = i;
        }
    }
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "🎯 FINAL PREDICTION: " << (max_probability * 100) 
              << "% likely the digit [" << max_digit << "]" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}
