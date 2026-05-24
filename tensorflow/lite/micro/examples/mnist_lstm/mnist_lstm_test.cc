#include <iostream>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

// 1. Include the official model array definition provided in this directory.
// After verification, the official header uses the array name g_trained_lstm_int8_model_data.
#include "trained_lstm_int8_model_data.h"

int8_t test_image_data[784] = {
-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,
-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,
-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,
-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,
-128,-128,-128,-128,-128,-128,-116, -87,  18,  18, -80,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,
-128,-128,-128,-128,-128,-116,   1, 124, 124, 124, 121,  35,-110,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,
-128,-128,-128,-128,-128,   5, 124, 124, 124, 124, 124, 124, 100, -58,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,
-128,-128,-128,-128,-128, -27, 124, 123,  17, -26, -21, 108, 124, 118,   0,-118,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,
-128,-128,-128,-128,-128,  53, 124,  39,-128,-128,-128, -67, 106, 124, 124,  35,-123,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,
-128,-128,-128,-128,-128, 126, 124, -85,-128,-128,-128,-128, -70,  64, 124, 124,  36,-124,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,
-128,-128,-128,-128,-128,  59, 124, -96,-128,-128,-128,-128,-128, -73, 107, 124, 124, -42,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,
-128,-128,-128,-128,-128,  18, 124, -96,-128, -28,  62, -41, -41, -41,  19, 124, 124,  -5,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,
-128,-128,-128,-128,-128, -34, 124, -50, -88, 119, 124, 124, 124, 124, 124, 124, 124,  94, -44,-113,-128,-128,-128,-128,-128,-128,-128,-128,
-128,-128,-128,-128,-128,-114, -36,-116, -93, 111, 124, 124, 124, 124, 124, 124, 124, 124, 124, 115, -39,-118,-128,-128,-128,-128,-128,-128,
-128,-128,-128,-128,-128,-128,-128,-128,-128, -53,  33,  51, 124, 124, 124, 124, 124, 124, 124, 124, 124,  80, -85,-128,-128,-128,-128,-128,
-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-125,-112,-112, -89, -90,-112,-112,  17, 114, 124, 124,  57, -80,-128,-128,-128,-128,
-128,-128,-128,-128,-128,-128,-128,-128,-128,-108, -70,-128,-128,-128,-128,-128,-128,-128,-128, -70,  80, 124, 124,  55,-128,-128,-128,-128,
-128,-128,-128,-128,-128,-128,-128,-128, -51,  92, 118, -49,-128,-128,-128,-128,-128,-128,-128,-128,-115,  90, 124, 111, -56,-128,-128,-128,
-128,-128,-128,-128,-128,-128,-128,-128, -38, 118, 124, 123, -71,-128,-128,-128,-128,-128,-128,-128,-128, -75, 122, 124,  63,-128,-128,-128,
-128,-128,-128,-128,-128,-128,-128,-128,-128, -12, 124, 124, -69,-128,-128,-128,-128,-128,-128,-128,-128, -29, 123, 124,  17,-128,-128,-128,
-128,-128,-128,-128,-128,-128,-128,-128,-128,-114,  60, 124,  92,  30, -90,-128,-128,-128,-128, -17,  82, 117, 124, 124,  17,-128,-128,-128,
-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-116,  92, 117, 124, 122, 120, 120, 120, 120, 124, 124, 124, 124,  71,-109,-128,-128,-128,
-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128, -33,  55,  99, 124, 124, 124, 124, 124, 124,  66,  -4,-105,-128,-128,-128,-128,
-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-104, -91,  10, -54,  -2, -40, -91,-121,-128,-128,-128,-128,-128,-128,
-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,
-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,
-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,
-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128
};

// 2. Allocate the TFLM tensor arena (64KB for desktop testing).
constexpr int kTensorArenaSize = 64 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

int main(int argc, char* argv[]) {
  // Initialize the TFLM environment.
  tflite::InitializeTarget();

  std::cout << "=== Loading MNIST LSTM model ===" << std::endl;

  // Load the model using the official int8 array from the header.
  const tflite::Model* model = tflite::GetModel(tensorflow_lite_micro_examples_mnist_lstm_trained_lstm_int8_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    std::cerr << "Model schema version does not match!" << std::endl;
    return 1;
  }

  // 3. Register LSTM and all required operator dependencies.
  static tflite::MicroMutableOpResolver<6> micro_op_resolver;
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddSoftmax();
  micro_op_resolver.AddUnidirectionalSequenceLSTM(); // fixed casing

  // 4. Initialize the interpreter.
  static tflite::MicroInterpreter interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize);
  
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    std::cerr << "Failed to allocate tensor memory!" << std::endl;
    return 1;
  }

  // 5. Get input and output tensor structures.
  TfLiteTensor* input = interpreter.input(0);
  TfLiteTensor* output = interpreter.output(0);

  std::cout << "Model initialized successfully!" << std::endl;
  std::cout << "Input shape: " << input->dims->size << "D [";
  for(int i=0; i<input->dims->size; ++i) std::cout << input->dims->data[i] << " ";
  std::cout << "]" << std::endl;

  // 6. Simulate a handwritten digit image input using test data (desktop test).
  int8_t* input_buffer = input->data.int8;
  int input_size = input->bytes;
  std::cout << "buffer size: " << input_size << std::endl;
  for (int i = 0; i < input_size; ++i) {
    input_buffer[i] = test_image_data[i]; // simulate black background
  }



  // 7. Run inference.
  std::cout << "Running model inference (Invoke)..." << std::endl;
  if (interpreter.Invoke() != kTfLiteOk) {
    std::cerr << "Model inference failed!" << std::endl;
    return 1;
  }

  // 8. Reset internal state (LSTM core step).
  interpreter.Reset();

  // 9. Print output information.
  std::cout << "Inference complete! Output probabilities (quantized int8 values):" << std::endl;
  int8_t* output_buffer = output->data.int8;

  float out_scale = output->params.scale;
  int out_zero_point = output->params.zero_point;

  std::cout << std::fixed;
  std::cout.precision(2);

  int max_digit = 0;
  float max_probability = -1.0f; // track the highest probability

  for (int i = 0; i < 10; ++i) {
    // 10. Convert int8 output values to floating-point probabilities in [0.0, 1.0].
    float probability = (static_cast<float>(output_buffer[i]) - out_zero_point) * out_scale;
    
    // Clamp tiny numerical errors so probability stays between 0.0 and 1.0.
    if (probability < 0.0f) probability = 0.0f;
    if (probability > 1.0f) probability = 1.0f;

    std::cout << "Digit " << i << " probability: " << probability << std::endl;

    // Update the best result if this probability is higher.
    if (probability > max_probability) {
      max_probability = probability;
      max_digit = i;
    }
  }

  // 11. Print the final result after the loop.
  std::cout << "\n===============================" << std::endl;
  std::cout << "🎯 Final prediction: this image is " << (max_probability * 100.0f) 
            << "% likely the digit [" << max_digit << "]" << std::endl;
  std::cout << "===============================\n" << std::endl;
  
  std::cout << "=== Desktop C++ validation passed! ===" << std::endl;
  return 0;
}
