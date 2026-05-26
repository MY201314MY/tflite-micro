load("@rules_python//python:defs.bzl", "py_binary", "py_test")
load("@tflm_pip_deps//:requirements.bzl", "requirement")
load("//tensorflow/lite/micro:build_def.bzl", "INCOMPATIBLE_WITH_WINDOWS")

cc_binary(
    name = "mnist_lstm_test",
    srcs = [
        "mnist_lstm_test.cc",
        "test_input_digital.h",
        "mnist_cnn_quant_model_data.h",
    ],
    deps = [
        "//tensorflow/lite/micro:micro_framework",
        "//tensorflow/lite/micro:micro_log",
        "//tensorflow/lite/micro:system_setup",
        "//tensorflow/lite/schema:schema_fbs",
    ],
)

py_binary(
    name = "train",
    srcs = ["train.py"],
    deps = [
        requirement("numpy"),
        requirement("tensorflow"),
    ],
)

py_binary(
    name = "ptq",
    srcs = ["ptq.py"],
    deps = [
        requirement("absl_py"),
        requirement("numpy"),
        requirement("tensorflow"),
        "//python/tflite_micro:runtime",
    ],
)

py_binary(
    name = "evaluate",
    srcs = ["evaluate.py"],
    deps = [
        "//python/tflite_micro:runtime",
        requirement("absl_py"),
        requirement("pillow"),
    ],
)

filegroup(
    name = "sample_images",
    srcs = glob(["samples/*.png"]),
)

py_test(
    name = "evaluate_test",
    srcs = ["evaluate_test.py"],
    data = [
        "trained_lstm.tflite",
        "trained_lstm_int8.tflite",
        ":sample_images",
    ],
    main = "evaluate_test.py",
    tags = [
        "noasan",
        "nomsan",  # Python doesn't like these symbols
        "noubsan",
    ],
    target_compatible_with = INCOMPATIBLE_WITH_WINDOWS,
    deps = [
        ":evaluate",
        ":train",
        "//tensorflow/lite/micro/tools:requantize_flatbuffer",
    ],
)
