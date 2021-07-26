#include "Classifier.h"
#include "classifier_model_data.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#include <Arduino.h>

#define DEBUG

const int kArenaSize = 40 * 1024;

Classifier::Classifier()
{
    error_reporter = new tflite::MicroErrorReporter();

    model = tflite::GetModel(classifier_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION)
    {
        TF_LITE_REPORT_ERROR(error_reporter, "Model provided is schema version %d not equal to supported version %d.",
                             model->version(), TFLITE_SCHEMA_VERSION);
        return;
    }
    // This pulls in the operators implementations we need
    resolver = new tflite::MicroMutableOpResolver<10>();
    resolver->AddFullyConnected();
    resolver->AddMul();
    resolver->AddAdd();
    resolver->AddLogistic();
    resolver->AddReshape();
    resolver->AddQuantize();
    resolver->AddDequantize();
    resolver->AddConv2D();
    resolver->AddMaxPool2D();
    resolver->AddSoftmax();

    tensor_arena = (uint8_t *)malloc(kArenaSize);
    if (!tensor_arena)
    {
        TF_LITE_REPORT_ERROR(error_reporter, "Could not allocate arena");
        return;
    }

    // Build an interpreter to run the model with.
    interpreter = new tflite::MicroInterpreter(
        model, *resolver, tensor_arena, kArenaSize, error_reporter);

    // Allocate memory from the tensor_arena for the model's tensors.
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk)
    {
        TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
        return;
    }

    size_t used_bytes = interpreter->arena_used_bytes();
    TF_LITE_REPORT_ERROR(error_reporter, "Used bytes %d\n", used_bytes);

    // Obtain pointers to the model's input and output tensors.
    input = interpreter->input(0);
    output = interpreter->output(0);
}

void Classifier::checkInputParameters()
{
    if ((input->dims->size != 4) ||
        (input->dims->data[0] != 1) ||
        (input->dims->data[1] != kNumRows) ||
        (input->dims->data[2] != kNumCols) ||
        (input->dims->data[3]) != 1 ||
        (input->type != kTfLiteFloat32))
    {
        TF_LITE_REPORT_ERROR(error_reporter, "Bad input tensor parameters in model");
        return;
    }
}

void Classifier::setModelInput(int d[][kNumCols])
{
    float *input_data_ptr = interpreter->typed_input_tensor<float>(0);

    for (int i = 0; i < kNumRows; i++)
    {
        for (int j = 0; j < kNumCols; j++)
        {
            *(input_data_ptr) = (float)d[i][j];
            input_data_ptr++;
        }
    }
}

distribution Classifier::inference()
{
    if (interpreter->Invoke() != kTfLiteOk)
    {
        TF_LITE_REPORT_ERROR(error_reporter, "Inference failed");
        Serial.println("FAILED");
    }
    float no_signal_probability = output->data.f[0];
    float signal_probability = output->data.f[1];
#ifdef DEBUG
    Serial.print("Signal prob: ");
    Serial.print(signal_probability);
    Serial.print("  No signal prob: ");
    Serial.println(no_signal_probability);
#endif
    return distribution{
      signal_probability,
      no_signal_probability
      };

}
