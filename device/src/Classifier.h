#ifndef __Classifier__
#define __Classifier__

#include <stdint.h>

// All of these values are derived from the values used during model training,
constexpr int kNumCols = 7;     // frequency bands
constexpr int kNumRows = 600;   // time points
constexpr int kNumChannels = 1; // spectrogram is a one channel entity (as black and whit image)

constexpr int kMaxSpectrogramSize = kNumCols * kNumRows * kNumChannels;
constexpr int kCategoryCount = 2; // Two classes - signal of interest or everything else

namespace tflite
{
    template <unsigned int tOpCount>
    class MicroMutableOpResolver;
    class ErrorReporter;
    class Model;
    class MicroInterpreter;
}

struct TfLiteTensor;


typedef struct
  {
    float probability_signal;
    float probability_no_signal;
  } distribution;

class Classifier
{
private:
    tflite::MicroMutableOpResolver<10> *resolver;
    tflite::ErrorReporter *error_reporter;
    const tflite::Model *model;
    tflite::MicroInterpreter *interpreter;
    TfLiteTensor *input;
    TfLiteTensor *output;
    uint8_t *tensor_arena;

public:
    void checkInputParameters();
    void setModelInput(int data[][kNumCols]);
    distribution inference();
    Classifier();
};

#endif
