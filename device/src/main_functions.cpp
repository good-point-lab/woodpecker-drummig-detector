#include <Arduino.h>
#include <CircularBuffer.h>
#include <driver/adc.h>
#include <MD_MSGEQ7.h>
#include "Classifier.h"
#include "SSD1306Wire.h"

#define OPERATION_MODE
// #define DEBUG
// #define PROFILING

#define RESTART_FREQ_MS 1000 * 60 * 20
#define CALIBRATION_PERIOD_MS 1000 * 30

// I2C
#define I2C_SDA_PIN 21
#define I2C_SCL_PIN 22

// 7-band MSGEQ7 spectra analyser
#define NUM_BANDS MAX_BAND
#define DATA_PIN 34 // The same as ESP32 ADC chennel 6
#define RESET_PIN 27
#define STROBE_PIN 26
#define NOISE_LEVEL 100

// ESP32 ADC configuration
#define ADC_MIN_BIT 0
#define ADC_MAX_BIT 4095
#define ADC_CHANNEL ADC1_CHANNEL_6

// Availible attenuations ADC_ATTEN_0db, ADC_ATTEN_2_5db, or ADC_ATTEN_6db
#define ADC_ATTENUATION ADC_ATTEN_0db
#define SNAPSHOT_PERIOD_MS 1000

// Sampling parameters
// Maximum time resolution with the MSGEQ7 spectra analyser is about 0.5 msec
#define SAMPLE_PERIOD_MS 5
#define TIME_POINTS 600
#define BUFFER_SZ TIME_POINTS
#define MAX_SIGNAL_AMPLITUDE ADC_MAX_BIT

// Inference parameters
#define SIGNAL_PROB_THRESHOLD 0.99
#define INFERENCE_BUFFER_SIZE 6
#define SIGNAL_CRITERIA 0.5

// MSGEQ7 frequency bands
enum bands
{
  b63Hz,
  b160Hz,
  b400Hz,
  b1000Hz,
  b2500Hz,
  b6250Hz,
  b16000Hz
};

namespace data
{
  typedef struct
  {
    uint16_t band1;
    uint16_t band2;
    uint16_t band3;
    uint16_t band4;
    uint16_t band5;
    uint16_t band6;
    uint16_t band7;
  } record;

  typedef struct
  {
    float mean;
    float std;
    float max_val;
    float min_val;
  } statistics;

}

MD_MSGEQ7 MSGEQ7(RESET_PIN, STROBE_PIN, DATA_PIN);
// The circular buffers are from this library - https://github.com/rlogiacco/CircularBuffer
// Sampling buffer
CircularBuffer<data::record, BUFFER_SZ> buffer;
Classifier *classifier;

// To avoid fluctuations coused by classification outliers
// recording running values
CircularBuffer<bool, INFERENCE_BUFFER_SIZE> inference_buf;

// Display
SSD1306Wire display(0x3c, I2C_SDA_PIN, I2C_SCL_PIN);

// Time stamps
unsigned long ts = 0;
unsigned long t_us = 0;
volatile unsigned long duration = 0;

// Raw time-frequency data for spectrogram
int spectrogram[TIME_POINTS][NUM_BANDS];

void configureADC()
{
  // ADC1 channels GPIO 32-39
  // 12 bits wide (range 0-4095)
  // Channel 6 is GPO34
  adc1_config_width(ADC_WIDTH_BIT_12);
  adc1_config_channel_atten(ADC_CHANNEL, ADC_ATTENUATION);
}

void recordSample()
{
  MSGEQ7.read(true);
  buffer.push(data::record{
      MSGEQ7.get(b63Hz),
      MSGEQ7.get(b160Hz),
      MSGEQ7.get(b400Hz),
      MSGEQ7.get(b1000Hz),
      MSGEQ7.get(b2500Hz),
      MSGEQ7.get(b6250Hz),
      MSGEQ7.get(b16000Hz),
  });
}

void resetSpectrogram()
{
  for (int i = 0; i < TIME_POINTS; i++)
  {
    for (int j = 0; j < NUM_BANDS; j++)
    {
      spectrogram[i][j] = 0;
    }
  }
}

String formatSpectrogram()
{
  String rv = "\n";
  for (int i = 0; i < BUFFER_SZ; i++)
  {
    for (int j = 0; j < NUM_BANDS; j++)
    {
      rv = rv + String(spectrogram[i][j]) + "  ";
    }
    rv = rv + " : \n";
  }
  return rv;
}

String serializeSeries(int band)
{
  String rv = "";
  for (int i = 0; i < TIME_POINTS; i++)
  {
    String val = String(spectrogram[i][band]);
    bool last_point = i == TIME_POINTS - 1;
    if (last_point)
    {
      rv += val;
    }
    else
    {
      rv += val + ", ";
    }
  }
  return "[" + rv + "]";
}

String serializeSpectrogram(int *dd, int totalSz, int freqSz)
{
  String rv = "";
  for (int i = 0; i < totalSz; i++)
  {
    bool new_slice = i % freqSz == 0;
    bool first_point = i == 0;
    String val = String(dd[i]);

    if (new_slice && !first_point)
    {
      rv += " ], [" + val;
    }
    else if (first_point)
    {
      rv += val;
    }
    else
    {
      rv += ", " + val;
    }
  }
  return "[[" + rv + "]]";
}

void createSpectrogram()
{
  using index_t = decltype(buffer)::index_t;
  for (index_t i = 0; i < buffer.size(); i++)
  {
    spectrogram[i][b63Hz] = buffer[i].band1 < NOISE_LEVEL ? 0 : buffer[i].band1;
    spectrogram[i][b160Hz] = buffer[i].band2 < NOISE_LEVEL ? 0 : buffer[i].band2;
    spectrogram[i][b400Hz] = buffer[i].band3 < NOISE_LEVEL ? 0 : buffer[i].band3;
    spectrogram[i][b1000Hz] = buffer[i].band4 < NOISE_LEVEL ? 0 : buffer[i].band4;
    spectrogram[i][b2500Hz] = buffer[i].band5 < NOISE_LEVEL ? 0 : buffer[i].band5;
    spectrogram[i][b6250Hz] = buffer[i].band6 < NOISE_LEVEL ? 0 : buffer[i].band6;
    spectrogram[i][b16000Hz] = buffer[i].band7 < NOISE_LEVEL ? 0 : buffer[i].band7;
  }
}

data::statistics computeSpectrogamStatistics()
{
  float mean = 0.0;
  float std = 0.0;
  float maxv = -99999.0;
  float minv = 99999.0;

  for (int i = 0; i < TIME_POINTS; i++)
  {
    for (int j = 0; j < NUM_BANDS; j++)
    {
      mean += (float)spectrogram[i][j];
      if (spectrogram[i][j] > maxv)
      {
        maxv = spectrogram[i][j];
      }
      if (spectrogram[i][j] < minv)
      {
        minv = spectrogram[i][j];
      }
    }
  }
  mean = mean / (NUM_BANDS * BUFFER_SZ);

  for (int i = 0; i < TIME_POINTS; i++)
  {
    for (int j = 0; j < NUM_BANDS; j++)
    {
      std += sq((float)spectrogram[i][j] - mean);
    }
  }
  std = sqrt(std / (NUM_BANDS * BUFFER_SZ));

  return data::statistics{
      mean,
      std,
      maxv,
      minv};
}

// z-score normalization
void NormalizeSpectrogram(data::statistics stats)
{
  int factor = 1.0;
  for (int i = 0; i < TIME_POINTS; i++)
  {
    for (int j = 0; j < NUM_BANDS; j++)
    {
      float zc = factor * ((spectrogram[i][j] - stats.mean) / stats.std);
      spectrogram[i][j] = (int)zc;
    }
  }
}

// Serialize spectrogram and metadata to JSON formatted serial I/O
String serializeWindowSnapshot(int band)
{
  String json_string = "{";
  json_string += "\"spectrogram\":" + serializeSpectrogram(spectrogram[0], NUM_BANDS * TIME_POINTS, NUM_BANDS) + ",";
  json_string += "\"series\":" + serializeSeries(band) + ",";
  json_string += "\"meta\":{\"time_min\":" + String(0) + ",";
  json_string += "\"time_max\":" + String((SAMPLE_PERIOD_MS * TIME_POINTS) / 1000) + ",";
  json_string += "\"time_points\":" + String(TIME_POINTS) + ",";
  json_string += "\"bands\":" + String(NUM_BANDS) + ",";
  json_string += "\"freq_grid_sz\":" + String(NUM_BANDS) + ",";
  json_string += "\"time_grid_sz\":" + String(TIME_POINTS) + ",";
  json_string += "\"amplitude_scale\":" + String(MAX_SIGNAL_AMPLITUDE) + ",";
  json_string += "\"band_feature\":" + String(band) + "}}";
  return json_string;
}


// Inference Heuristics
bool isSignalDetected(distribution instantDist)
{
  bool instantPrediction = false;
  if (instantDist.probability_signal > SIGNAL_PROB_THRESHOLD)
  {
      instantPrediction = true;
  }
  inference_buf.push(instantPrediction);
  using index_t = decltype(inference_buf)::index_t;
  int cnt = 0;
  for (index_t i = 0; i < inference_buf.size(); i++)
  {
    if (inference_buf[i])
    {
      cnt++;
    }
  }

  if ((float)cnt/INFERENCE_BUFFER_SIZE > SIGNAL_CRITERIA)
  {
    return true;
  }
  else
  {
    return false;
  }
}

/*
Operational Messages
*/
void onSignalDetection()
{
  display.clear();
  display.drawString(0, 0, "Woodpecker");
  display.drawString(0, 20, "sound");
  display.drawString(0, 40, "noticed");
  display.display();
  delay(200);
  display.resetDisplay();
}

void setup()
{
  Serial.begin(115200);
  while (!Serial)
  {
    ;
  }
  configureADC();

  // Join I2C bus and set fast mode, 400kHz
  Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN);
  Wire.setClock(400000);

  // Prepair display
  display.init();
  display.setFont(ArialMT_Plain_16);
  display.flipScreenVertically();

  resetSpectrogram();
  MSGEQ7.begin();
#ifdef OPERATION_MODE
  classifier = new Classifier();
  delay(1000);
  classifier->checkInputParameters();
#endif
  ts = millis();
  t_us = micros();
}

void loop()
{
  if (millis() - ts >= RESTART_FREQ_MS)
  {
    ESP.restart();
  }
  if (micros() - t_us >= 1000 * SAMPLE_PERIOD_MS)
  {
    recordSample();
    t_us = micros();
  }
  if (millis() - ts >= SNAPSHOT_PERIOD_MS)
  {
    #ifdef PROFILING
      unsigned long t_start_inference = millis();
    #endif
    createSpectrogram();
    data::statistics stats = computeSpectrogamStatistics();
    NormalizeSpectrogram(stats);
#ifdef DEBUG
    Serial.print(stats.mean);
    Serial.print("   ");
    Serial.print(stats.std);
    Serial.print("   ");
    Serial.print(stats.max_val);
    Serial.print("   ");
    Serial.println(stats.min_val);
#endif

// Run inference or 
#ifdef OPERATION_MODE
    classifier->setModelInput(spectrogram);
    distribution dist = classifier->inference();
    #ifdef PROFILING
      unsigned long inference_duration = millis() - t_start_inference;
      Serial.println(inference_duration);
    #endif
    if (isSignalDetected(dist))
    {
      onSignalDetection();
    }
#else
// Send data to capture/annotation tool via serial connection
    int aMiddleBand = 4;
    String json = serializeWindowSnapshot(aMiddleBand);
    Serial.println(json);
    Serial.flush();
#endif
    ts = millis();
  }
}
