#include <Arduino.h>
#include "driver/i2s.h"
#include <math.h>

// =============================
// I2S CONFIGURATION
// =============================

#define I2S_PORT I2S_NUM_0

#define I2S_SCK  18
#define I2S_WS   17
#define I2S_SD   8

#define SAMPLE_RATE 16000
#define FRAME_SIZE  512

// =============================
// VAD CONFIGURATION
// =============================

// Number of consecutive speech frames
#define SPEECH_FRAMES_REQUIRED 3

// Number of consecutive silent frames
#define SILENCE_FRAMES_REQUIRED 8

// Threshold multiplier
#define START_MULTIPLIER 3.0
#define END_MULTIPLIER   2.0

// =============================
// GLOBAL VARIABLES
// =============================

int32_t samples[FRAME_SIZE];

float noiseFloor = 0;

float startThreshold = 0;
float endThreshold = 0;

bool speaking = false;

int speechFrameCount = 0;
int silenceFrameCount = 0;

// =============================
// I2S SETUP
// =============================

void setupI2S()
{
  i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(
      I2S_MODE_MASTER |
      I2S_MODE_RX
    ),

    .sample_rate = SAMPLE_RATE,

    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,

    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,

    .communication_format = I2S_COMM_FORMAT_I2S,

    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,

    .dma_buf_count = 8,

    .dma_buf_len = FRAME_SIZE,

    .use_apll = false,

    .tx_desc_auto_clear = false,

    .fixed_mclk = 0
  };

  i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_SCK,
    .ws_io_num = I2S_WS,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num = I2S_SD
  };

  i2s_driver_install(
    I2S_PORT,
    &i2s_config,
    0,
    NULL
  );

  i2s_set_pin(
    I2S_PORT,
    &pin_config
  );

  i2s_zero_dma_buffer(I2S_PORT);
}

// =============================
// READ AUDIO FRAME
// =============================

bool readAudioFrame()
{
  size_t bytesRead = 0;

  esp_err_t result = i2s_read(
    I2S_PORT,
    samples,
    sizeof(samples),
    &bytesRead,
    portMAX_DELAY
  );

  if (result != ESP_OK)
  {
    return false;
  }

  if (bytesRead != sizeof(samples))
  {
    return false;
  }

  return true;
}

// =============================
// CALCULATE RMS
// =============================

float calculateRMS()
{
  double mean = 0;

  // -------------------------
  // Calculate DC offset
  // -------------------------

  for (int i = 0; i < FRAME_SIZE; i++)
  {
    mean += samples[i];
  }

  mean /= FRAME_SIZE;

  // -------------------------
  // Calculate RMS
  // -------------------------

  double sumSquares = 0;

  for (int i = 0; i < FRAME_SIZE; i++)
  {
    double value = samples[i] - mean;

    // Scale down 32-bit I2S values
    value = value / 256.0;

    sumSquares += value * value;
  }

  double rms = sqrt(
    sumSquares / FRAME_SIZE
  );

  return (float)rms;
}

// =============================
// CALIBRATE NOISE
// =============================

void calibrateNoise()
{
  Serial.println();
  Serial.println("==============================");
  Serial.println(" VAD CALIBRATION");
  Serial.println("==============================");

  Serial.println(
    "Keep the room quiet for 3 seconds..."
  );

  delay(1000);

  float total = 0;

  const int calibrationFrames = 90;

  for (int i = 0; i < calibrationFrames; i++)
  {
    if (readAudioFrame())
    {
      float rms = calculateRMS();

      total += rms;

      Serial.print(".");
    }
  }

  noiseFloor = total / calibrationFrames;

  // Prevent extremely low threshold
  if (noiseFloor < 10)
  {
    noiseFloor = 10;
  }

  startThreshold =
    noiseFloor * START_MULTIPLIER;

  endThreshold =
    noiseFloor * END_MULTIPLIER;

  Serial.println();

  Serial.print("Noise Floor: ");
  Serial.println(noiseFloor);

  Serial.print("Start Threshold: ");
  Serial.println(startThreshold);

  Serial.print("End Threshold: ");
  Serial.println(endThreshold);

  Serial.println("==============================");
  Serial.println(" VAD READY");
  Serial.println("==============================");
  Serial.println();
}

// =============================
// SETUP
// =============================

void setup()
{
  Serial.begin(115200);

  delay(1000);

  pinMode(2, OUTPUT);

  digitalWrite(2, LOW);

  Serial.println();
  Serial.println("================================");
  Serial.println(" ESP32 VOICE ACTIVITY DETECTOR");
  Serial.println("================================");

  setupI2S();

  delay(500);

  calibrateNoise();
}

// =============================
// LOOP
// =============================

void loop()
{
  if (!readAudioFrame())
  {
    return;
  }

  float rms = calculateRMS();

  // ==================================
  // IDLE STATE
  // ==================================

  if (!speaking)
  {
    if (rms > startThreshold)
    {
      speechFrameCount++;

      silenceFrameCount = 0;

      if (speechFrameCount >= SPEECH_FRAMES_REQUIRED)
      {
        speaking = true;

        speechFrameCount = 0;

        digitalWrite(2, HIGH);

        Serial.println();
        Serial.println(">>> VOICE START <<<");
      }
    }
    else
    {
      speechFrameCount = 0;
    }
  }

  // ==================================
  // SPEAKING STATE
  // ==================================

  else
  {
    if (rms < endThreshold)
    {
      silenceFrameCount++;

      if (silenceFrameCount >= SILENCE_FRAMES_REQUIRED)
      {
        speaking = false;

        silenceFrameCount = 0;

        digitalWrite(2, LOW);

        Serial.println(">>> VOICE END <<<");
        Serial.println();
      }
    }
    else
    {
      silenceFrameCount = 0;
    }
  }

  // ==================================
  // SERIAL OUTPUT
  // ==================================

  Serial.print("RMS: ");
  Serial.print(rms);

  Serial.print(" | Noise: ");
  Serial.print(noiseFloor);

  Serial.print(" | ");

  if (speaking)
  {
    Serial.println("SPEECH");
  }
  else
  {
    Serial.println("SILENCE");
  }

  delay(1);
}