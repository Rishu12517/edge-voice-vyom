#include <Arduino.h>
#include <math.h>

#include "driver/i2s.h"

#include <Chirale_TensorFlowLite.h>
#include "vyom_model.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"


// ============================================================
// I2S / MICROPHONE
// ============================================================

#define I2S_PORT I2S_NUM_0

#define I2S_SCK  18
#define I2S_WS   17
#define I2S_SD   8

#define SAMPLE_RATE 16000

#define AUDIO_SAMPLES 16000


// ============================================================
// MFCC SETTINGS
// ============================================================

#define FFT_SIZE     512
#define HOP_LENGTH   160

#define N_MFCC       13
#define N_MELS       128

#define FRAME_COUNT   101


// ============================================================
// MODEL SETTINGS
// ============================================================

constexpr int TENSOR_ARENA_SIZE = 80 * 1024;

alignas(16)
uint8_t tensor_arena[TENSOR_ARENA_SIZE];

const tflite::Model* model = nullptr;

tflite::MicroInterpreter* interpreter = nullptr;

TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;


// ============================================================
// AUDIO BUFFER
// ============================================================

float audio[AUDIO_SAMPLES];

int32_t i2sBuffer[512];


// ============================================================
// FFT ARRAYS
// ============================================================

float fftReal[FFT_SIZE];

float fftImag[FFT_SIZE];

float powerSpectrum[FFT_SIZE / 2 + 1];


// ============================================================
// MEL / MFCC ARRAYS
// ============================================================

float melEnergy[N_MELS];

// IMPORTANT:
// MFCC is stored in coefficient-major order:
//
// mfcc[0][0..100]
// mfcc[1][0..100]
// ...
// mfcc[12][0..100]
//
// This matches Python's:
//
// librosa.feature.mfcc()
// shape = (13, 101)
//
// and NumPy C-order flattening.

float mfcc[N_MFCC * FRAME_COUNT];

// IMPORTANT:
// Keep melDB global.
// This array is too large for the task stack.

float melDB[FRAME_COUNT][N_MELS];

float mfccMin = 0.0f;

float mfccMax = 0.0f;


// ============================================================
// FFT
// ============================================================

void fft(float* real, float* imag, int n)
{
  // ----------------------------------------------------------
  // Bit reversal
  // ----------------------------------------------------------

  int j = 0;

  for (int i = 1; i < n; i++)
  {
    int bit = n >> 1;

    while (j & bit)
    {
      j ^= bit;
      bit >>= 1;
    }

    j ^= bit;

    if (i < j)
    {
      float temp = real[i];
      real[i] = real[j];
      real[j] = temp;

      temp = imag[i];
      imag[i] = imag[j];
      imag[j] = temp;
    }
  }


  // ----------------------------------------------------------
  // FFT stages
  // ----------------------------------------------------------

  for (int len = 2; len <= n; len <<= 1)
  {
    float angle = -2.0f * PI / len;

    float wlenReal = cosf(angle);
    float wlenImag = sinf(angle);

    for (int i = 0; i < n; i += len)
    {
      float wReal = 1.0f;
      float wImag = 0.0f;

      int half = len >> 1;

      for (int j = 0; j < half; j++)
      {
        int u = i + j;
        int v = i + j + half;

        float vReal =
          real[v] * wReal -
          imag[v] * wImag;

        float vImag =
          real[v] * wImag +
          imag[v] * wReal;

        float uReal = real[u];
        float uImag = imag[u];

        real[u] = uReal + vReal;
        imag[u] = uImag + vImag;

        real[v] = uReal - vReal;
        imag[v] = uImag - vImag;

        float nextWReal =
          wReal * wlenReal -
          wImag * wlenImag;

        float nextWImag =
          wReal * wlenImag +
          wImag * wlenReal;

        wReal = nextWReal;
        wImag = nextWImag;
      }
    }
  }
}


// ============================================================
// MEL SCALE
// ============================================================

float hzToMel(float hz)
{
  return 2595.0f *
         log10f(
           1.0f + hz / 700.0f
         );
}


float melToHz(float mel)
{
  return 700.0f *
         (
           powf(
             10.0f,
             mel / 2595.0f
           ) - 1.0f
         );
}


// ============================================================
// CAPTURE AUDIO
// ============================================================

bool captureAudio()
{
  Serial.println();
  Serial.println("Listening...");

  int collected = 0;

  while (collected < AUDIO_SAMPLES)
  {
    size_t bytesRead = 0;

    esp_err_t result =
      i2s_read(
        I2S_PORT,
        i2sBuffer,
        sizeof(i2sBuffer),
        &bytesRead,
        portMAX_DELAY
      );

    if (result != ESP_OK)
    {
      Serial.println("I2S read error");

      return false;
    }

    int samplesRead =
      bytesRead / sizeof(int32_t);

    for (
      int i = 0;
      i < samplesRead &&
      collected < AUDIO_SAMPLES;
      i++
    )
    {
      /*
         INMP441 gives 24-bit audio inside
         a 32-bit I2S word.

         Convert approximately to
         -1.0 ... +1.0.
      */

      audio[collected] =
        (float)i2sBuffer[i] /
        2147483648.0f;

      collected++;
    }
  }

  Serial.println("1 second captured.");


  // ==========================================================
  // AUDIO DEBUG
  // ==========================================================

  float audioMin = audio[0];

  float audioMax = audio[0];

  float audioSum = 0.0f;


  for (int i = 0;
       i < AUDIO_SAMPLES;
       i++)
  {
    if (audio[i] < audioMin)
      audioMin = audio[i];

    if (audio[i] > audioMax)
      audioMax = audio[i];

    audioSum +=
      audio[i] * audio[i];
  }


  float audioRMS =
    sqrtf(
      audioSum /
      AUDIO_SAMPLES
    );


  Serial.print("Audio min = ");
  Serial.println(audioMin, 5);

  Serial.print("Audio max = ");
  Serial.println(audioMax, 5);

  Serial.print("Audio RMS = ");
  Serial.println(audioRMS, 5);


  return true;
}


// ============================================================
// CALCULATE MFCC
// ============================================================

void calculateMFCC()
{
  Serial.println("Calculating MFCC...");

  Serial.println("MFCC processing started...");


  // ----------------------------------------------------------
  // Mel filter bank boundaries
  // ----------------------------------------------------------

  float melMin =
    hzToMel(0.0f);

  float melMax =
    hzToMel(
      SAMPLE_RATE / 2.0f
    );


  int bin[N_MELS + 2];


  for (
    int i = 0;
    i < N_MELS + 2;
    i++
  )
  {
    float mel =
      melMin +
      (
        melMax - melMin
      ) *
      (
        (float)i /
        (N_MELS + 1)
      );


    float hz =
      melToHz(mel);


    int b =
      floorf(
        (
          (FFT_SIZE + 1) *
          hz
        ) /
        SAMPLE_RATE
      );


    if (b < 0)
      b = 0;


    if (b > FFT_SIZE / 2)
      b = FFT_SIZE / 2;


    bin[i] = b;
  }


  // ----------------------------------------------------------
  // First pass:
  // calculate mel energies
  // ----------------------------------------------------------

  float globalMax =
    -1.0e30f;


  for (
    int frame = 0;
    frame < FRAME_COUNT;
    frame++
  )
  {
    if (frame % 10 == 0)
    {
      Serial.print("MFCC frame: ");

      Serial.println(frame);
    }


    /*
       librosa uses center=True.

       Therefore frame 0 is centered at
       sample 0 with 256 samples of
       zero padding before it.
    */

    int center =
      frame * HOP_LENGTH;


    int start =
      center -
      FFT_SIZE / 2;


    // --------------------------------------------------------
    // Copy frame + Hann window
    // --------------------------------------------------------

    for (
      int n = 0;
      n < FFT_SIZE;
      n++
    )
    {
      int audioIndex =
        start + n;


      float sample = 0.0f;


      if (
        audioIndex >= 0 &&
        audioIndex < AUDIO_SAMPLES
      )
      {
        sample =
          audio[audioIndex];
      }


      /*
         Periodic Hann window.

         This corresponds to
         scipy/librosa fftbins=True.
      */

      float window =
        0.5f -
        0.5f *
        cosf(
          2.0f *
          PI *
          n /
          FFT_SIZE
        );


      fftReal[n] =
        sample * window;

      fftImag[n] =
        0.0f;
    }


    // --------------------------------------------------------
    // FFT
    // --------------------------------------------------------

    fft(
      fftReal,
      fftImag,
      FFT_SIZE
    );


    // --------------------------------------------------------
    // Power spectrum
    // --------------------------------------------------------

    for (
      int k = 0;
      k <= FFT_SIZE / 2;
      k++
    )
    {
      float re =
        fftReal[k];

      float im =
        fftImag[k];


      powerSpectrum[k] =
        re * re +
        im * im;
    }


    // --------------------------------------------------------
    // Mel filter bank
    // --------------------------------------------------------

    for (
      int m = 0;
      m < N_MELS;
      m++
    )
    {
      int left =
        bin[m];

      int centerBin =
        bin[m + 1];

      int right =
        bin[m + 2];


      float energy =
        0.0f;


      // ------------------------------------------------------
      // Rising slope
      // ------------------------------------------------------

      if (centerBin > left)
      {
        for (
          int k = left;
          k < centerBin;
          k++
        )
        {
          float weight =
            (float)(k - left) /
            (float)(centerBin - left);


          energy +=
            powerSpectrum[k] *
            weight;
        }
      }


      // ------------------------------------------------------
      // Falling slope
      // ------------------------------------------------------

      if (right > centerBin)
      {
        for (
          int k = centerBin;
          k < right;
          k++
        )
        {
          float weight =
            (float)(right - k) /
            (float)(right - centerBin);


          energy +=
            powerSpectrum[k] *
            weight;
        }
      }


      // ------------------------------------------------------
      // Slaney-style area normalization
      // ------------------------------------------------------

      float lowFreq =
        melToHz(
          melMin +
          (
            melMax - melMin
          ) *
          (
            (float)m /
            (N_MELS + 1)
          )
        );


      float highFreq =
        melToHz(
          melMin +
          (
            melMax - melMin
          ) *
          (
            (float)(m + 2) /
            (N_MELS + 1)
          )
        );


      float enorm =
        2.0f /
        (
          highFreq -
          lowFreq
        );


      energy *= enorm;


      if (energy < 1.0e-10f)
        energy = 1.0e-10f;


      melEnergy[m] =
        energy;
    }


    // --------------------------------------------------------
    // Power -> dB
    // --------------------------------------------------------

    for (
      int m = 0;
      m < N_MELS;
      m++
    )
    {
      float db =
        10.0f *
        log10f(
          melEnergy[m]
        );


      melDB[frame][m] =
        db;


      if (db > globalMax)
        globalMax = db;
    }
  }


  // ----------------------------------------------------------
  // librosa power_to_db top_db = 80
  // ----------------------------------------------------------

  float minimumDB =
    globalMax - 80.0f;


  for (
    int frame = 0;
    frame < FRAME_COUNT;
    frame++
  )
  {
    for (
      int m = 0;
      m < N_MELS;
      m++
    )
    {
      if (
        melDB[frame][m] <
        minimumDB
      )
      {
        melDB[frame][m] =
          minimumDB;
      }
    }
  }


  // ----------------------------------------------------------
  // DCT-II
  // ----------------------------------------------------------

  mfccMin =
    1.0e30f;

  mfccMax =
    -1.0e30f;


  for (
    int frame = 0;
    frame < FRAME_COUNT;
    frame++
  )
  {
    for (
      int k = 0;
      k < N_MFCC;
      k++
    )
    {
      float sum =
        0.0f;


      for (
        int n = 0;
        n < N_MELS;
        n++
      )
      {
        float angle =
          PI *
          k *
          (
            2.0f * n +
            1.0f
          ) /
          (
            2.0f * N_MELS
          );


        sum +=
          melDB[frame][n] *
          cosf(angle);
      }


      // ------------------------------------------------------
      // Orthonormal DCT-II
      // ------------------------------------------------------

      float scale =
        sqrtf(
          2.0f /
          N_MELS
        );


      if (k == 0)
      {
        scale =
          sqrtf(
            1.0f /
            N_MELS
          );
      }


      float value =
        sum * scale;


      // ======================================================
      // IMPORTANT FIX
      //
      // Python shape:
      //
      //     (13, 101)
      //
      // Therefore store:
      //
      //     coefficient-major
      //
      // mfcc[coefficient][frame]
      //
      // Flattened:
      //
      //     k * FRAME_COUNT + frame
      // ======================================================

      int index =
        k * FRAME_COUNT +
        frame;


      mfcc[index] =
        value;


      if (value < mfccMin)
        mfccMin = value;


      if (value > mfccMax)
        mfccMax = value;
    }
  }


  // ==========================================================
  // MFCC DEBUG
  // ==========================================================

  Serial.print("MFCC min = ");

  Serial.println(
    mfccMin,
    2
  );


  Serial.print("MFCC max = ");

  Serial.println(
    mfccMax,
    2
  );


  Serial.println();

  Serial.println(
    "========== MFCC COMPARISON =========="
  );


  // ----------------------------------------------------------
  // FRAME 0
  // ----------------------------------------------------------

  Serial.println(
    "FRAME 0:"
  );


  for (int i = 0; i < 13; i++)
  {
    Serial.print("MFCC[");

    Serial.print(i);

    Serial.print("] = ");

    Serial.println(
      mfcc[
        i * FRAME_COUNT + 0
      ],
      6
    );
  }


  // ----------------------------------------------------------
  // FRAME 50
  // ----------------------------------------------------------

  Serial.println();

  Serial.println(
    "FRAME 50:"
  );


  for (int i = 0; i < 13; i++)
  {
    Serial.print("MFCC[");

    Serial.print(i);

    Serial.print("] = ");

    Serial.println(
      mfcc[
        i * FRAME_COUNT + 50
      ],
      6
    );
  }


  // ----------------------------------------------------------
  // FRAME 100
  // ----------------------------------------------------------

  Serial.println();

  Serial.println(
    "FRAME 100:"
  );


  for (int i = 0; i < 13; i++)
  {
    Serial.print("MFCC[");

    Serial.print(i);

    Serial.print("] = ");

    Serial.println(
      mfcc[
        i * FRAME_COUNT + 100
      ],
      6
    );
  }


  Serial.println(
    "===================================="
  );


  // ----------------------------------------------------------
  // First 20 flattened MFCC values
  // ----------------------------------------------------------

  Serial.println();

  Serial.println(
    "FIRST 20 FLATTENED MFCC VALUES:"
  );


  for (int i = 0; i < 20; i++)
  {
    Serial.print(
      mfcc[i],
      4
    );

    Serial.print(" ");
  }


  Serial.println();

  Serial.println(
    "MFCC calculation complete."
  );
}


// ============================================================
// RUN VYOM MODEL
// ============================================================

void runVYOM()
{
  Serial.println(
    "Quantizing MFCC..."
  );


  // ----------------------------------------------------------
  // Quantize according to actual INT8 model parameters
  // ----------------------------------------------------------

  for (
    int i = 0;
    i < N_MFCC * FRAME_COUNT;
    i++
  )
  {
    float value =
      mfcc[i];


    int q =
      (int)roundf(
        value /
        input->params.scale
        +
        input->params.zero_point
      );


    if (q < -128)
      q = -128;


    if (q > 127)
      q = 127;


    input->data.int8[i] =
      (int8_t)q;
  }


  Serial.println(
    "Running VYOM CNN..."
  );


  unsigned long inferenceStart =
    millis();


  TfLiteStatus status =
    interpreter->Invoke();


  unsigned long inferenceEnd =
    millis();


  if (
    status != kTfLiteOk
  )
  {
    Serial.println(
      "ERROR: Inference failed!"
    );

    return;
  }


  // ----------------------------------------------------------
  // Read INT8 output
  // ----------------------------------------------------------

  int8_t rawOutput =
    output->data.int8[0];


  float probability =
    (
      (float)rawOutput -
      (float)output->params.zero_point
    )
    *
    output->params.scale;


  // ----------------------------------------------------------
  // Clamp probability
  // ----------------------------------------------------------

  if (probability < 0.0f)
    probability = 0.0f;


  if (probability > 1.0f)
    probability = 1.0f;


  // ----------------------------------------------------------
  // Print result
  // ----------------------------------------------------------

  Serial.print(
    "Raw output = "
  );

  Serial.println(
    (int)rawOutput
  );


  Serial.print(
    "VYOM probability = "
  );

  Serial.println(
    probability,
    4
  );


  Serial.print(
    "MFCC + inference time = "
  );


  Serial.print(
    inferenceEnd -
    inferenceStart
  );


  Serial.println(
    " ms"
  );


  // ----------------------------------------------------------
  // Detection
  // ----------------------------------------------------------

  if (
    probability >= 0.50f
  )
  {
    Serial.println();

    Serial.println(
      ">>> VYOM DETECTED <<<"
    );
  }
  else
  {
    Serial.println();

    Serial.println(
      ">>> VYOM DETECTED<<<"
    );
  }
}


// ============================================================
// SETUP I2S
// ============================================================

void setupI2S()
{
  i2s_config_t i2s_config =
  {
    .mode =
      (i2s_mode_t)(
        I2S_MODE_MASTER |
        I2S_MODE_RX
      ),

    .sample_rate =
      SAMPLE_RATE,

    .bits_per_sample =
      I2S_BITS_PER_SAMPLE_32BIT,

    .channel_format =
      I2S_CHANNEL_FMT_ONLY_LEFT,

    .communication_format =
      I2S_COMM_FORMAT_I2S,

    .intr_alloc_flags =
      ESP_INTR_FLAG_LEVEL1,

    .dma_buf_count =
      8,

    .dma_buf_len =
      512,

    .use_apll =
      false,

    .tx_desc_auto_clear =
      false,

    .fixed_mclk =
      0
  };


  i2s_pin_config_t pin_config =
  {
    .bck_io_num =
      I2S_SCK,

    .ws_io_num =
      I2S_WS,

    .data_out_num =
      I2S_PIN_NO_CHANGE,

    .data_in_num =
      I2S_SD
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


  i2s_zero_dma_buffer(
    I2S_PORT
  );
}


// ============================================================
// SETUP MODEL
// ============================================================

bool setupModel()
{
  Serial.println(
    "Loading VYOM model..."
  );


  // ----------------------------------------------------------
  // Load model
  // ----------------------------------------------------------

  model =
    tflite::GetModel(
      vyom_tinycnn_int8_tflite
    );


  if (
    model->version() !=
    TFLITE_SCHEMA_VERSION
  )
  {
    Serial.println(
      "ERROR: Model schema mismatch!"
    );

    return false;
  }


  // ----------------------------------------------------------
  // Resolver
  // ----------------------------------------------------------

  static tflite::AllOpsResolver resolver;


  // ----------------------------------------------------------
  // Interpreter
  // ----------------------------------------------------------

  static tflite::MicroInterpreter
    staticInterpreter(
      model,
      resolver,
      tensor_arena,
      TENSOR_ARENA_SIZE
    );


  interpreter =
    &staticInterpreter;


  // ----------------------------------------------------------
  // Allocate tensors
  // ----------------------------------------------------------

  if (
    interpreter->AllocateTensors()
    != kTfLiteOk
  )
  {
    Serial.println(
      "ERROR: AllocateTensors failed!"
    );

    return false;
  }


  // ----------------------------------------------------------
  // Get tensors
  // ----------------------------------------------------------

  input =
    interpreter->input(0);


  output =
    interpreter->output(0);


  if (input == nullptr)
  {
    Serial.println(
      "ERROR: Input tensor is NULL!"
    );

    return false;
  }


  if (output == nullptr)
  {
    Serial.println(
      "ERROR: Output tensor is NULL!"
    );

    return false;
  }


  // ----------------------------------------------------------
  // Print tensor information
  // ----------------------------------------------------------

  Serial.println(
    "VYOM model ready."
  );


  Serial.print(
    "Input scale: "
  );


  Serial.println(
    input->params.scale,
    8
  );


  Serial.print(
    "Input zero point: "
  );


  Serial.println(
    input->params.zero_point
  );


  Serial.print(
    "Output scale: "
  );


  Serial.println(
    output->params.scale,
    8
  );


  Serial.print(
    "Output zero point: "
  );


  Serial.println(
    output->params.zero_point
  );


  return true;
}


// ============================================================
// SETUP
// ============================================================

void setup()
{
  Serial.begin(9600);

  delay(1500);


  Serial.println();

  Serial.println(
    "=============================="
  );


  Serial.println(
    " VYOM LIVE MICROPHONE TEST"
  );


  Serial.println(
    "=============================="
  );


  // ----------------------------------------------------------
  // I2S
  // ----------------------------------------------------------

  Serial.println(
    "Starting microphone..."
  );


  setupI2S();


  delay(500);


  // ----------------------------------------------------------
  // Model
  // ----------------------------------------------------------

  if (!setupModel())
  {
    Serial.println();

    Serial.println(
      "MODEL SETUP FAILED"
    );


    while (true)
    {
      delay(1000);
    }
  }


  Serial.println();

  Serial.println(
    "=============================="
  );


  Serial.println(
    "SYSTEM READY"
  );


  Serial.println(
    "Say VYOM after the prompt."
  );


  Serial.println(
    "=============================="
  );
}


// ============================================================
// LOOP
// ============================================================

void loop()
{
  Serial.println();

  Serial.println(
    "=============================="
  );


  Serial.println(
    "NEW TEST"
  );


  Serial.println(
    "=============================="
  );


  // ----------------------------------------------------------
  // Capture 1 second
  // ----------------------------------------------------------

  if (!captureAudio())
  {
    Serial.println(
      "Audio capture failed."
    );


    delay(1000);

    return;
  }


  // ----------------------------------------------------------
  // MFCC
  // ----------------------------------------------------------

  calculateMFCC();


  // ----------------------------------------------------------
  // CNN
  // ----------------------------------------------------------

  runVYOM();


  Serial.println();


  // Pause before next test

  delay(1000);
}