# VYOM — Low-Latency Edge Voice Activator

An ultra-lightweight TinyML wake-word detection system designed for
edge devices using the ESP32-S3.

The system continuously listens locally for the custom wake word
**"VYOM"**. When the wake word is detected, the ESP32 can activate
remote speech recognition and send the subsequent audio to a backend
for transcription.

---

## Project

**SIH Problem Statement:** SIH26172  
**Title:** Low Latency and Efficient Voice Activator for Edge Devices  
**Organization:** ISRO  
**Category:** Hardware  
**Domain:** Miscellaneous

---

## System Architecture

```text
                ┌─────────────────┐
                │    INMP441      │
                │ MEMS Microphone │
                └────────┬────────┘
                         │ I2S
                         ▼
                ┌─────────────────┐
                │    ESP32-S3     │
                │                 │
                │ Audio Capture   │
                │      ↓          │
                │      VAD        │
                │      ↓          │
                │     MFCC        │
                │      ↓          │
                │ Tiny CNN / KWS  │
                │      ↓          │
                │ "VYOM" detected │
                └────────┬────────┘
                         │
                    WebSocket
                         │
                         ▼
                ┌─────────────────┐
                │  FastAPI Server  │
                │                 │
                │ Audio Buffering  │
                │      ↓          │
                │      ASR        │
                │      ↓          │
                │     Text        │
                └─────────────────┘
