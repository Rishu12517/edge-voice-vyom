import sounddevice as sd
import soundfile as sf
import os

SAMPLE_RATE = 16000
DURATION = 1.2

SPEAKER = "speaker_01"

OUTPUT_DIR = f"wakeword_dataset/positive/{SPEAKER}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("====================================")
print("   WAKE WORD DATASET RECORDER")
print("====================================")
print()
print("Speaker:", SPEAKER)
print("Sample rate:", SAMPLE_RATE, "Hz")
print("Duration:", DURATION, "seconds")
print()
print("Press ENTER, then say your wake word.")
print("Press Ctrl+C to stop.")
print()

count = 1

while True:

    input("Press ENTER to record...")

    print("🎤 Recording... SAY YOUR WAKE WORD!")

    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )

    sd.wait()

    filename = f"{SPEAKER}_{count:04d}.wav"
    filepath = os.path.join(OUTPUT_DIR, filename)

    sf.write(
        filepath,
        audio,
        SAMPLE_RATE,
        subtype="PCM_16"
    )

    print("✅ Saved:", filepath)
    print()

    count += 1