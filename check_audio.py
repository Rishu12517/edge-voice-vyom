import os
import soundfile as sf

DATASET_DIRS = [
    "wakeword_dataset/positive",
    "VYOM_dataset/negative"
]

total = 0
good = 0
bad = 0

for dataset_dir in DATASET_DIRS:

    print(f"\nChecking: {dataset_dir}")

    for root, dirs, files in os.walk(dataset_dir):

        for filename in files:

            if not filename.lower().endswith(".wav"):
                continue

            total += 1
            path = os.path.join(root, filename)

            try:
                info = sf.info(path)

                ok = (
                    info.samplerate == 16000
                    and info.channels == 1
                    and info.subtype == "PCM_16"
                )

                if ok:
                    good += 1

                else:
                    bad += 1

                    print("\nBAD FILE:")
                    print(path)
                    print("Sample rate:", info.samplerate)
                    print("Channels:", info.channels)
                    print("Format:", info.subtype)

            except Exception as e:

                bad += 1

                print("\nERROR:")
                print(path)
                print(e)


print("\n==============================")
print("AUDIO DATASET CHECK")
print("==============================")
print("Total WAV files:", total)
print("Valid:", good)
print("Invalid:", bad)