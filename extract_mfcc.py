import os
import numpy as np
import librosa

DATASET_DIR = "VYOM_dataset/split"
OUTPUT_DIR = "VYOM_dataset/features"

SAMPLE_RATE = 16000
DURATION = 1.0
SAMPLES = int(SAMPLE_RATE * DURATION)

N_MFCC = 13
N_FFT = 512
HOP_LENGTH = 160


def extract_mfcc(filepath):

    audio, sr = librosa.load(
        filepath,
        sr=SAMPLE_RATE,
        mono=True
    )

    # Make every sample exactly 1 second
    if len(audio) < SAMPLES:
        audio = np.pad(
            audio,
            (0, SAMPLES - len(audio))
        )
    else:
        audio = audio[:SAMPLES]

    # MFCC
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    return mfcc.astype(np.float32)


def process_split(split):

    X = []
    y = []

    for label, class_name in enumerate(["negative", "positive"]):

        folder = os.path.join(
            DATASET_DIR,
            split,
            class_name
        )

        print(f"\nProcessing {split}/{class_name}")

        for filename in sorted(os.listdir(folder)):

            if not filename.lower().endswith(".wav"):
                continue

            filepath = os.path.join(folder, filename)

            try:
                features = extract_mfcc(filepath)

                X.append(features)
                y.append(label)

            except Exception as e:
                print("ERROR:", filepath)
                print(e)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    np.save(
        os.path.join(
            OUTPUT_DIR,
            f"X_{split}.npy"
        ),
        X
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            f"y_{split}.npy"
        ),
        y
    )

    print("\n", split)
    print("Samples:", len(X))
    print("Feature shape:", X.shape)
    print("Labels:", y.shape)


for split in ["train", "validation", "test"]:
    process_split(split)

print("\n==============================")
print("MFCC EXTRACTION COMPLETE")
print("==============================")