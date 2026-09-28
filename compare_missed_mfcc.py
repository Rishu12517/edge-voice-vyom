import os
import numpy as np
import librosa
import matplotlib.pyplot as plt

TEST_DIR = "VYOM_dataset/split/test/positive"

missed = [
    "00009_spk_voice4_take_A_0106_normal_0.5m_clean.wav",
    "00067_spk_voice4_take_A_0102_loud_0.5m_clean.wav",
    "00085_spk_voice4_take_A_0010_quiet_0.5m_clean.wav"
]

# Some correctly detected speaker-4 samples
correct = [
    "00011_spk_voice4_take_A_0088_normal_2.0m_rain.wav",
    "00021_spk_voice4_take_A_0082_normal_1.0m_room_tone.wav",
    "00053_spk_voice4_take_A_0023_loud_2.0m_room_tone.wav"
]

def get_mfcc(filename):
    path = os.path.join(TEST_DIR, filename)

    audio, sr = librosa.load(
        path,
        sr=16000,
        mono=True
    )

    # Exactly 1 second
    if len(audio) < 16000:
        audio = np.pad(
            audio,
            (0, 16000 - len(audio))
        )
    else:
        audio = audio[:16000]

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=16000,
        n_mfcc=13,
        n_fft=512,
        hop_length=160
    )

    return mfcc


print("\nMISSED SAMPLES")
print("==============================")

for filename in missed:

    mfcc = get_mfcc(filename)

    print(
        filename,
        "mean:",
        round(float(mfcc.mean()), 3),
        "std:",
        round(float(mfcc.std()), 3)
    )


print("\nCORRECTLY DETECTED SAMPLES")
print("==============================")

for filename in correct:

    mfcc = get_mfcc(filename)

    print(
        filename,
        "mean:",
        round(float(mfcc.mean()), 3),
        "std:",
        round(float(mfcc.std()), 3)
    )


# -----------------------------
# Visualize one missed vs one correct
# -----------------------------

missed_mfcc = get_mfcc(missed[0])
correct_mfcc = get_mfcc(correct[0])

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(
    missed_mfcc,
    aspect="auto",
    origin="lower"
)
plt.title("Missed VYOM")
plt.xlabel("Time")
plt.ylabel("MFCC")

plt.subplot(1, 2, 2)
plt.imshow(
    correct_mfcc,
    aspect="auto",
    origin="lower"
)
plt.title("Correct VYOM")
plt.xlabel("Time")
plt.ylabel("MFCC")

plt.tight_layout()

plt.savefig("mfcc_comparison.png")

print("\nSaved:")
print("mfcc_comparison.png")