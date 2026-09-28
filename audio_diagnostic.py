import glob
import numpy as np
import librosa

files = glob.glob(
    "wakeword_dataset/positive/speaker_04/**/*.wav",
    recursive=True
)

FILE = files[0]

print("Using:", FILE)

audio, sr = librosa.load(
    FILE,
    sr=16000,
    mono=True
)

audio = audio[:16000]

print("\n==============================")
print("AUDIO")
print("==============================")

print("Sample rate:", sr)
print("Samples:", len(audio))
print("Min:", audio.min())
print("Max:", audio.max())
print("Mean:", audio.mean())
print("Std:", audio.std())

print(
    "RMS:",
    np.sqrt(np.mean(audio ** 2))
)

print(
    "Centered RMS:",
    np.sqrt(np.mean((audio - audio.mean()) ** 2))
)

print("\nFirst 30 samples:")
print(audio[:30])

print("\n==============================")
print("MFCC")
print("==============================")

mfcc = librosa.feature.mfcc(
    y=audio,
    sr=16000,
    n_mfcc=13,
    n_fft=512,
    hop_length=160
)

print("Shape:", mfcc.shape)
print("Min:", mfcc.min())
print("Max:", mfcc.max())
print("Mean:", mfcc.mean())
print("Std:", mfcc.std())

print("\nFirst frame:")
print(mfcc[:, 0])

print("\nFrame 50:")
print(mfcc[:, 50])