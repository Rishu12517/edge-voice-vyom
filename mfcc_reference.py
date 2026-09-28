import glob
import numpy as np
import librosa

files = glob.glob(
    "wakeword_dataset/positive/speaker_04/**/*.wav",
    recursive=True
)

print("Found WAV files:", len(files))

FILE = files[0]

print("Using:", FILE)

audio, sr = librosa.load(
    FILE,
    sr=16000,
    mono=True
)

if len(audio) < 16000:
    audio = np.pad(
        audio,
        (0, 16000 - len(audio))
    )
else:
    audio = audio[:16000]

print("Audio shape:", audio.shape)
print("Audio min:", audio.min())
print("Audio max:", audio.max())
print("Audio RMS:", np.sqrt(np.mean(audio ** 2)))

mfcc = librosa.feature.mfcc(
    y=audio,
    sr=16000,
    n_mfcc=13,
    n_fft=512,
    hop_length=160
)

mfcc = mfcc.astype(np.float32)

print("MFCC shape:", mfcc.shape)

print("\nFirst 5 frames:")
print(mfcc[:, :5])

np.save("reference_mfcc.npy", mfcc)