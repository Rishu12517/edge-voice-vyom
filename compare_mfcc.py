import librosa
import numpy as np

FILE = "wakeword_dataset/positive/speaker_04/Dataset2/positive/positive_001.wav"

audio, sr = librosa.load(FILE, sr=16000, mono=True)

audio = audio[:16000]

if len(audio) < 16000:
    audio = np.pad(audio, (0, 16000 - len(audio)))

print("Audio RMS:", np.sqrt(np.mean(audio ** 2)))

mfcc = librosa.feature.mfcc(
    y=audio,
    sr=16000,
    n_mfcc=13,
    n_fft=512,
    hop_length=160
)

print("Shape:", mfcc.shape)

print("\nFIRST MFCC FRAME:")
for i in range(13):
    print(f"MFCC[{i}] = {mfcc[i,0]:.6f}")

print("\nFRAME 50:")
for i in range(13):
    print(f"MFCC[{i}] = {mfcc[i,50]:.6f}")

print("\nFRAME 100:")
for i in range(13):
    print(f"MFCC[{i}] = {mfcc[i,100]:.6f}")