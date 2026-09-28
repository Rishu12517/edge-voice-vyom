import os
import numpy as np
import soundfile as sf

folder = os.path.expanduser(
    "~/Desktop/edge-voice-backend/wakeword_dataset/positive/speaker_04/Dataset2/positive"
)

files = sorted(
    f for f in os.listdir(folder)
    if f.lower().endswith(".wav")
)

print("Total WAV files:", len(files))
print()

audio = {}

for f in files:
    path = os.path.join(folder, f)

    x, sr = sf.read(path)

    if x.ndim > 1:
        x = x[:, 0]

    # Compare only the first 1 second,
    # because your model currently uses the first 1 second.
    x = x[:16000]

    audio[f] = x.astype(np.float32)


threshold = 0.0001

pairs = []

for i in range(len(files)):
    for j in range(i + 1, len(files)):

        a = audio[files[i]]
        b = audio[files[j]]

        n = min(len(a), len(b))

        if n == 0:
            continue

        diff = np.abs(a[:n] - b[:n])

        max_diff = np.max(diff)
        mean_diff = np.mean(diff)

        if max_diff < threshold:
            pairs.append(
                (files[i], files[j], max_diff, mean_diff)
            )


print("==============================")
print("NEAR-DUPLICATE PAIRS")
print("==============================")

print("Threshold:", threshold)
print("Pairs found:", len(pairs))
print()

for a, b, max_diff, mean_diff in pairs:
    print(a)
    print("vs")
    print(b)
    print("Max difference :", max_diff)
    print("Mean difference:", mean_diff)
    print()