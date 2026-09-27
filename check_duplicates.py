import os
import hashlib
import numpy as np
import soundfile as sf

TEST_DIR = "VYOM_dataset/split/test/positive"

files = [
    "00009_spk_voice4_take_A_0106_normal_0.5m_clean.wav",
    "00067_spk_voice4_take_A_0102_loud_0.5m_clean.wav",
    "00085_spk_voice4_take_A_0010_quiet_0.5m_clean.wav"
]

audios = {}

for filename in files:
    path = os.path.join(TEST_DIR, filename)

    audio, sr = sf.read(path)

    audios[filename] = audio

    print("\n", filename)

    # Entire recording hash
    full_hash = hashlib.sha256(
        audio.tobytes()
    ).hexdigest()

    # First 1 second hash
    first_second = audio[:16000]

    first_hash = hashlib.sha256(
        first_second.tobytes()
    ).hexdigest()

    print("Full audio hash:")
    print(full_hash)

    print("First 1-second hash:")
    print(first_hash)


print("\n==============================")
print("PAIRWISE DIFFERENCES")
print("==============================")

names = list(audios.keys())

for i in range(len(names)):
    for j in range(i + 1, len(names)):

        a = audios[names[i]]
        b = audios[names[j]]

        difference = np.abs(a - b)

        print("\n")
        print(names[i])
        print("vs")
        print(names[j])

        print("Max difference:", np.max(difference))
        print("Mean difference:", np.mean(difference))

        print(
            "Identical:",
            np.array_equal(a, b)
        )

        print(
            "First 1 second identical:",
            np.array_equal(a[:16000], b[:16000])
        )