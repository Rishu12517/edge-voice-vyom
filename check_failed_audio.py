import os
import numpy as np
import soundfile as sf

TEST_DIR = "VYOM_dataset/split/test/positive"

files = [
    "00009_spk_voice4_take_A_0106_normal_0.5m_clean.wav",
    "00067_spk_voice4_take_A_0102_loud_0.5m_clean.wav",
    "00085_spk_voice4_take_A_0010_quiet_0.5m_clean.wav"
]

for filename in files:

    path = os.path.join(TEST_DIR, filename)

    audio, sr = sf.read(path)

    print("\n==============================")
    print(filename)
    print("==============================")

    print("Sample rate:", sr)
    print("Samples:", len(audio))
    print("Duration:", round(len(audio) / sr, 3), "sec")

    print("Min:", float(np.min(audio)))
    print("Max:", float(np.max(audio)))
    print("Mean:", float(np.mean(audio)))
    print("RMS:", float(np.sqrt(np.mean(audio ** 2))))
    print("Std:", float(np.std(audio)))