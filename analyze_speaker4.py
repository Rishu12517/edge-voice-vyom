import os
import numpy as np
import tensorflow as tf

MODEL_PATH = "vyom_tinycnn.keras"
TEST_DIR = "VYOM_dataset/split/test/positive"
X_PATH = "VYOM_dataset/features/X_test.npy"

# Load model and features
model = tf.keras.models.load_model(MODEL_PATH)
X_test = np.load(X_PATH)

# Positive test filenames
files = sorted([
    f for f in os.listdir(TEST_DIR)
    if f.lower().endswith(".wav")
])

# IMPORTANT:
# X_test contains negative samples first, then positive samples.
# There are 33 negative test samples.
positive_start = 33

X_positive = X_test[positive_start:]

# Predictions
probabilities = model.predict(
    X_positive[..., np.newaxis],
    verbose=0
).flatten()

print("\n==============================")
print("SPEAKER 4 TEST ANALYSIS")
print("==============================")

speaker4 = []

for filename, probability in zip(files, probabilities):

    if "spk_voice4" in filename:

        speaker4.append((probability, filename))

# Sort from lowest confidence to highest
speaker4.sort()

print("\nTotal Speaker 4 samples:", len(speaker4))

print("\nLOWEST CONFIDENCE")
print("------------------------------")

for probability, filename in speaker4[:15]:
    print(f"{probability:.4f}  {filename}")

print("\nHIGHEST CONFIDENCE")
print("------------------------------")

for probability, filename in speaker4[-15:]:
    print(f"{probability:.4f}  {filename}")

# Statistics
values = np.array([p for p, _ in speaker4])

print("\n==============================")
print("STATISTICS")
print("==============================")

print("Minimum probability:", round(values.min(), 4))
print("Maximum probability:", round(values.max(), 4))
print("Average probability:", round(values.mean(), 4))
print("Median probability:", round(np.median(values), 4))

print("\nBelow 0.5:",
      np.sum(values < 0.5))

print("0.5 or higher:",
      np.sum(values >= 0.5))