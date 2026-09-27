import os
import numpy as np
import tensorflow as tf

# -----------------------------
# Paths
# -----------------------------
MODEL_PATH = "vyom_tinycnn.keras"
TEST_DIR = "VYOM_dataset/split/test"
X_PATH = "VYOM_dataset/features/X_test.npy"
Y_PATH = "VYOM_dataset/features/y_test.npy"

# -----------------------------
# Load model and test data
# -----------------------------
print("Loading model...")

model = tf.keras.models.load_model(MODEL_PATH)

X_test = np.load(X_PATH)
y_test = np.load(Y_PATH)

# CNN expects: (samples, 13, 101, 1)
X_test_cnn = X_test[..., np.newaxis]

print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# -----------------------------
# Rebuild exact filename order
# -----------------------------
files = []

# IMPORTANT:
# This must match the order used by extract_mfcc.py
for label, class_name in enumerate(["negative", "positive"]):

    folder = os.path.join(TEST_DIR, class_name)

    for filename in sorted(os.listdir(folder)):

        if filename.lower().endswith(".wav"):

            filepath = os.path.join(folder, filename)

            files.append((filepath, label))

# -----------------------------
# Verify alignment
# -----------------------------
print("Number of files:", len(files))
print("Number of features:", len(X_test))
print("Number of labels:", len(y_test))

assert len(files) == len(X_test) == len(y_test), (
    "ERROR: File/feature/label counts do not match!"
)

print("Filename ↔ feature ↔ label alignment OK.")

# -----------------------------
# Predictions
# -----------------------------
print("\nRunning predictions...")

probabilities = model.predict(
    X_test_cnn,
    verbose=0
).flatten()

predictions = (probabilities >= 0.5).astype(int)

# -----------------------------
# False Negatives
# -----------------------------
print("\n==============================")
print("FALSE NEGATIVES")
print("==============================")

false_negative_count = 0

for (filepath, label), actual, probability, prediction in zip(
    files,
    y_test,
    probabilities,
    predictions
):

    # Actual VYOM but predicted negative
    if actual == 1 and prediction == 0:

        false_negative_count += 1

        print()
        print("File:", os.path.basename(filepath))
        print("Path:", filepath)
        print("Probability:", round(float(probability), 4))
        print("Actual: VYOM")
        print("Predicted: NOT VYOM")

print()
print("Total False Negatives:", false_negative_count)

# -----------------------------
# False Positives
# -----------------------------
print("\n==============================")
print("FALSE POSITIVES")
print("==============================")

false_positive_count = 0

for (filepath, label), actual, probability, prediction in zip(
    files,
    y_test,
    probabilities,
    predictions
):

    # Actual negative but predicted VYOM
    if actual == 0 and prediction == 1:

        false_positive_count += 1

        print()
        print("File:", os.path.basename(filepath))
        print("Path:", filepath)
        print("Probability:", round(float(probability), 4))
        print("Actual: NEGATIVE")
        print("Predicted: VYOM")

print()
print("Total False Positives:", false_positive_count)