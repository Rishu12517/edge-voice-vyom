import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

# Load model
model = tf.keras.models.load_model("vyom_tinycnn.keras")

# Load test data
X_test = np.load(
    "VYOM_dataset/features/X_test.npy"
)

y_test = np.load(
    "VYOM_dataset/features/y_test.npy"
)

# Add CNN channel
X_test = X_test[..., np.newaxis]

# Get probabilities
probabilities = model.predict(
    X_test,
    verbose=0
).flatten()

# Default threshold
THRESHOLD = 0.5

predictions = (
    probabilities >= THRESHOLD
).astype(int)

# Confusion matrix
cm = confusion_matrix(
    y_test,
    predictions
)

print("\n==============================")
print("CONFUSION MATRIX")
print("==============================")

print(cm)

print("\nRows = Actual")
print("Columns = Predicted")

print("\n             Predicted")
print("             NEG   POS")
print("Actual NEG   ", cm[0][0], " ", cm[0][1])
print("Actual POS   ", cm[1][0], " ", cm[1][1])


# Classification report
print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================")

print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "Negative",
            "VYOM"
        ]
    )
)


# Explicit error counts
false_positives = np.sum(
    (y_test == 0) & (predictions == 1)
)

false_negatives = np.sum(
    (y_test == 1) & (predictions == 0)
)

true_positives = np.sum(
    (y_test == 1) & (predictions == 1)
)

true_negatives = np.sum(
    (y_test == 0) & (predictions == 0)
)

print("\n==============================")
print("WAKE-WORD RESULTS")
print("==============================")

print("True Positives :", true_positives)
print("True Negatives :", true_negatives)
print("False Positives:", false_positives)
print("False Negatives:", false_negatives)