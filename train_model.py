import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# -----------------------------
# Load MFCC features
# -----------------------------

X_train = np.load("VYOM_dataset/features/X_train.npy")
y_train = np.load("VYOM_dataset/features/y_train.npy")

X_val = np.load("VYOM_dataset/features/X_validation.npy")
y_val = np.load("VYOM_dataset/features/y_validation.npy")

X_test = np.load("VYOM_dataset/features/X_test.npy")
y_test = np.load("VYOM_dataset/features/y_test.npy")

print("Train:", X_train.shape, y_train.shape)
print("Validation:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)


# -----------------------------
# Add CNN channel dimension
# -----------------------------

X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]
X_test = X_test[..., np.newaxis]

print("\nCNN input shape:", X_train.shape)


# -----------------------------
# Build Tiny CNN
# -----------------------------

model = models.Sequential([

    layers.Input(shape=(13, 101, 1)),

    layers.Conv2D(
        8,
        kernel_size=(3, 3),
        padding="same",
        activation="relu"
    ),

    layers.MaxPooling2D(
        pool_size=(2, 2)
    ),

    layers.Conv2D(
        16,
        kernel_size=(3, 3),
        padding="same",
        activation="relu"
    ),

    layers.MaxPooling2D(
        pool_size=(2, 2)
    ),

    layers.Flatten(),

    layers.Dense(
        16,
        activation="relu"
    ),

    layers.Dropout(0.2),

    layers.Dense(
        1,
        activation="sigmoid"
    )
])


# -----------------------------
# Compile
# -----------------------------

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# -----------------------------
# Train
# -----------------------------

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=16,
    shuffle=True
)


# -----------------------------
# Test
# -----------------------------

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=1
)

print("\n==============================")
print("TEST RESULT")
print("==============================")

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)


# -----------------------------
# Save model
# -----------------------------

model.save("vyom_tinycnn.keras")

print("\nModel saved:")
print("vyom_tinycnn.keras")