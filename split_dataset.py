import os
import shutil
import random

random.seed(42)

POSITIVE_DIR = "wakeword_dataset/positive"
NEGATIVE_DIR = "VYOM_dataset/negative"
OUTPUT_DIR = "VYOM_dataset/split"

TRAIN = 0.70
VAL = 0.15
TEST = 0.15


def get_wav_files(folder):
    files = []

    for root, dirs, filenames in os.walk(folder):
        for filename in filenames:
            if filename.lower().endswith(".wav"):
                files.append(os.path.join(root, filename))

    return files


def split_files(files):
    random.shuffle(files)

    n = len(files)

    train_end = int(n * TRAIN)
    val_end = train_end + int(n * VAL)

    return (
        files[:train_end],
        files[train_end:val_end],
        files[val_end:]
    )


def copy_files(files, destination):
    os.makedirs(destination, exist_ok=True)

    for i, source in enumerate(files):
        filename = f"{i:05d}_{os.path.basename(source)}"
        shutil.copy2(source, os.path.join(destination, filename))


# Collect files
positive = get_wav_files(POSITIVE_DIR)
negative = get_wav_files(NEGATIVE_DIR)

print("Positive WAV files:", len(positive))
print("Negative WAV files:", len(negative))

# Split
pos_train, pos_val, pos_test = split_files(positive)
neg_train, neg_val, neg_test = split_files(negative)

# Create directories
for split in ["train", "validation", "test"]:
    os.makedirs(os.path.join(OUTPUT_DIR, split, "positive"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, split, "negative"), exist_ok=True)

# Copy
copy_files(pos_train, os.path.join(OUTPUT_DIR, "train", "positive"))
copy_files(pos_val, os.path.join(OUTPUT_DIR, "validation", "positive"))
copy_files(pos_test, os.path.join(OUTPUT_DIR, "test", "positive"))

copy_files(neg_train, os.path.join(OUTPUT_DIR, "train", "negative"))
copy_files(neg_val, os.path.join(OUTPUT_DIR, "validation", "negative"))
copy_files(neg_test, os.path.join(OUTPUT_DIR, "test", "negative"))

print("\n==============================")
print("DATASET SPLIT COMPLETE")
print("==============================")

print("\nPositive:")
print("Train:", len(pos_train))
print("Validation:", len(pos_val))
print("Test:", len(pos_test))

print("\nNegative:")
print("Train:", len(neg_train))
print("Validation:", len(neg_val))
print("Test:", len(neg_test))

print("\nOutput:")
print(OUTPUT_DIR)