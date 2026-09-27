import os
import shutil

POSITIVE_DIR = "wakeword_dataset/positive"
NEGATIVE_DIR = "VYOM_dataset/negative"
OUTPUT_DIR = "VYOM_dataset/split"

speaker_split = {
    "train": ["speaker_01", "speaker_02"],
    "validation": ["speaker_03"],
    "test": ["speaker_04"]
}
def get_wav_files(folder):
    files = []

    for root, dirs, filenames in os.walk(folder):
        for filename in filenames:
            if filename.lower().endswith(".wav"):
                files.append(os.path.join(root, filename))

    return files


def copy_files(files, destination):
    os.makedirs(destination, exist_ok=True)

    for i, source in enumerate(files):
        filename = f"{i:05d}_{os.path.basename(source)}"
        shutil.copy2(source, os.path.join(destination, filename))


# --------------------------------
# POSITIVE DATA
# --------------------------------

positive_counts = {}

for split, speakers in speaker_split.items():

    files = []

    for speaker in speakers:
        speaker_dir = os.path.join(POSITIVE_DIR, speaker)

        if not os.path.isdir(speaker_dir):
            print("WARNING: Missing:", speaker_dir)
            continue

        files.extend(get_wav_files(speaker_dir))

    destination = os.path.join(
        OUTPUT_DIR,
        split,
        "positive"
    )

    copy_files(files, destination)

    positive_counts[split] = len(files)


# --------------------------------
# NEGATIVE DATA
# --------------------------------

negative_files = get_wav_files(NEGATIVE_DIR)

# Simple deterministic split
negative_files.sort()

n = len(negative_files)

train_end = int(n * 0.70)
val_end = train_end + int(n * 0.15)

negative_split = {
    "train": negative_files[:train_end],
    "validation": negative_files[train_end:val_end],
    "test": negative_files[val_end:]
}

negative_counts = {}

for split, files in negative_split.items():

    destination = os.path.join(
        OUTPUT_DIR,
        split,
        "negative"
    )

    copy_files(files, destination)

    negative_counts[split] = len(files)


# --------------------------------
# RESULTS
# --------------------------------

print("\n==============================")
print("SPEAKER-INDEPENDENT SPLIT")
print("==============================")

print("\nPositive:")
print("Train:", positive_counts["train"])
print("Validation:", positive_counts["validation"])
print("Test:", positive_counts["test"])

print("\nNegative:")
print("Train:", negative_counts["train"])
print("Validation:", negative_counts["validation"])
print("Test:", negative_counts["test"])

print("\nOutput:")
print(OUTPUT_DIR)