import os
import csv
from datasets import load_dataset

IMAGE_DIR = "triplet_output"
OUTPUT_CSV = "metadata.csv"
DATASET_NAME = "artem9k/ai-text-detection-pile"
DATASET_SPLIT = "train"

print("Loading Hugging Face dataset...")
ds = load_dataset(DATASET_NAME, split=DATASET_SPLIT)

# text_id -> source
id_to_source = {}
for row in ds:
    id_to_source[str(row["id"])] = row["source"]

rows = []

for fname in sorted(os.listdir(IMAGE_DIR)):
    if not fname.endswith(".png"):
        continue

    try:
        text_id, variant_ext = fname.split("_", 1)
        variant = variant_ext.replace(".png", "")
    except ValueError:
        print(f"Skipping unexpected filename: {fname}")
        continue

    source = id_to_source.get(text_id)

    if source is None:
        print(f"Warning: text_id {text_id} not found in HF dataset")
        continue

    label = 1 if source.lower() == "ai" else 0

    rows.append({
        "image_name": fname,
        "text_id": text_id,
        "variant": variant,
        "label": label,
        "source": source
    })

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["image_name", "text_id", "variant", "label", "source"]
    )
    writer.writeheader()
    writer.writerows(rows)

print("======================================")
print(f"Metadata written to: {OUTPUT_CSV}")
print(f"Total images indexed: {len(rows)}")
print("======================================")
