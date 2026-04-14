# create_annotations.py

import json
import os

IMAGE_DIR = "VQASynth_Dataset/images"
OUTPUT_FILE = "VQASynth_Dataset/annotations.json"

data = []

for i, file in enumerate(os.listdir(IMAGE_DIR)):
    entry = {
        "id": i,
        "image": f"images/{file}",
        "question": "Which object is closer?",
        "answer": "unknown",
        "cot": ""
    }
    data.append(entry)

with open(OUTPUT_FILE, "w") as f:
    json.dump(data, f, indent=4)