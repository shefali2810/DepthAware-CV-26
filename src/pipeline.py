# pipeline.py

import os
import shutil
from PIL import Image
from tqdm import tqdm

INPUT_DIR = "dataset/adversarial_data"
OUTPUT_DIR = "processed_data"
MIN_SIZE = 256

os.makedirs(OUTPUT_DIR, exist_ok=True)


def is_valid_image(path):
    try:
        img = Image.open(path)
        return img.size[0] >= MIN_SIZE and img.size[1] >= MIN_SIZE
    except:
        return False


def process_images():
    for root, _, files in os.walk(INPUT_DIR):
        for file in tqdm(files):
            if file.lower().endswith((".jpg", ".png", ".jpeg")):
                path = os.path.join(root, file)

                if is_valid_image(path):
                    new_path = os.path.join(OUTPUT_DIR, file)
                    shutil.copyfile(path, new_path)


if __name__ == "__main__":
    process_images()