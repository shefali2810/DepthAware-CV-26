# clip_filter.py

import torch
import clip
from PIL import Image
import os
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load("ViT-B/32", device=device)

INPUT_DIR = "processed_data"
OUTPUT_DIR = "filtered_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Prompts for filtering
prompts = [
    "a complex indoor scene",
    "a cluttered room with multiple objects",
    "a detailed environment",
]

text = clip.tokenize(prompts).to(device)


def filter_images():
    for file in tqdm(os.listdir(INPUT_DIR)):
        path = os.path.join(INPUT_DIR, file)

        try:
            image = preprocess(Image.open(path)).unsqueeze(0).to(device)

            with torch.no_grad():
                image_features = model.encode_image(image)
                text_features = model.encode_text(text)

                similarity = (image_features @ text_features.T).softmax(dim=-1)

            score = similarity.max().item()

            if score > 0.3:  # threshold (tune later)
                os.rename(path, os.path.join(OUTPUT_DIR, file))

        except:
            continue


if __name__ == "__main__":
    filter_images()