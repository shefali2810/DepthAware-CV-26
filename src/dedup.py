# dedup.py

from imagededup.methods import PHash
import os

phasher = PHash()
encodings = phasher.encode_images(image_dir='filtered_data')

duplicates = phasher.find_duplicates(encoding_map=encodings)

for k, v in duplicates.items():
    for dup in v:
        try:
            os.remove(os.path.join("filtered_data", dup))
        except:
            pass