import os
import sys
import json
import torch
import numpy as np
from PIL import Image

# Add DUSt3R to path
dust3r_path = "dust3r"
if dust3r_path not in sys.path:
    sys.path.insert(0, dust3r_path)

try:
    from dust3r.inference import inference
    from dust3r.model import AsymmetricCroCo3DStereo
    from dust3r.utils.image import load_images
except ImportError as e:
    print(f"Failed to import DUSt3R components. Is it installed in {dust3r_path}?")
    raise e

def load_dust3r_model(model_name="naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt", device="cuda"):
    model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)
    model.eval()
    return model

def filter_pairs_by_geometry(pairs_json, output_json, conf_threshold=1.5):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading DUSt3R model to {device}...")
    model = load_dust3r_model(device=device)

    with open(pairs_json, 'r') as f:
        pairs = json.load(f)

    valid_pairs = []
    print(f"Running geometric verification on {len(pairs)} pairs...")

    for pair in pairs:
        img1_path = pair.get('image1')
        img2_path = pair.get('image2')
        
        if not (os.path.exists(img1_path) and os.path.exists(img2_path)):
            continue

        try:
            # Load images as batches for dust3r
            imgs = load_images([img1_path, img2_path], size=512)
            
            with torch.no_grad():
                output = inference([tuple(imgs)], model, device, batch_size=1)
            
            # output['conf'] contains the confidence map
            conf_map = output['conf'][0] # Check first pair batch
            avg_conf = float(torch.median(conf_map).cpu().numpy())

            if avg_conf >= conf_threshold:
                pair['geometric_confidence'] = avg_conf
                valid_pairs.append(pair)
            else:
                print(f"Discarding pair {img1_path} & {img2_path} due to low confidence: {avg_conf:.3f}")

        except Exception as e:
            print(f"Inference failed for {img1_path}: {e}")

    with open(output_json, 'w') as f:
        json.dump(valid_pairs, f, indent=4)
    print(f"Saved {len(valid_pairs)} geometrically verified pairs to {output_json}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs_json', type=str, required=True, help="Input JSON with list of pairs [{'image1':'..', 'image2':'..'}]")
    parser.add_argument('--output_json', type=str, required=True)
    parser.add_argument('--conf', type=float, default=1.5)
    args = parser.parse_args()
    filter_pairs_by_geometry(args.pairs_json, args.output_json, args.conf)
