import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCausalLM

def load_florence_model(model_id="microsoft/Florence-2-large"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype, trust_remote_code=True).to(device)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return model, processor, device

def run_florence_inference(model, processor, device, image, task_prompt="<DETAILED_CAPTION>"):
    inputs = processor(text=task_prompt, images=image, return_tensors="pt").to(device, model.dtype)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3,
        do_sample=False
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    parsed_answer = processor.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height))
    return parsed_answer

def main():
    dataset_root = "/data/vinit/datasets/scannetv2/scans"
    output_file = "scannet_semantic_index.json"
    
    # Load model
    print("Loading Florence-2 model...")
    model, processor, device = load_florence_model()
    
    scenes = sorted(os.listdir(dataset_root))
    print(f"Found {len(scenes)} scenes in ScanNet.")
    
    semantic_index = []
    
    for scene in tqdm(scenes, desc="Processing Scenes"):
        color_dir = os.path.join(dataset_root, scene, "color")
        if not os.path.exists(color_dir):
            continue
            
        # Get frame files
        frames = sorted([f for f in os.listdir(color_dir) if f.endswith(".jpg")])
        if not frames:
            continue
            
        # Pick a middle frame for semantic checking
        sample_frame_idx = len(frames) // 2
        sample_frame_path = os.path.join(color_dir, frames[sample_frame_idx])
        
        try:
            image = Image.open(sample_frame_path).convert("RGB")
            
            # Run Detailed Caption to check for complexity
            caption_result = run_florence_inference(model, processor, device, image, "<DETAILED_CAPTION>")
            detailed_caption = caption_result["<DETAILED_CAPTION>"]
            
            # Check for object density via detection (optional but helpful)
            # detect_result = run_florence_inference(model, processor, device, image, "<OD>")
            
            # Conceptual Filtering: Keep if caption length suggests complexity or specific keywords found
            # (e.g., 'cluttered', 'multiple objects', 'furniture')
            # For now, we save everything and will filter in the next step based on metadata.
            
            semantic_index.append({
                "scene": scene,
                "frame": frames[sample_frame_idx],
                "caption": detailed_caption,
                "total_frames": len(frames)
            })
            
        except Exception as e:
            print(f"Error processing {scene}: {e}")
            
        # Intermediate Save
        if len(semantic_index) % 50 == 0:
            with open(output_file, 'w') as f:
                json.dump(semantic_index, f, indent=2)

    with open(output_file, 'w') as f:
        json.dump(semantic_index, f, indent=2)
    print(f"Semantic index saved to {output_file}")

if __name__ == "__main__":
    main()
