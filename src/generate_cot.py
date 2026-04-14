import os
import json
import base64
import requests
from dotenv import load_dotenv

# Load OpenAI key securely without exposing it in the script
load_dotenv('.env')
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("CRITICAL: OpenAI API key not found in .env")

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def generate_cot_for_image(image_path, is_golden_edge_case=False):
    """
    Submits the image to OpenAI GPT-4o vision to generate a Chain-of-Thought spatial reasoning block.
    """
    base64_image = encode_image(image_path)
    
    prompt_text = (
        "Analyze this image and describe the spatial depth relationships geometrically. "
        "Describe what objects are closer or further."
    )
    
    if is_golden_edge_case:
        prompt_text += (
            " \nCRITICAL INSTRUCTION: This image is a Golden Edge Case (e.g. an optical illusion, "
            "a mirror, TV reflection, or flat poster). Provide a Chain-of-Thought explaining exactly "
            "why standard monocular depth estimation might completely fail here, but why a multi-view "
            "geometrically-consistent system would succeed in spotting the illusion."
        )
        
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ],
        "max_tokens": 250
    }
    
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        print(f"Error {response.status_code}: {response.text}")
        return ""

def pop_annotations_with_cot(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    print(f"Connecting to OpenAI... Generating CoT for {len(data)} annotations.")
    
    for idx, entry in enumerate(data):
        img_path = os.path.join("/data/shefali/DepthAware2026Project", entry["image"])
        if os.path.exists(img_path):
            print(f"[{idx+1}/{len(data)}] Generating CoT for {entry['image']}...")
            
            # Since this is the Adversarial set, we flag them all as Golden Edge Cases
            # Usually, you'd match this against Phase 3's S-Scale divergence array flag
            cot_text = generate_cot_for_image(img_path, is_golden_edge_case=True)
            entry["cot"] = cot_text
            entry["answer"] = "Spatial verification complete."

    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)
        
    print(f"\nSuccessfully updated {json_path} with GPT-4o spatial reasoning annotations.")

if __name__ == "__main__":
    pop_annotations_with_cot("VQASynth_Dataset/annotations.json")
