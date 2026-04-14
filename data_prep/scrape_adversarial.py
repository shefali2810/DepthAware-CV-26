import os
import json
from bing_image_downloader import downloader

def scrape_adversarial_images():
    output_dir = 'dataset/adversarial_data'
    os.makedirs(output_dir, exist_ok=True)
    
    queries = [
        "mirror reflection room",
        "mirror illusion hallway",
        "fake window indoor",
        "poster wall realistic",
        "optical illusion room depth",
        "anamorphic illusion street"
    ]
    
    print(f"Starting controlled scraping for {len(queries)} adversarial categories...")
    
    # Download images
    for q in queries:
        try:
            downloader.download(q, limit=500, output_dir=output_dir, adult_filter_off=False, force_replace=False, timeout=60, verbose=False)
        except Exception as e:
            print(f"Error scraping {q}: {e}")

    # Standardize data structure and tag types as requested
    formatted_data = []
    for root, _, files in os.walk(output_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                filepath = os.path.join(root, file)
                
                # Tagging logic
                adv_type = "illusion"
                if "mirror" in root.lower(): adv_type = "mirror"
                elif "fake" in root.lower() or "window" in root.lower(): adv_type = "fake_window"
                elif "poster" in root.lower(): adv_type = "poster"
                
                formatted_data.append({
                    "image": filepath,
                    "source": "bing_scraper",
                    "type": adv_type
                })

    json_path = 'dataset/adversarial.json'
    with open(json_path, 'w') as f:
        json.dump(formatted_data, f, indent=4)
        
    print(f"\nScraping complete! Indexed {len(formatted_data)} adversarial images into {json_path}.")

if __name__ == "__main__":
    scrape_adversarial_images()
