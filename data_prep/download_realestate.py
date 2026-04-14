import os
import json
import glob
import subprocess

def download_and_extract(metadata_dir, output_dir, target_frames=30000):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "videos"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "frames"), exist_ok=True)
    
    txt_files = glob.glob(os.path.join(metadata_dir, "train", "*.txt"))
    formatted_data = []
    frames_collected = 0
    videos_processed = 0
    
    print(f"Found {len(txt_files)} metadata files. Starting extraction to '{output_dir}'")
    
    for txt_file in txt_files:
        if frames_collected >= target_frames:
            break
            
        with open(txt_file, 'r') as f:
            url = f.readline().strip()
            
        if not url.startswith("http"):
            continue
            
        video_id = url.split("v=")[-1].split("&")[0][:11] # Basic extraction
        video_path = os.path.join(output_dir, "videos", f"{video_id}.mp4")
        frame_prefix = os.path.join(output_dir, "frames", f"{video_id}_%04d.jpg")
        
        # Download
        if not os.path.exists(video_path):
            try:
                # To prevent massive downloads of live streams or huge bounds, format and limit it
                subprocess.run(
                    ["yt-dlp", "-f", "worst[ext=mp4]", "--match-filter", "duration < 600", "-o", video_path, url],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120
                )
            except Exception as e:
                continue
                
        if not os.path.exists(video_path):
            continue
            
        # Extract Frames
        try:
            # check if frames already exist
            existing_frames = glob.glob(os.path.join(output_dir, "frames", f"{video_id}_*.jpg"))
            if not existing_frames:
                subprocess.run(
                    ["ffmpeg", "-i", video_path, "-vf", "fps=3", "-hide_banner", "-loglevel", "error", frame_prefix],
                    timeout=120
                )
                
            new_frames = sorted(glob.glob(os.path.join(output_dir, "frames", f"{video_id}_*.jpg")))
            for frame in new_frames:
                formatted_data.append({
                    "image": frame,
                    "source": "realestate10k"
                })
                frames_collected += 1
                
            videos_processed += 1
            if videos_processed % 5 == 0:
                print(f"Processed {videos_processed} videos. Frames collected: {frames_collected}/{target_frames}")
                
            # Maintain local original raw video file
            # os.remove(video_path)
            
        except Exception as e:
            pass

    # Save JSON standardization
    output_json = "dataset/realestate10k.json"
    with open(output_json, 'w') as f:
        json.dump(formatted_data, f, indent=4)
        
    print(f"Finished! Collected {frames_collected} frames from {videos_processed} videos.")
    print(f"Metadata saved to {output_json}")

if __name__ == "__main__":
    download_and_extract(
        metadata_dir="dataset/RealEstate10K",
        output_dir="dataset/realestate",
        target_frames=30000
    )
