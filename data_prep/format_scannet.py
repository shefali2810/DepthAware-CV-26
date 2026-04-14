import os
import json
import glob

def format_scannet(scans_dir, output_file):
    scans = os.listdir(scans_dir)
    formatted_data = []
    
    for scan in scans:
        color_dir = os.path.join(scans_dir, scan, "color")
        depth_dir = os.path.join(scans_dir, scan, "depth")
        pose_dir = os.path.join(scans_dir, scan, "pose")
        
        if not os.path.isdir(color_dir):
            continue
            
        color_files = sorted(glob.glob(os.path.join(color_dir, "*.jpg")))
        for color_file in color_files:
            basename = os.path.basename(color_file)
            prefix = basename.split('.')[0]
            
            entry = {
                "image": color_file,
                "source": "scannet"
            }
            
            depth_file = os.path.join(depth_dir, f"{prefix}.png")
            if os.path.exists(depth_file):
                entry["depth"] = depth_file
                
            pose_file = os.path.join(pose_dir, f"{prefix}.txt")
            if os.path.exists(pose_file):
                entry["pose"] = pose_file
                
            formatted_data.append(entry)
            
    with open(output_file, 'w') as f:
        json.dump(formatted_data, f, indent=4)
    print(f"Standardized {len(formatted_data)} frames from ScanNet.")

if __name__ == "__main__":
    base_proj_dir = os.path.dirname(os.path.abspath(__file__))
    format_scannet(os.path.join(base_proj_dir, "dataset", "scannet"), os.path.join(base_proj_dir, "dataset", "scannet.json"))
