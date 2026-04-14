import os
import sys
import torch
import numpy as np
import cv2

def load_maps(dust3r_path, depthpro_path):
    """
    Mock loading logic for the depth arrays.
    In practice, DUSt3R outputs a (H,W,3) pts3d mapping and DepthPro outputs an (H,W) array.
    """
    # dust3r mapping is relative, depthpro is absolute (metric)
    dust3r_pts = np.load(dust3r_path)  # Expecting shape (H,W,3)
    depthpro_map = np.load(depthpro_path) # Expecting shape (H,W)
    
    # Get Z dimension (depth) from DUSt3R
    if len(dust3r_pts.shape) == 3 and dust3r_pts.shape[2] == 3:
        dust3r_depth = dust3r_pts[:, :, 2]
    else:
        dust3r_depth = dust3r_pts

    # Ensure identical shape (Bilinear resize depthpro to match Dust3r)
    if depthpro_map.shape != dust3r_depth.shape:
        depthpro_map = cv2.resize(depthpro_map, (dust3r_depth.shape[1], dust3r_depth.shape[0]), interpolation=cv2.INTER_LINEAR)
        
    return dust3r_depth, depthpro_map

def align_scale_and_detect_anomalies(dust3r_depth, depthpro_map, valid_mask=None, threshold=2.0):
    """
    Implements: S = median(DepthPro / DUSt3R)
    Then checks for massive divergence indicating a "Golden Edge Case" (e.g. hitting a mirror).
    """
    if valid_mask is None:
        valid_mask = (dust3r_depth > 0) & (depthpro_map > 0)
        
    if not np.any(valid_mask):
        return None, False, 0.0
        
    d_dust3r = dust3r_depth[valid_mask]
    d_depthpro = depthpro_map[valid_mask]
    
    # Calculate global median scale multiplier
    S = np.median(d_depthpro / (d_dust3r + 1e-6))
    
    # Scale up DUSt3R to Metric Space
    metric_dust3r = dust3r_depth * S
    
    # Calculate geometric disparity
    diff = np.abs(metric_dust3r - depthpro_map)
    diff[~valid_mask] = 0.0
    
    # Golden Edge Case Detection:
    # If the maximum error between Ground Truth Geometry (DUSt3R) and Surface Prediction (DepthPro)
    # is extremely high (e.g. > threshold meters), this is a spatial hallucination.
    max_error = np.max(diff)
    mse = np.mean(diff[valid_mask] ** 2)
    
    is_golden_edge_case = max_error > threshold
    
    return S, is_golden_edge_case, mse

def process_batch(dust3r_dir, depthpro_dir):
    aligned_results = []
    
    for f in os.listdir(dust3r_dir):
        if f.endswith(".npy"):
            d3_path = os.path.join(dust3r_dir, f)
            dp_path = os.path.join(depthpro_dir, f)
            
            if os.path.exists(dp_path):
                try:
                    d3, dp = load_maps(d3_path, dp_path)
                    S, is_edge, mse = align_scale_and_detect_anomalies(d3, dp)
                    
                    aligned_results.append({
                        "file": f,
                        "scale_multiplier": float(S) if S else None,
                        "golden_edge_case": bool(is_edge),
                        "mse": float(mse)
                    })
                    print(f"[{f}] Scale={S:.3f} | Edge Case={is_edge} | MSE={mse:.3f}")
                except Exception as e:
                    print(f"Failed on {f}: {e}")
                    
    return aligned_results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dust3r_dir', type=str, required=True)
    parser.add_argument('--depthpro_dir', type=str, required=True)
    args = parser.parse_args()
    
    process_batch(args.dust3r_dir, args.depthpro_dir)
