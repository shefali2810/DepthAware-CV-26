# Milestone 4: Depth Fusion & Golden Edge Cases

## Overview
The most critical thesis of this project is discovering situations where current pre-trained state-of-the-art vision models fundamentally hallucinate depth logic (e.g. assessing a 2D poster of a room as an actual 3D space). Milestone 4 implemented the specific mathematical triggers to automatically classify these anomalies.

## Execution Details
1. **Metric Scale Alignment Calculation:**
   Using `scale_alignment.py`, we defined the formula `S = median(DepthPro / DUSt3R)`. DepthPro inherently predicts metric distances monocularly but suffers heavily from spatial hallucination on paradoxes. DUSt3R constructs rigid, reliable multi-view point planes but lacks absolute metric scaling. Thus, calculating the global median scalar `S` aligns the geometric truth into metric coordinate space seamlessly.
2. **Disparity Tracking (MSE):**
   Once aligned `(Metric DUSt3R = DUSt3R_Depth * S)`, we subtracted the normalized arrays (`np.abs(Metric DUSt3R - DepthPro_map)`). 
3. **Golden Edge Case Flagging:**
   Any specific pixel coordinates exhibiting geometric disparity greater than `threshold = 2.0` meters signify a total model breakdown. We dynamically flag these images with `is_golden_edge_case = True`. This creates an autonomous "lie detector" capable of generating our Adversarial Robustness Set without intense manual human labeling.

**Status: [COMPLETED]**
