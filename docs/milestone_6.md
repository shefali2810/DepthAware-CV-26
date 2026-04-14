# Milestone 6: VLM Training & Evaluation

## Overview
With the 50k geometric dataset scrubbed, labeled, and mathematically refined, the final focus is wrapping the target architecture locally to fine-tune its spatial matrices so it learns the generated Chain-of-Thought depth rules.

## Execution Details
1. **LoRA Fine-Tuning Execution Environment:**
   Using HuggingFace `peft` integrations, we authored `lora_finetune.py`. The base `Qwen2.5-VL-3B` weights are locked (`requires_grad = False`). Only lightweight low-rank adaptation arrays mapping across structural dimension paths (`q_proj`, `v_proj`, `k_proj`, `o_proj`) are explicitly fine-tuned using the generated Chain-of-Thought answers.
2. **Compute Profile Allocation:**
   Given the 16-24GB local VRAM constraints, `fp16` quantization and gradient accumulation (`steps=4`) are utilized to keep local processing budgets firmly underneath the 10 GPU-hour estimate provided in the presentation proposal. The trainer executes in standard epoch saving structures.

## Execution Metrics (Post-Train Phase)
- **SpatialSense Evaluation:** (`eval_spatialsense.py`) Verifies broad reasoning metrics based on semantic geometric bounds (left of, closer than) executing across thousands of sample queries.
- **NYU-Depth V2 Zero-Shot:** (`eval_nyu.py`) Validates the `S = median` prediction tracking evaluating hit/miss percentages under a strict 10% threshold variance.
- **Adversarial Set Validation:** (`eval_hallucination.py`) Calculates geometric robustness by throwing constructed 2D illusions at the fine-tuned spatial matrix to log true hallucination rates.

**Status: [COMPLETED AND RUNNING]**
