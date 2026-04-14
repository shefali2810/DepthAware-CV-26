# Depth-Aware Spatial Reasoning Project: Execution Guide 🚀

Because this project utilizes heavy VLM matrices (like Qwen2.5-VL) and multi-view geometric modules (DUSt3R), it requires strict execution through the configured **SpatialScore micromamba environment**. 

To run any script in this project, you must **always prefix your commands** with the environment wrapper, or use the automated execution script.

---

### Automated Full Pipeline Execution:
Instead of running scripts manually, the easiest way to launch the complete end-to-end processing pipeline is:
```bash
chmod +x run_all.sh
bash run_all.sh
```

---

## Manual Execution Sequence

If you wish to execute specific pipeline phases directly, follow this order navigating through the structurally decoupled directories:

### 1. Dataset Procurement & Formatting (`data_prep/` & `src/`)
Format the unstructured datasets into the unified `VQASynth` evaluation architecture.
```bash
micromamba run -n SpatialScore python data_prep/download_realestate.py
micromamba run -n SpatialScore python src/pipeline.py
micromamba run -n SpatialScore python src/clip_filter.py
micromamba run -n SpatialScore python src/dedup.py
micromamba run -n SpatialScore python src/create_annotation.py
```

### 2. Geometric & Semantic Verification (`src/`)
Filter out low-complexity scenes and mathematically confirm absolute geometry relationships using Florence-2 indexing and DUSt3R multi-view stereo matches.
```bash
micromamba run -n SpatialScore python src/florence_filter.py
micromamba run -n SpatialScore python src/dust3r_filter.py --pairs_json [path] --output_json [verified]
```

### 3. Metric Scale Alignment (`src/`)
Align DUSt3R geometry to monocular DepthPro and isolate spatial discrepancies representing "Golden Edge Cases."
```bash
micromamba run -n SpatialScore python src/scale_alignment.py --dust3r_dir [path_to_pts] --depthpro_dir [path_to_maps]
```

### 4. Golden Edge Case CoT Generation (`src/`)
Trigger the secured generic OpenAI API logic to automatically analyze failed spatial assumptions and generate detailed linguistic Chain-of-Thought arrays.
```bash
micromamba run -n SpatialScore python src/generate_cot.py 
```

### 5. Start LoRA Model Fine-Tuning (`training/`)
Execute the parameter-efficient fine-tuning (PEFT) loop on the base Qwen2.5-VL model utilizing the generated json logic structure. *(Requires 16GB-24GB VRAM)*.
```bash
micromamba run -n SpatialScore python training/lora_finetune.py
```

### 6. Run Empirical Evaluations (`evals/`)
Trigger automated evaluations against external benchmark testbeds.
```bash
micromamba run -n SpatialScore python evals/eval_nyu.py
micromamba run -n SpatialScore python evals/eval_hallucination.py
micromamba run -n SpatialScore python evals/eval_spatialsense.py
```

---

### Quick Optical Illusion Probing (Optional)
Directly test how baseline Qwen2.5-VL completely fails on an inputted optical spatial paradox:
```bash
micromamba run -n SpatialScore python src/SpaceQwen.py
```
