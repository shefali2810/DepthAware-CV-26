# 👁️ DepthAware-CV-26 🧩

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Qwen2.5-VL](https://img.shields.io/badge/Model-Qwen2.5--VL-orange)
![Framework](https://img.shields.io/badge/Framework-PyTorch-red)

Welcome to the **DepthAware-CV-26** project! This repository contains a complete end-to-end pipeline designed to fine-tune Vision-Language Models (VLMs) like Qwen2.5-VL-7B-Instruct to inherently understand complex 3D geometry and metric spatial scales. 

By integrating multi-view stereo matchers and semantic segmenters, this system prevents models from failing or hallucinating when presented with optical illusions, mirror distortions, or complex physical spacing constraint breaks.

---

## 🏗️ Repository Architecture

This codebase is clean, tightly isolated, completely portable, and structured linearly across the development lifecycle:

* 📁 **`data_prep/`**: Automated procurement scripts targeting heavy spatial datasets (RealEstate10K, ScanNet, KITTI-360) and optical boundary testbeds (`scrape_adversarial.py`).
* 📁 **`src/`**: The core cross-modal verification engine. Contains semantic mapping (`florence_filter.py`), 3D metric geometry evaluation (`dust3r_filter.py`, scale_alignment.py), and OpenAI-integrated Chain-of-Thought (CoT) extraction (`generate_cot.py`).
* 📁 **`training/`**: Dedicated execution setup to precisely inject Spatial representations into Qwen2.5-VL using parameter-efficient LoRA (`lora_finetune.py`).
* 📁 **`evals/`**: Independent automated test-beds extracting real mathematical statistics to grade the final baseline hallucination scores.
* 📁 **`docs/`**: Official project literature surveys, design walkthroughs (`walkthrough.md`, `task.md`), and structural project milestones.

---

## 🚀 Execution & Quick Start

Because this system integrates sophisticated spatial packages (DUSt3R, DepthPro, Flash-Attention 2), background execution relies seamlessly on our hardened `micromamba` virtual environment named **SpatialScore**.

**To automatically sequence the entire standardizing, checking, aligning, and testing pipeline locally:**
```bash
chmod +x run_all.sh
bash run_all.sh
```

For strict manual execution logic, API variable handlers, or module-by-module CLI commands, please refer exactly to the officially maintained [Execution Manual (`RUN_GUIDE.md`)](RUN_GUIDE.md).