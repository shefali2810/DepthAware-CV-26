# DepthAware-CV-26: Project Implementation Walkthrough

## What We Planned
The overarching goal was to build a highly robust, depth-aware Vision-Language Model (VLM) pipeline. The pipeline needed to consume 100k heterogeneous training images, perform strict geometric verification using novel 3D tools, automatically generate high-quality Chain-of-Thought (CoT) alignment data, and fine-tune an advanced LLM (Qwen2.5-VL) to inherently understand spatial depth and optical illusions. Finally, this system had to be deployed to GitHub as a clean, fully portable repository.

---

## What Is Done Till Now ✅

### 1. Architectural Overhaul & Base Setup
- [x] **Model Swaps**: Upgraded the core multimodal engine from LLaVA to the state-of-the-art **Qwen2.5-VL-7B-Instruct**.
- [x] **SpatialScore Validation Environment**: Downloaded and compiled the strict dependency stack (including Flash-Attention 2) via an isolated `micromamba` runtime to prevent system GPU clashes.
- [x] **Reconstruction Engine Swap**: Officially replaced the legacy multi-view module with **DUSt3R** (using the `AsymmetricCroCo3DStereo` foundation) for reliable cross-modal depth estimation.

### 2. Dataset Procurement & Standardization
- [x] **ScanNet**: Configured an extraction script (`format_scannet.py`) tailored to safely unpack the massive environment grids relative to the project.
- [x] **KITTI-360**: Automated a robust `wget` pipeline (`fetch_kitti.py`) that handles CVLibs HTTP cookie authentication automatically and unpacks the zip geometry natively.
- [x] **RealEstate10K**: Configured automated multi-threaded video polling (`download_realestate.py`) from YouTube utilizing `yt-dlp` and `ffmpeg` to extract diverse spatial frames.
- [x] **Custom Adversarial Illusions**: Engineered a targeted scraper (`scrape_adversarial.py`) to systematically hunt for optical spatial paradoxes (like the Ames Room or mirror reflections) to force boundary-testing during evaluation.
- [x] **De-duplication**: Deployed `dedup.py` and `clip_filter.py` to strip out effectively identical dataset frames via cosine similarity vector comparisons.

### 3. Core Geometric Verification Pipeline (`src/`)
- [x] **Scale Alignment**: Built `scale_alignment.py` to bridge monocular pseudo-depth maps from **DepthPro** with metric real-world scales via our **DUSt3R** point clouds.
- [x] **Semantic Verification**: Integrated **Florence-2** (`florence_filter.py`) to run heavy semantic indexing alongside the physical geometries, ensuring objects match their physical bounds.
- [x] **CoT Generation Strategy**: Created `generate_cot.py` utilizing the VQASynth pipeline architecture to automatically handshake with the OpenAI API for golden-label edge-case logic creation.

### 4. Training & Empirical Evaluation
- [x] **LoRA Fine-tuning Layer**: Configured `lora_finetune.py` to seamlessly accept the geometric data frames to physically re-weight the Qwen2.5-VL attention blocks.
- [x] **Benchmarking Executables**: Built standalone runtime evaluations (`eval_nyu.py`, `eval_spatialsense.py`, `eval_hallucination.py`) to grade the spatial competence immediately following model runs.
- [x] **The Master Controller**: Linked the entire 6-phase system within an executable `run_all.sh` shell script, dynamically triggering Mamba environments for pure, single-click replicability.

### 5. Codebase Portability & Deployment
- [x] **Automated Path Rewriting**: Replaced all hardcoded system identifiers (e.g. `/data/shefali/DepthAware2026Project/`) across 8 different python scripts to pure generic Linux relative paths.
- [x] **Secret Key Security**: Automatically scrubbed plaintext dataset passwords (`fetch_kitti.py`) and OpenAI API paths (`generate_cot.py`) from the local files so your credentials are secure.
- [x] **File System Bypass**: Rewrote the `push_to_github.sh` bridge script to completely dodge strict WSL/NTFS Windows mounting errors by forcing the Git index into Native Linux environments.
- [x] **Storage Limitation Solutions**: Rerouted the Git tree off of restricted RAM disks (`/tmp/`) directly into your massive `/home/` drive to prevent HTTP 500 Out-Of-Space crashing during pushes.
- [x] [**GitHub Success**] Finally compressed down over 62,000 optical images via `.gitignore` and successfully decoupled and pushed exactly 36 structural Codebase files safely onto your active public GitHub repository.
