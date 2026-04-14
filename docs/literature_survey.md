# Detailed Literature Survey: Depth-Aware Spatial Reasoning in VLMs

## 1. SpatialVLM: Endowing Vision-Language Models with Spatial Reasoning Capabilities (Chen et al., 2024)
**Core Contribution:** 
SpatialVLM aims to solve the inherent 2D semantic bias found in traditional Large Vision-Language Models (VLMs) by training them directly on massive arrays of depth-grounded visual question answering (VQA) data. 

**Methodology & Limitations:**
The authors utilized Monocular Depth Estimators (MDE) to generate relative distance mappings for millions of images, effectively lifting 2D planes into estimated 3D annotations. By training the VLM to interpret textual queries about these metric maps (e.g., "Is object A closer than object B?"), the model's spatial reasoning vastly improved. However, the critical flaw in the SpatialVLM pipeline is its pure reliance on monocular depth. Without multi-view geometric verification, the estimator falls victim to the "Oracle Paradox"—hallucinating 3D depth when analyzing 2D planar surfaces like mirrors, televisions, or realistic posters. 

**Relevance to Project:**
This paper forms the foundational baseline of our project. We are adopting the exact VQA structural training methodology of SpatialVLM, but replacing their flawed monocular generation pipeline with mathematically verified, multi-view geometric consistency.

---

## 2. DUSt3R: Geometric 3D Vision Made Easy (Wang et al., 2024)
**Core Contribution:** 
DUSt3R introduces a groundbreaking, uncalibrated paradigm for dense 3D reconstruction using multi-view images without requiring prior knowledge of camera intrinsics or spatial pose bounding.

**Methodology & Strengths:**
Operating natively on arbitrary image pairs, DUSt3R triangulates corresponding pixels across views to output a unified, viewpoint-agnostic 3D point map alongside a rigid confidence array. Because it operates on cross-view geometry rather than semantic guessing, it cannot be fooled by optical illusions that only trick single-vantage points (like mirrors). If an object lacks physical depth across multiple frames, DUSt3R immediately flags the geometry as invalid.

**Relevance to Project:**
DUSt3R acts as our autonomous "Lie Detector." By checking DUSt3R's geometric point maps against a monocular depth estimator's output via Phase 3's `S = median` alignment, we can instantly filter out false spatial data and identify the "Golden Edge Cases" needed for our Adversarial Robustness Set.

---

## 3. MVImgNet: A Large-Scale Dataset of Multi-View Images (Yu et al., 2023)
**Core Contribution:** 
MVImgNet bridges a critical data gap in the computer vision ecosystem between highly specific indoor 3D datasets (like ScanNet) and highly generalized, object-centric 2D datasets (like ImageNet).

**Methodology & Strengths:**
The dataset provides millions of multi-view trajectories orbiting singular objects and sprawling scenes from diverse, everyday environments. This provides the necessary continuous-frame overlap required for geometric triangulation models.

**Relevance to Project:**
While we rely on sequences from ScanNet and RealEstate10K, MVImgNet provides the conceptual framework for ensuring that our 100k generated training samples remain scalable and computationally viable. It proves that sampling multi-view temporal data across varied semantics yields stronger generalized reasoning than solely training on domain-specific indoor scenes.

---

## 4. VGGT: Visual Geometry Grounded Transformers (Wang et al., 2025)
**Core Contribution:** 
VGGT explores the direct topological integration of geometric embeddings (e.g., rigid voxel grids, depth maps, constraint vectors) natively into the Transformer attention architecture.

**Methodology & Strengths:**
Instead of treating depth simply as an extra image channel or relying on a textual proxy (VQA text), VGGT fundamentally alters how self-attention heads weigh visual patches by calculating their explicit geometric proximity in 3D space. 

**Relevance to Project:**
While our primary strategy relies on lightweight parameter-efficient fine-tuning (PEFT/LoRA) using textual Chain-of-Thought (CoT) alignment on pre-trained weights (Qwen2.5-VL), VGGT highlights the absolute structural maximum capability of depth-aware modeling. It provides the necessary literature citation to justify why aligning 2D semantics with true geometric constraints solves VLM hallucination.
