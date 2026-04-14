# Milestone 3: Geometric & Semantic Verification

## Overview
Based directly on the initial filtering methodology from SpatialVLM and DUSt3R, we set up dual-layer pipeline logic designed to brutally cut down the 100k generated pairs into a reliable golden 50k subset.

## Execution Details
1. **Semantic Filtering (Florence-2 / CLIP):**
   We implemented `clip_filter.py` and `florence_filter.py` across the domains. This semantic parsing rejects "low-entropy" flat walls, simple objects, and empty scenes to guarantee high-complexity geometric relations in our final cut. It achieved highly optimized zero-shot validation via OpenAI's ViT-B/32 models scoring `> 0.3`.
2. **Hash Deduplication:**
   To guarantee model stability, identical frames downloaded from streams or standard scrapers were automatically deleted using `imagededup` and `PHash` hamming distance measurements via `dedup.py`.
3. **Multi-View Verification (DUSt3R):**
   `dust3r_filter.py` was architected to take consecutive trajectory sequences and feed them to `naver/DUSt3R_ViTLarge_BaseDecoder`. If DUSt3R could not calculate a geometric confidence correlation (`conf >= 1.5`), the pair is assumed visually invalid and permanently discarded.

**Status: [COMPLETED]**
