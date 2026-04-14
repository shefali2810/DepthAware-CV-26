# Milestone 5: CoT VQA Generation

## Overview
A standard VQA output (e.g. '1.2 meters') does not teach an LLM *how* to reason conceptually through depth paradoxes. Milestone 5 programmatically generates Chain-of-Thought (CoT) logic to fine-tune the model to understand its own hallucination triggers.

## Execution Details
1. **OpenAI Integration Pipeline:**
   We wrote `generate_cot.py` which actively imports the user's secured OpenAI API key without openly placing it onto disk tracking software. This API invokes the highly conceptual `gpt-4o` multimodal inference system over the chosen dataset instances.
2. **Spatial Distance Prompts (Normal Set):**
   For standard dataset inclusions like ScanNet and RealEstate10K, the generation query asks the model to document depth layers contextually (`"Describe what objects are closer or further"`).
3. **Golden Edge Case Paradox Engine:**
   For images specifically flagged as hallucinatory in Milestone 4, we inject an explicit prompt system: `"CRITICAL INSTRUCTION: This image is a Golden Edge Case... Provide a Chain-of-Thought explaining exactly why standard monocular depth estimation might completely fail here, but why a multi-view geometrically-consistent system would succeed in spotting the illusion."`
4. **JSON Standardization:**
   All `gpt-4o` responses are merged via Python directly into the target `VQASynth_Dataset/annotations.json` file structure under the `cot` dictionary field, ready strictly for PEFT loading.

**Status: [COMPLETED]**
