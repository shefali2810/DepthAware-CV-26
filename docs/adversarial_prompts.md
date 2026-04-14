# SpaceThinker-Qwen2.5VL Limitations & Adversarial Prompts

The goal of this document is to catalog the specific edge-case prompts and visual paradoxes where a pure Monocular Vision-Language Model (like `SpaceThinker-Qwen2.5VL-3B`) fails to accurately reason about metric depth and spatial relationships. 

Because these models rely on 2D visual priors rather than multi-view geometric checks (like DUSt3R), they fall for the **Monocular Oracle Paradox**.

Here are 5 core limitations and the specific prompts you can use to break the model:

### 1. The Mirror / Reflection Hallucination
*Limitation: The model thinks reflections are real 3D extensions of a room.*
* **Image Input:** A photograph of a large mirror on a wall reflecting the rest of the room.
* **Adversarial Prompt:** "If I throw a ball perfectly straight ahead into the exact center of this image, how far will it travel in meters before hitting a solid physical surface?"
* **Why it fails:** The model will estimate the depth of the *reflected* room instead of the physical surface of the mirror, giving a completely incorrect spatial distance.

### 2. The Trompe-l'œil (Flat Poster) Trap
*Limitation: The model cannot differentiate between high-resolution 2D textures and actual 3D geometry.*
* **Image Input:** A realistic large poster of an open doorway or a realistic painting on a brick wall.
* **Adversarial Prompt:** "Describe the physical depth of the archway in the center. Is there enough physical space for an adult human to walk through it?"
* **Why it fails:** The model will hallucinate deep 3D coordinates behind the wall and say "Yes", failing to realize it is a flat 2D graphic.

### 3. Transparent Surface Blindness (Glass)
*Limitation: Depth models often pass directly through clear obstacles.*
* **Image Input:** A photo taken looking directly through a closed, spotless glass door (with no heavy glare).
* **Adversarial Prompt:** "What is the exact distance in meters to the nearest physical obstacle directly in front of the camera?"
* **Why it fails:** It will ignore the glass barrier entirely and return the depth of the objects on the *other side* of the glass.

### 4. Forced Perspective (Scale Disconnect)
*Limitation: Models rely on semantic context for scale (e.g., "cars are big").*
* **Image Input:** A macro photograph of a HotWheels toy car sitting on a normal carpet or sidewalk, shot to look realistic.
* **Adversarial Prompt:** "What is the absolute physical height of that car in meters?"
* **Why it fails:** Without binocular depth cues, the model assumes it's a real car and will confidently answer something like "1.5 meters," completely failing to notice the scale of the carpet fibers.

### 5. Water Reflections vs Ground Depth
*Limitation: Confusing a reflected sky/environment with physical drop-offs.*
* **Image Input:** A perfectly still puddle reflecting a tall building on an otherwise flat street.
* **Adversarial Prompt:** "What is the geometric depth of the area directly inside the puddle? Is it a flat surface or a deep hole?"
* **Why it fails:** The monocular model sees the reflection of the tall building and will mathematically estimate a massive drop-off, failing to recognize it as a flat planar surface of water. 

---
### Running these in Colab
You can load any of these image scenarios via URL directly into the existing `SpaceQwen.py` script by replacing the `"image"` tag in the JSON payload, and submitting these exact prompts to log the hallucinated failure responses!
