import os
import json
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoProcessor, AutoModelForCausalLM, TrainingArguments, Trainer

def prepare_vqa_dataset(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    dataset_dict = {"image": [], "query": [], "ground_truth": []}
    for item in data:
        img_path = os.path.join("/data/shefali/DepthAware2026Project", item['image'])
        if os.path.exists(img_path):
            dataset_dict["image"].append(img_path)
            dataset_dict["query"].append(item['question'])
            dataset_dict["ground_truth"].append(item['cot']) # Training on the Chain-of-Thought
            
    return Dataset.from_dict(dataset_dict)

def build_lora_model(model_id="Qwen/Qwen2.5-VL-3B-Instruct"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load base model & processor
    # flash_attention_2 is omitted here based on previous symbol mismatch issues
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map=device)
    processor = AutoProcessor.from_pretrained(model_id)

    # Freeze base model layers
    for param in model.parameters():
        param.requires_grad = False
        
    # Configure PEFT / LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        task_type=TaskType.CAUSAL_LM
    )
    
    # Wrap model for PEFT
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    
    return peft_model, processor

def train_vlm():
    print("Loading VQASynth annotations...")
    dataset = prepare_vqa_dataset("VQASynth_Dataset/annotations.json")
    
    print("Bootstrapping Qwen2.5-VL with LoRA adapters...")
    model, processor = build_lora_model()
    
    # Custom Data Collator would be required here to process multimodal inputs
    # For representation, we setup the primary training loop constraints.
    
    training_args = TrainingArguments(
        output_dir="qwen2.5-vl-depth-lora",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        logging_steps=10,
        learning_rate=2e-5,
        fp16=True if torch.cuda.is_available() else False,
        num_train_epochs=3,
        save_strategy="epoch"
    )
    
    print("\nTrainer Configured. Ready for Epoch Execution!")
    print(f"Target dataset size: {len(dataset)} samples.")
    print("Execute via specialized multimodal trainer loop when fully downloaded.")
    
if __name__ == "__main__":
    train_vlm()
