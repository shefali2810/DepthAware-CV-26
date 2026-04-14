from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

if __name__ == "__main__":

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "remyxai/SpaceQwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("remyxai/SpaceQwen2.5-VL-3B-Instruct")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    # "image": "https://raw.githubusercontent.com/remyxai/VQASynth/refs/heads/main/assets/warehouse_sample_2.jpeg",
                    "image": "file://VQASynth_Dataset/images/Image_12.jpg",
                    # "image": "./street_view.jpg",
                },
                # {"type": "text", "text": "What is the height of the man in the red hat in feet?"},
                {"type": "text", "text": "Describe the image. How high is the car from the ground?"},
                # {"type": "test", "text": "What do you see on the left and on the right of the image? Is there enough space to walk between bus and truck?"},
                {"type": "test", "text": "What are the individual coordinates of the bus centre and truck centre, both of which are on the left side of image?"},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)
