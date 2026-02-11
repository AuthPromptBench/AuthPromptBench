import json
import os
from models.components.refiner.extractor import MainObjectExtractor
from itertools import batched
from PIL import Image
import torch
def add_main_object_to_dataset(input_file, output_file, model_name):
    extractor = MainObjectExtractor(model_name, "XCLIU/instaflow_0_9B_from_sd_1_5")
    extractor.to("cuda")
    print(f"Loaded MainObjectExtractor model to device {extractor.extra_model.device}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    data_dir = os.path.dirname(input_file)
    images_files = [os.path.join(data_dir, item['image_file']) for item in data]
    images = [Image.open(image_file).convert("RGB") for image_file in images_files]

    main_objects = []
    for batch in batched(images, 32):
        batch_main_objects = extractor(prompts=None, images=batch)
        main_objects.extend(batch_main_objects)

    for item, main_object in zip(data, main_objects):
        item['main_object'] = main_object

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


    print(f"Processed {len(data)} samples")
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Add instruction to dataset")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file path")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file path")
    parser.add_argument("--model", type=str, default="OpenGVLab/InternVL3-2B", help="Model name for MainObjectExtractor")
    args = parser.parse_args()

    # Set input and output file paths
    input_jsonl = args.input
    output_jsonl = args.output
    # Run add main object operation
    add_main_object_to_dataset(input_jsonl, output_jsonl, args.model)