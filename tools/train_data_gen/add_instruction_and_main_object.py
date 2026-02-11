import json

def add_instruction_to_dataset(input_file, output_file, instruction_text="Improve the prompt for image generation", main_object_dict=None):
    """
    Add 'instruction' and 'main_object' fields to each sample in the dataset.

    Args:
        input_file: Input JSON file path
        output_file: Output JSON file path
        instruction_text: Instruction text to add
    """

    # Read original data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Add instruction and main_object fields to each sample
    for sample in data:
        sample['instruction'] = instruction_text
        if main_object_dict:
            sample['main_object'] = main_object_dict.get(sample['image_file'])

    # Save to new file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"Processed {len(data)} samples")
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Add instruction to dataset")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file path")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file path")
    parser.add_argument("--instruction", type=str, default="Convert to short description for text-to-image generation", help="Instruction text to add")
    parser.add_argument("--main_object_files", type=str, default="data/diffusionDB/part-000001/train_prompt_caption_mainobj_v0-000001.json", help="Path to main object files")
    args = parser.parse_args()

    # Set input and output file paths
    input_jsonl = args.input
    output_jsonl = args.output
    instruction_text = args.instruction
    main_object_files = args.main_object_files

    main_object_dict = {}
    with open(main_object_files, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for item in data:
            main_object_dict[item['image_file']] = item['main_object']
    # Run add instruction operation
    add_instruction_to_dataset(input_jsonl, output_jsonl, instruction_text, main_object_dict)