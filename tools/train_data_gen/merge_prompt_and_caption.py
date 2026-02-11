import json
import os

def merge_prompt_and_caption(filter_file_path, caption_file_path, output_file_path):
    """
    Merge human prompt from filter file and caption from caption file.

    Args:
        filter_file_path: Path to filter-010001.json file
        caption_file_path: Path to caption-010001-100.json file
        output_file_path: Path to output file
    """

    # Read filter file
    with open(filter_file_path, 'r', encoding='utf-8') as f:
        filter_data = json.load(f)

    # Read caption file
    with open(caption_file_path, 'r', encoding='utf-8') as f:
        caption_data = json.load(f)

    # Create caption mapping dictionary, using filename as key
    caption_dict = {item['filename']: item['caption'] for item in caption_data}

    # Merge data
    merged_data = []
    for k, v in filter_data.items():
        filename = k
        human_prompt = v.get('p', "")
        caption = caption_dict.get(filename, "")

        merged_item = {
            "input": human_prompt,
            "output": caption,
            "image_file": filename
        }
        merged_data.append(merged_item)

    # Save merged data
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)

    print(f"Merging completed! Processed {len(merged_data)} items.")
    print(f"Output file: {output_file_path}")

# Example usage
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Merge prompt and caption files")
    parser.add_argument("--filter_file", type=str, required=True, help="Path to the filter file")
    parser.add_argument("--caption_file", type=str, required=True, help="Path to the caption file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output file")
    args = parser.parse_args()

    # Run merging
    merge_prompt_and_caption(args.filter_file, args.caption_file, args.output_file)