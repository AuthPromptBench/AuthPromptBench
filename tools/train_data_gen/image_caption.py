import requests
import base64
import json
import os
from pathlib import Path
from typing import Optional, List
import argparse

class ImageCaptioner:
    def __init__(self, api_key: str, model: str = "google/gemini-2.0-flash-001"):
        """
        Initialize image caption generator
        
        Args:
            api_key: OpenRouter API key
            model: Model name to use
        """
        self.api_key = api_key
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        
    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 format"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def create_prompt(self) -> str:
        """Create specialized prompt for analyzing Stable Diffusion generated images"""
        return """You are a professional AI image analyst specializing in analyzing Stable Diffusion generated images. Please analyze this image and generate a prompt that could have been used to create this image.

Requirements:
1. The generated prompt should be concise, accurate, and suitable for CLIP model understanding
2. Use English with comma-separated keyword format
3. Include the following elements (if applicable):
   - Subject description (people, objects, scenes)
   - Art style (e.g., realistic, anime, oil painting, digital art, etc.)
   - Quality descriptors (e.g., highly detailed, 8k, masterpiece, etc.)
   - Composition description (e.g., portrait, full body, close-up, etc.)
   - Lighting effects (e.g., soft lighting, dramatic lighting, etc.)
   - Color characteristics (e.g., vibrant colors, monochrome, etc.)

4. Avoid overly complex descriptions, keep the prompt practical
5. Sort by importance, with the most important keywords first

Please output the prompt directly without additional explanations."""

    def caption_image(self, image_path: str) -> Optional[str]:
        """
        Caption a single image
        
        Args:
            image_path: Path to the image
            
        Returns:
            Generated caption, None if failed
        """
        try:
            # Encode image
            base64_image = self.encode_image(image_path)
            
            # Build request
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": self.create_prompt()
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 500,
                "temperature": 0.3  # Lower temperature for consistent output
            }
            
            # Send request
            response = requests.post(self.base_url, headers=headers, json=payload)
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            caption = result['choices'][0]['message']['content'].strip()
            
            return caption
            
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            return None
    
    def process_directory(self, input_dir: str, output_file: str = None, 
                         supported_formats: List[str] = None) -> List[dict]:
        """
        Batch process images in directory
        
        Args:
            input_dir: Input directory path
            output_file: Output file path (optional)
            supported_formats: List of supported image formats
            
        Returns:
            List of dictionaries containing filename and caption
        """
        if supported_formats is None:
            supported_formats = ['.jpg', '.jpeg', '.png', '.webp', '.bmp']
        
        input_path = Path(input_dir)
        results = []
        
        # Get all image files
        image_files = []
        for ext in supported_formats:
            image_files.extend(input_path.glob(f"*{ext}"))
            # image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        print(f"Found {len(image_files)} images")
        print(image_files)
        
        # Process each image
        for i, image_file in enumerate(image_files, 1):
            print(f"Processing {i}/{len(image_files)}: {image_file.name}")
            
            caption = self.caption_image(str(image_file))
            
            if caption:
                result = {
                    'filename': image_file.name,
                    'filepath': str(image_file),
                    'caption': caption
                }
                results.append(result)
                print(f"Generated caption: {caption}")
            else:
                print(f"Failed: {image_file.name}")
            
            print("-" * 50)
        
        # Save results to file
        if output_file and results:
            self.save_results(results, output_file)
        
        return results
    
    def save_results(self, results: List[dict], output_file: str):
        """Save results to file"""
        output_path = Path(output_file)
        
        # Save different formats based on file extension
        if output_path.suffix.lower() == '.json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        else:
            # Default to text format, one caption per line
            with open(output_path, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(f"{result['filename']}: {result['caption']}\n")
        
        print(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate captions for Stable Diffusion images using Gemini 2.0 Flash')
    parser.add_argument('--api-key', help='OpenRouter API key')
    parser.add_argument('--input', required=True, help='Input image path or directory')
    parser.add_argument('--output', help='Output file path (optional)')
    parser.add_argument('--model', default='google/gemini-2.0-flash-001', help='Model to use')
    
    args = parser.parse_args()
    
    # Create captioner instance
    captioner = ImageCaptioner(args.api_key, args.model)
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Process single file
        print(f"Processing single file: {input_path}")
        caption = captioner.caption_image(str(input_path))
        if caption:
            print(f"Caption: {caption}")
            
            # Save result if output file specified
            if args.output:
                result = [{
                    'filename': input_path.name,
                    'filepath': str(input_path),
                    'caption': caption
                }]
                captioner.save_results(result, args.output)
        else:
            print("Caption generation failed")
            
    elif input_path.is_dir():
        # Process directory
        print(f"Processing directory: {input_path}")
        results = captioner.process_directory(str(input_path), args.output)
        print(f"Successfully processed {len(results)} images")
        
    else:
        print(f"Error: Path does not exist - {input_path}")


if __name__ == "__main__":
    # Example usage
    print("Image Caption Generation Tool")
    print("=" * 50)
    
    # Show usage instructions if no command line arguments
    import sys
    if len(sys.argv) == 1:
        print("Usage:")
        print("python image_caption.py --api-key YOUR_API_KEY --input IMAGE_PATH")
        print("python image_caption.py --api-key YOUR_API_KEY --input IMAGE_DIR --output results.txt")
        print("\nArguments:")
        print("--api-key: OpenRouter API key")
        print("--input: Input image path or directory containing images")
        print("--output: Output file path (optional, supports .txt and .json formats)")
        print("--model: Model to use (default: google/gemini-2.0-flash-001)")
    else:
        main()