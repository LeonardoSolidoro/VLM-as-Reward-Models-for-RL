import argparse
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct", help="Base model ID")
    parser.add_argument("--adapter-dir", required=True, help="Path to your trained LoRA adapter")
    parser.add_argument("--output-dir", required=True, help="Where to save the completely merged model")
    args = parser.parse_args()

    print(f"1. Loading processor...")
    processor = AutoProcessor.from_pretrained(args.model_id)

    print(f"2. Loading base model in 16-bit (BFloat16)...")
    # We must load in 16-bit to perform the math required for a stable merge
    base_model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    print(f"3. Injecting LoRA adapter from {args.adapter_dir}...")
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)

    print(f"4. Merging LoRA weights permanently into the base model...")
    model = model.merge_and_unload()

    print(f"5. Saving merged model to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    
    print("Done! You can now use this merged model as your base model.")

if __name__ == "__main__":
    main()
