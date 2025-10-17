# how to test

# fix_quantization.py - Use transformers' built-in quantization support
import torch
from transformers import AutoProcessor, AutoModelForCTC, BitsAndBytesConfig
import os

def properly_save_quantized_model():
    print("Loading original model...")
    processor = AutoProcessor.from_pretrained("CLEAR-Global/w2v-bert-2.0-yoruba_naijavoices_250h")
    
    # Use 8-bit quantization via BitsAndBytes
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=True
    )
    
    try:
        model = AutoModelForCTC.from_pretrained(
            "CLEAR-Global/w2v-bert-2.0-yoruba_naijavoices_250h",
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
    except:
        # Fallback to regular loading
        print("8-bit quantization failed, loading regular model...")
        model = AutoModelForCTC.from_pretrained("CLEAR-Global/w2v-bert-2.0-yoruba_naijavoices_250h")
    
    # Save directory
    save_dir = "./quantized_asr_model"
    os.makedirs(save_dir, exist_ok=True)
    
    print("Saving model...")
    # Save processor and model
    processor.save_pretrained(save_dir)
    model.save_pretrained(save_dir)
    
    print("Model saved successfully!")
    
    # Verify the files
    print("\nSaved files:")
    for file in os.listdir(save_dir):
        print(f"  - {file}")

if __name__ == "__main__":
    properly_save_quantized_model()