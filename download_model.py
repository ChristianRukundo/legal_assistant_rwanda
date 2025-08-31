import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# This script performs a one-time download of the model and its configuration.
# It requires you to be logged in via `huggingface-cli login`.

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

def download_and_cache_model():
    """
    Downloads and caches the specified Hugging Face model and tokenizer.
    This function will connect to the internet and requires authentication.
    """
    print(f"--- Starting download for {MODEL_NAME} ---")
    
    # Define quantization config to match the application's runtime config
    # This isn't strictly necessary for the download, but ensures compatibility
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32,
    )

    try:
        # Download tokenizer - this will fetch all necessary tokenizer files
        print("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=True)
        print("Tokenizer downloaded and cached successfully.")

        # Download model - this will fetch the config and model weight files
        print("Downloading model (this may take a long time)...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=quantization_config,
            token=True # Use the authenticated token
        )
        print("Model downloaded and cached successfully.")
        
        print("\n--- Verification Complete ---")
        print("All necessary files for the model are now in your local cache.")
        print("You can now run the main application.")

    except Exception as e:
        print(f"\n--- An Error Occurred ---")
        print(f"Error: {e}")
        print("\nPlease ensure you have:")
        print("1. A stable internet connection.")
        print("2. You have accepted the license terms for the model on the Hugging Face website.")
        print("3. You are logged in correctly via 'huggingface-cli login'.")

if __name__ == "__main__":
    download_and_cache_model()