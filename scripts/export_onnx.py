import os
from pathlib import Path
from huggingface_hub import hf_hub_download

# Configuration: Using verified public optimized models
MODELS = {
    "target": {
        "repo": "Xenova/TinyLlama-1.1B-Chat-v1.0",
        "file": "onnx/model_quantized.onnx"
    },
    "draft": {
        "repo": "onnx-community/Llama-160M-Chat-v1-ONNX",
        "file": "onnx/decoder_model_merged_int8.onnx"
    }
}

def setup_models():
    base_dir = Path("models")
    
    for folder, info in MODELS.items():
        target_dir = base_dir / folder
        target_dir.mkdir(parents=True, exist_ok=True)
        final_path = target_dir / "model_quantized.onnx"

        if final_path.exists():
            print(f"✅ {folder.capitalize()} model already exists at {final_path}")
            continue

        print(f"📡 Downloading {folder} model from {info['repo']}...")
        try:
            # Download specific file to temporary location
            downloaded_path = hf_hub_download(
                repo_id=info['repo'],
                filename=info['file'],
                local_dir=target_dir
            )
            # Rename to our standardized name: model_quantized.onnx
            os.rename(downloaded_path, final_path)
            print(f"✨ Successfully prepared {folder} model.")
        except Exception as e:
            print(f"❌ Error downloading {folder}: {e}")

if __name__ == "__main__":
    setup_models()