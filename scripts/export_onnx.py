import os
from pathlib import Path
from optimum.onnxruntime import ORTModelForCausalLM, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoTokenizer, AutoConfig

# Optimized for your project structure
MODELS = [
    ("target", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
    ("draft", "jackaduma/Llama-160M"),
]

def export_and_quantize(folder_name, model_id):
    model_dir = Path(f"models/{folder_name}")
    model_dir.mkdir(parents=True, exist_ok=True)
    quantized_path = model_dir / "model_quantized.onnx"

    if quantized_path.exists():
        print(f"[SKIP] {model_id} already exists.")
        return

    print(f"[1/3] Downloading and Exporting {model_id} to ONNX...")
    # This downloads and converts to ONNX in one step
    model = ORTModelForCausalLM.from_pretrained(
        model_id, 
        export=True, 
        use_cache=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Save the base ONNX model temporarily to quantize it
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    print(f"[2/3] Quantizing {model_id} (INT8/AVX2)...")
    quantizer = ORTQuantizer.from_pretrained(model_dir)
    dqconfig = AutoQuantizationConfig.avx2_dynamic()
    
    # Apply quantization
    quantizer.quantize(
        save_dir=model_dir,
        quantization_config=dqconfig,
    )

    # Clean up: Rename the file to our standard 'model_quantized.onnx' if needed
    # ORT usually saves it as 'model_quantized.onnx' by default
    print(f"[3/3] SUCCESS: {model_id} is ready in {model_dir}")

if __name__ == "__main__":
    for folder, m_id in MODELS:
        export_and_quantize(folder, m_id)