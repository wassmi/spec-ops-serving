from engine import SpeculativeEngine
import time

def run():
    # Paths relative to the root of spec-ops-mlops
    engine = SpeculativeEngine(
        target_path="models/target/model_quantized.onnx",
        draft_path="models/draft/model_quantized.onnx",
        tokenizer_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )
    
    prompt = "The key to successful machine learning operations is"
    
    final_text = engine.generate(prompt)
    print(f"\n✨ FINAL OUTPUT: {final_text}")

if __name__ == "__main__":
    run()