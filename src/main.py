import logging
import time
import traceback
import gc
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.engine import SpeculativeEngine

# 1. Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("SpecOps-API")

app = FastAPI(
    title="Spec-Ops Speculative Inference API",
    version="1.1.0"
)

# Global engine variable
engine = None

# --- Data Models for FastAPI ---
class Query(BaseModel):
    prompt: str
    max_new_tokens: int = 15
    temperature: float = 0.7
    k_draft: int = 3 # Added control for speculative window

class PredictionResponse(BaseModel):
    generated_text: str
    time_taken_ms: float
    tokens_per_second: float
    avg_tokens_per_jump: float # Key Metric: How well the draft model is doing
    status: str

# --- Endpoints ---

@app.get("/health")
async def health():
    """Verify the API, Model status, and simple RAM check."""
    # Note: In Step 3 we will add real RAM usage metrics here
    return {
        "status": "online",
        "engine_loaded": engine is not None,
        "environment": "Codespaces/WSL"
    }

@app.post("/generate", response_model=PredictionResponse)
async def generate(query: Query):
    global engine
    
    # 2. Lazy Loading Logic
    if engine is None:
        logger.info("🤖 [INIT] Loading ONNX Models into RAM (Target + Draft)...")
        try:
            start_load = time.time()
            # Paths adjusted for Docker WORKDIR /app
            engine = SpeculativeEngine(
                "/app/models/target/model_quantized.onnx",
                "/app/models/draft/model_quantized.onnx",
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            ) 
            logger.info(f"✅ [INIT] Models loaded in {time.time() - start_load:.2f}s")
        except Exception as e:
            logger.error(f"❌ [INIT] Failed: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail="Model engine failed to load.")

    # 3. Inference Logic with Benchmarking
    try:
        # Safety cap on tokens to prevent OOM in 8GB environment
        safe_limit = min(query.max_new_tokens, 50)
        
        # Execute speculative generation
        # Now returns (text, stats_dict) thanks to our engine.py update
        result_text, stats = engine.generate(
            query.prompt, 
            max_new_tokens=safe_limit, 
            K=query.k_draft
        )
        
        logger.info(f"📊 [METRICS] TPS: {stats['tokens_per_second']} | Jump Avg: {stats['avg_tokens_per_jump']}")

        return {
            "generated_text": result_text,
            "time_taken_ms": stats["latency_ms"],
            "tokens_per_second": stats["tokens_per_second"],
            "avg_tokens_per_jump": stats["avg_tokens_per_jump"],
            "status": "success"
        }

    except Exception as e:
        logger.error(f"❌ [RUNTIME] Inference failed:\n{traceback.format_exc()}")
        gc.collect() 
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Use reload=False in production/Docker to save resources
    uvicorn.run(app, host="0.0.0.0", port=8000)