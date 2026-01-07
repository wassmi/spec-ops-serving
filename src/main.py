import time
import logging
import threading
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from src.engine import SpeculativeEngine
from src.metrics import SessionMetrics

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("SpecOps-API")

# --- PROMETHEUS METRICS ---
REQUEST_COUNT = Counter("specops_requests_total", "Total generation requests")
TOKEN_COUNT = Counter("specops_tokens_total", "Total tokens generated")
LATENCY_HIST = Histogram("specops_latency_seconds", "Time spent generating text")
JUMP_GAUGE = Gauge("specops_avg_jump", "Average speculative jump per request")

# --- ENGINE STATE ---
# Initialized as None to allow the API to start while the model downloads
engine_instance = None


def load_engine_background():
    """Background task to download and initialize the engine."""
    global engine_instance
    try:
        logger.info("🤖 [BOOT] Starting background engine initialization...")
        engine_instance = SpeculativeEngine(
            tokenizer_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        )
        logger.info("✅ [BOOT] Engine is now ONLINE and ready for requests.")
    except Exception as e:
        logger.error(f"❌ [BOOT] Background initialization failed: {str(e)}")


# Start loading IMMEDIATELY in a separate thread
# This ensures port 8888 opens so GitHub Actions/Health Checks don't get 'Connection Refused'
threading.Thread(target=load_engine_background, daemon=True).start()

# --- API SETUP ---
app = FastAPI(title="Spec-Ops API")


class Query(BaseModel):
    prompt: str
    max_new_tokens: int = 15
    k_draft: int = Field(default=3, ge=1, le=5)


@app.get("/health")
async def health():
    """Endpoint for CI/CD and monitoring to check readiness."""
    return {
        "status": "online",
        "engine_ready": engine_instance is not None,
        "mode": "heuristic-speculative",
    }


@app.get("/metrics")
async def metrics():
    """Endpoint for Prometheus to scrape."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/generate")
async def generate(query: Query):
    if engine_instance is None:
        raise HTTPException(
            status_code=503,
            detail="Engine is still loading or downloading weights. Please wait.",
        )

    REQUEST_COUNT.inc()
    start_time = time.time()

    try:
        result, stats = engine_instance.generate(
            query.prompt, max_new_tokens=query.max_new_tokens, K=query.k_draft
        )

        # Update Prometheus Metrics
        duration = time.time() - start_time
        LATENCY_HIST.observe(duration)
        TOKEN_COUNT.inc(stats["total_tokens"])
        JUMP_GAUGE.set(stats["avg_tokens_per_jump"])

        return {
            "generated_text": result,
            "tokens_per_second": stats["tokens_per_second"],
            "avg_tokens_per_jump": stats["avg_tokens_per_jump"],
            "latency_ms": stats["latency_ms"],
        }
    except Exception as e:
        logger.error(f"❌ [RUNTIME] {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8888)
