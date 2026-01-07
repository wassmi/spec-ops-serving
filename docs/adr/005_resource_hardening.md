# 📄 005 - Resource Hardening & Heuristic Speculation

**Status:** STABLE
**Problem:** Sequential loading of Phi-3 and TinyLlama exceeded 8GB RAM (Killed).
**Solution:** 1. Transitioned to Heuristic Speculation (Lookahead) to save 1.5GB RAM.
2. Disabled ONNX Memory Arena (enable_cpu_mem_arena=0) to stop allocation spikes.
3. Implemented global-scope model initialization to bypass lifespan timeouts.
