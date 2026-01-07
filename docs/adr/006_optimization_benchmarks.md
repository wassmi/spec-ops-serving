# Dev Log 006: Speculative Decoding Optimization & Observability

**Date:** January 7, 2026  
**Status:** Completed  
**Objective:** Resolve 0% speculative jump rate and implement real-time performance tracking.

## 1. Architectural Changes
### N-Gram Heuristic Speculator
* **Problem:** The previous "Repeat-Last-Token" heuristic resulted in a **0% acceptance rate**, as language models rarely repeat the exact same token sequentially.
* **Solution:** Implemented a **Predictive Lookback (N-Gram) Heuristic**. The engine now scans the previous context to identify repeating patterns (e.g., if "A B" was seen before, it speculates "B" will follow "A" again).
* **Result:** Increased average tokens per jump from **0.0 to 2.0**.

## 2. Observability Stack
* **Integration:** Connected the FastAPI engine to a **Prometheus** exporter.
* **Visualization:** Deployed **Grafana** with a custom dashboard to track:
    * `specops_avg_jump`: Efficiency of the speculation logic.
    * `specops_tokens_total`: Real-time throughput (Tokens/Sec).
* **Proof of Work:** Performance verified at **15.8 TPS** (a ~5x increase over the 3.2 TPS baseline).

## 3. System Hardening
* **Memory Management:** Configured `mmap` for ONNX weight streaming to prevent OOM (Out of Memory) crashes on 8GB RAM.
* **Storage Optimization:** Implemented Docker pruning and cache purging to maintain functionality within high-constraint environments (<1% disk space).

## 4. Final Performance Metrics
| Metric | Value |
| :--- | :--- |
| **Max Throughput** | 15.8 Tokens/Sec |
| **Avg Jump Rate** | 2.0 |
| **Latency** | 1.2s |
