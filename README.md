# Spec-Ops: Speculative Decoding API 🚀

A high-performance LLM inference engine using ONNX Runtime and Speculative Decoding to achieve **15.8 TPS** on standard CPU environments.

### 🚀 Optimization Results
| Metric | Baseline (Vanilla) | Optimized (Spec-Ops) | Improvement |
| :--- | :--- | :--- | :--- |
| **Throughput** | ~3.2 Tokens/Sec | **15.8 Tokens/Sec** | **~4.9x Speedup** |
| **Latency** | ~6.2s | **1.2s** | **80% Reduction** |
| **Efficiency** | 0% Jump Rate | **~2.0 Avg Jump** | **Logic Success** |

### 🏗️ Technical Environment
* **Platform:** GitHub Codespaces (WSL/Ubuntu)
* **Compute:** 4-Core CPU / 8GB RAM
* **Runtime:** ONNX Runtime (`CPUExecutionProvider`)
* **Model:** Phi-3-mini (4-bit Quantized)
* **Verification:** Verified January 2026 via Prometheus/Grafana telemetry

### 📈 Monitoring Dashboard
![Performance Dashboard](/workspaces/spec-ops-mlops/docs/assets/Grafana.JPG)
*The graph above visualizes the throughput surge from 3.2 TPS to 15.8 TPS immediately following the activation of the N-Gram Heuristic Speculator.*

---

# 🗺️ Project Roadmap

## ✅ Phase 1: The Core Engine
- [x] Implement `SpeculativeEngine` in Python.
- [x] Integrate ONNX Runtime for CPU-bound inference.
- [x] Implement Draft-Target verification logic.

## ✅ Phase 2: Optimization & Observability
- [x] 4-bit quantization of Phi-3 models.
- [x] Dockerization of the API (FastAPI).
- [x] **Observability:** Docker Compose integration for real-time Prometheus/Grafana monitoring.

## ✅ Phase 3: Automated Quality Gate (CI/CD)
- [x] **Linting:** Automated code style enforcement (Black/Flake8).
- [x] **Security:** Static analysis for vulnerabilities (Bandit).
- [x] **Resource Mgmt:** GitHub Action disk optimization (reclaimed 10GB).

## ✅ Phase 4: Systems Hardening & Logic (COMPLETED)
- [x] **Heuristic Pivot:** Implemented Predictive N-Gram lookback to maximize speculative acceptance.
- [x] **Runtime Optimization:** Disabled ONNX Arena and utilized `mmap` for zero-crash weight streaming.
- [x] **Lifecycle Management:** Hardened server startup to bypass Cloud/Codespace watchdog timeouts.
- [x] **Weight Decoupling:** Successfully transitioned to Hugging Face Model Hub registry.

## 🏗️ Phase 5: System Maturity (NEXT)
- [ ] **Orchestration:** Kubernetes `deployment.yaml` with specific resource requests/limits.
- [ ] **Concurrency:** Stress testing with asynchronous request queuing.
- [ ] **Cache Persistence:** Redis-backed N-Gram storage for cross-session speculation.