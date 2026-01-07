# Spec-Ops: Speculative Decoding API 🚀

A high-performance LLM inference engine using ONNX Runtime and Speculative Decoding to achieve **8.19 TPS** on standard CPU environments.



## 📊 Performance Benchmark
- **Environment:** GitHub Actions Runner (Standard Ubuntu)
- **Engine:** ONNX Runtime (CPU)
- **Optimization:** 4-bit Quantization
- **Result:** **8.19 Tokens Per Second** (Verified January 2026)

---

## 🗺️ Project Roadmap

### ✅ Phase 1: The Core Engine
- [x] Implement `SpeculativeEngine` in Python.
- [x] Integrate ONNX Runtime for CPU-bound inference.
- [x] Implement Draft-Target verification logic.

### ✅ Phase 2: Optimization & Containerization
- [x] 4-bit quantization of Phi-3 models.
- [x] Dockerization of the API (FastAPI).
- [x] Local performance profiling (Cursor/WSL).

### ✅ Phase 3: Automated Quality Gate (CI/CD)
- [x] **Linting:** Automated code style enforcement (Black).
- [x] **Security:** Static analysis for vulnerabilities (Bandit).
- [x] **Smoke Test:** Automated performance benchmarking in the cloud.
- [x] **Resource Mgmt:** GitHub Action disk optimization (reclaimed 10GB).

### 🏗️ Phase 4: Model Registry & Decoupling (Current)
- [ ] Move models from Git LFS to **Hugging Face Model Hub**.
- [ ] Decouple binary weights from source code.
- [ ] Implement dynamic model pulling on container startup.

### 🔭 Phase 5: Production & Scaling (Next)
- [ ] Implement Docker Compose for multi-service monitoring.
- [ ] Kubernetes `deployment.yaml` for container orchestration.
- [ ] Stress testing and concurrency profiling.

---

## 🛠️ Infrastructure Stack
- **Runtime:** Python 3.11, ONNX Runtime
- **CI/CD:** GitHub Actions
- **Storage:** Git LFS (Migrating to Hugging Face)
- **Environment:** Docker (WSL / GitHub Runners)

## 🚦 Getting Started
1. **Clone:** `git clone ...`
2. **Setup:** `pip install -r requirements.txt`
3. **Run CI Locally:** `docker build -t spec-ops-api .`