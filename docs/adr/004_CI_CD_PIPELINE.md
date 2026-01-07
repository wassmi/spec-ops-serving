# 004: CI/CD Pipeline & Performance Gates

## Overview
This document outlines the automated testing and deployment strategy for the Speculative Decoding engine.



## Pipeline Stages
1. **Quality Gate:** Static analysis via `black` (linting) and `bandit` (security).
2. **Resource Preparation:** Clears 10GB of unused data from the GitHub runner to accommodate large ONNX models.
3. **Containerization:** Builds a production-ready Docker image.
4. **Smoke Test:** Deploys the container and runs a live benchmark against the `/generate` endpoint.

## Key Technical Hurdles
- **Disk Management:** We bypassed the 14GB GitHub Runner limit by removing pre-installed SDKs (/usr/share/dotnet, etc.).
- **LFS Integration:** Fixed model corruption by ensuring `lfs: true` was set during the checkout action.
- **Latency Verification:** Established a 2.0 TPS baseline; current version achieves ~8.19 TPS.

## Maintenance
To modify the performance threshold, update the `TPS < 2.0` logic in `.github/workflows/main.yml`.
