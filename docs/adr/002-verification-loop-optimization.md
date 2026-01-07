# ADR 002: Verification Loop Optimization Results

## Context
Following ADR 001, I identified that the 2.09 TPS bottleneck was due to redundant computations. I implemented a vectorized verification loop in the engine.

## Results (2026-01-07)
I performed a benchmark using the updated engine and observed:
- **Throughput:** 4.35 TPS (from 2.09 TPS)
- **Improvement:** ~108% increase
- **Acceptance Rate:** 0.50 (Avg tokens per jump)
- **CPU Utilization:** 15.8%

## Observations
The /health endpoint confirms resources are still healthy (0% Swap, 2.3GB RAM). However, the 15.8% CPU usage suggests the system is currently underutilized. The low acceptance rate (0.5) indicates the Draft model is frequently misaligning with the Target model.

## Next Steps
I will investigate increasing the "intra_op_num_threads" to utilize more CPU cores and look into model quantization or k-draft tuning to improve the acceptance rate.
