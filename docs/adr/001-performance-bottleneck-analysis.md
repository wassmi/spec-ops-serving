# ADR 001: Performance Bottleneck Analysis

## Context
I benchmarked the initial speculative inference engine and observed a throughput of **2.09 TPS** and an acceptance rate of **0.15**. My goal is to determine if this low performance is due to hardware limits or code inefficiency.

## Investigation
On 2026-01-07, I utilized the `/health` observability endpoint I built to check for resource constraints.
- **Process RAM:** 2328.14 MB
- **System Swap:** 0.0%
- **CPU Load:** 10.7%

## Decision
Because the swap is at 0%, I have confirmed that the bottleneck is NOT memory or disk paging. I have identified the root cause as the **Engine Logic**. Specifically, the lack of KV-Caching causes redundant computations.

## Status
**Accepted**. I am moving to implement KV-Caching in Phase 4 to optimize throughput.
