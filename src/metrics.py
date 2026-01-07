import time
from dataclasses import dataclass, field
from typing import List

@dataclass
class SessionMetrics:
    start_time: float = 0
    end_time: float = 0
    total_tokens: int = 0
    acceptance_records: List[int] = field(default_factory=list)

    def get_tps(self):
        duration = self.end_time - self.start_time
        return self.total_tokens / duration if duration > 0 else 0

    def get_avg_acceptance(self):
        if not self.acceptance_records:
            return 0
        return sum(self.acceptance_records) / len(self.acceptance_records)

    def report(self):
        return {
            "tokens_per_second": round(self.get_tps(), 2),
            "avg_tokens_per_jump": round(self.get_avg_acceptance(), 2),
            "total_tokens": self.total_tokens,
            "latency_ms": round((self.end_time - self.start_time) * 1000, 2)
        }