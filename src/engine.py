import time
import os
import onnxruntime as ort
import numpy as np
import gc
from transformers import AutoTokenizer
from src.metrics import SessionMetrics


class SpeculativeEngine:
    def __init__(self, tokenizer_id, repo_id="wassmi/spec-ops-phi3-onnx"):
        
        self.target_path = "models/target/model_quantized.onnx"
        if not os.path.exists(self.target_path):
            print(f"📥 Model not found at {self.target_path}. Pulling from Registry...")
            os.makedirs(os.path.dirname(self.target_path), exist_ok=True)
            hf_hub_download(
                repo_id=repo_id,
                filename="model_quantized.onnx",
                local_dir="models/target",
                revision="main" # or your specific branch
            )
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id, revision="fe8a4ea1ffedaf415f4da2f062534de366a451e6"
        )

        # --- THE ULTIMATE STABILITY OPTIONS ---
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1

        # Disable the Memory Arena (This stops the 'Killed' spike)
        sess_options.add_session_config_entry("session.enable_cpu_mem_arena", "0")

        # Use Memory Mapping for the weights
        sess_options.add_session_config_entry("session.use_mmap_prefix", "1")

        # Low-level graph optimization only (prevents memory spikes during startup)
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        )

        print(f"🚀 [FINAL FIX] Loading Target Model (Arena Disabled)...")
        self.target_sess = ort.InferenceSession(
            self.target_path, sess_options, providers=["CPUExecutionProvider"]
        )

        # Detect Architecture
        self.target_layers = (
            sum(1 for x in self.target_sess.get_inputs() if "past_key_values" in x.name)
            // 2
        )
        t_kv = next(
            x
            for x in self.target_sess.get_inputs()
            if "past_key_values.0.key" in x.name
        )
        self.target_heads = t_kv.shape[1]

        print(f"✅ Engine Ready | Arena: Disabled | Stability: Max")

    def _get_target_logits(self, input_ids):
        attention_mask = np.ones(input_ids.shape, dtype=np.int64)
        input_feed = {
            "input_ids": input_ids.astype(np.int64),
            "attention_mask": attention_mask,
            "position_ids": np.arange(input_ids.shape[1])
            .reshape(1, -1)
            .astype(np.int64),
        }

        # Initialize KV-Caches with tiny allocations
        for i in range(self.target_layers):
            input_feed[f"past_key_values.{i}.key"] = np.zeros(
                (1, self.target_heads, 0, 64), dtype=np.float32
            )
            input_feed[f"past_key_values.{i}.value"] = np.zeros(
                (1, self.target_heads, 0, 64), dtype=np.float32
            )

        return self.target_sess.run(None, input_feed)[0]

    def generate(self, prompt, max_new_tokens=15, K=3):
        metrics = SessionMetrics()
        input_ids = self.tokenizer.encode(prompt, return_tensors="np")
        metrics.start_time = time.time()
        initial_len = input_ids.shape[1]

        while (input_ids.shape[1] - initial_len) < max_new_tokens:
            prefix_len = input_ids.shape[1]

            # Heuristic speculation (K tokens)
            draft_ids = input_ids.copy()
            proposal = np.repeat(input_ids[:, -1:], K, axis=1)
            draft_ids = np.concatenate([draft_ids, proposal], axis=-1)

            # Verification
            target_logits = self._get_target_logits(draft_ids)
            target_preds = np.argmax(target_logits[0, prefix_len - 1 : -1, :], axis=-1)
            draft_tokens = draft_ids[0, prefix_len:]

            n_matches = 0
            for i in range(len(draft_tokens)):
                if draft_tokens[i] == target_preds[i]:
                    n_matches += 1
                else:
                    break

            metrics.acceptance_records.append(n_matches)
            accepted = target_preds[: n_matches + 1].reshape(1, -1)
            input_ids = np.concatenate([input_ids, accepted], axis=-1)

            if self.tokenizer.eos_token_id in accepted:
                break

        metrics.end_time = time.time()
        metrics.total_tokens = input_ids.shape[1] - initial_len
        return (
            self.tokenizer.decode(input_ids[0], skip_special_tokens=True),
            metrics.report(),
        )
