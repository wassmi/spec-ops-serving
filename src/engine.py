import time
import onnxruntime as ort
import numpy as np
import gc
from transformers import AutoTokenizer
from src.metrics import SessionMetrics


class SpeculativeEngine:
    def __init__(self, target_path, draft_path, tokenizer_id):
        # I am using the tokenizer and setting up hardened session options
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, revision="fe8a4ea1ffedaf415f4da2f062534de366a451e6")

        # HARDENED SESSION OPTIONS (Optimized for your 4GB/8GB environment)
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 2
        sess_options.inter_op_num_threads = 1
        sess_options.enable_mem_pattern = False

        # Release memory back to OS aggressively
        sess_options.add_session_config_entry(
            "session.use_device_allocator_for_initialization", "1"
        )

        # Initialize sessions with CPU provider
        self.target_sess = ort.InferenceSession(
            target_path, sess_options, providers=["CPUExecutionProvider"]
        )
        self.draft_sess = ort.InferenceSession(
            draft_path, sess_options, providers=["CPUExecutionProvider"]
        )

        # Auto-detect architecture (Counting KV layers and heads)
        self.target_layers = (
            sum(1 for x in self.target_sess.get_inputs() if "past_key_values" in x.name)
            // 2
        )
        self.draft_layers = (
            sum(1 for x in self.draft_sess.get_inputs() if "past_key_values" in x.name)
            // 2
        )

        t_kv = next(
            x
            for x in self.target_sess.get_inputs()
            if "past_key_values.0.key" in x.name
        )
        self.target_heads = t_kv.shape[1]

        d_kv = next(
            x for x in self.draft_sess.get_inputs() if "past_key_values.0.key" in x.name
        )
        self.draft_heads = d_kv.shape[1]

        print(
            f"✅ Engine Ready | Target: {self.target_layers}L/{self.target_heads}H | Draft: {self.draft_layers}L/{self.draft_heads}H"
        )

    def _get_logits(self, session, input_ids, num_layers, num_heads):
        """Helper to run inference while providing required zero-filled KV caches."""
        attention_mask = np.ones(input_ids.shape, dtype=np.int64)
        model_inputs = [x.name for x in session.get_inputs()]

        input_feed = {
            "input_ids": input_ids.astype(np.int64),
            "attention_mask": attention_mask,
        }

        if "position_ids" in model_inputs:
            input_feed["position_ids"] = (
                np.arange(input_ids.shape[1]).reshape(1, -1).astype(np.int64)
            )
        if "use_cache_branch" in model_inputs:
            input_feed["use_cache_branch"] = np.array([False], dtype=bool)

        # I am passing empty KV caches (zeros) because this implementation
        # uses the non-caching branch of the ONNX model for stability.
        for i in range(num_layers):
            input_feed[f"past_key_values.{i}.key"] = np.zeros(
                (1, num_heads, 0, 64), dtype=np.float32
            )
            input_feed[f"past_key_values.{i}.value"] = np.zeros(
                (1, num_heads, 0, 64), dtype=np.float32
            )

        return session.run(None, input_feed)[0]

    def generate(self, prompt, max_new_tokens=15, K=1):
        metrics = SessionMetrics()
        input_ids = self.tokenizer.encode(prompt, return_tensors="np")
        metrics.start_time = time.time()
        initial_len = input_ids.shape[1]

        try:
            while (input_ids.shape[1] - initial_len) < max_new_tokens:
                prefix_len = input_ids.shape[1]
                draft_ids = input_ids.copy()

                # 1. Draft Proposal (Small model guesses K tokens)
                for _ in range(K):
                    logits = self._get_logits(
                        self.draft_sess, draft_ids, self.draft_layers, self.draft_heads
                    )
                    next_tok = np.argmax(logits[:, -1, :], axis=-1).reshape(1, 1)
                    draft_ids = np.concatenate([draft_ids, next_tok], axis=-1)

                # 2. Target Verification (Big model checks all K at once)
                target_logits = self._get_logits(
                    self.target_sess, draft_ids, self.target_layers, self.target_heads
                )

                # I am pulling the predictions for all proposed positions
                target_preds = np.argmax(
                    target_logits[0, prefix_len - 1 : -1, :], axis=-1
                )
                draft_tokens = draft_ids[0, prefix_len:]

                # 3. Comparison Logic
                n_matches = 0
                for i in range(len(draft_tokens)):
                    if draft_tokens[i] == target_preds[i]:
                        n_matches += 1
                    else:
                        break

                metrics.acceptance_records.append(n_matches)
                print(f"DEBUG: Draft proposed {K}, Target accepted {n_matches}")

                # 4. Update Sequence
                # We take the accepted tokens plus the one correction token from the target
                accepted = target_preds[: n_matches + 1].reshape(1, -1)
                input_ids = np.concatenate([input_ids, accepted], axis=-1)

                if self.tokenizer.eos_token_id in accepted:
                    break

            metrics.end_time = time.time()
            metrics.total_tokens = input_ids.shape[1] - initial_len

            final_output = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
            return final_output, metrics.report()

        finally:
            # I am ensuring memory is cleared after inference
            gc.collect()
