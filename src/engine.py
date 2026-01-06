import onnxruntime as ort
import numpy as np
import time
from transformers import AutoTokenizer

class SpeculativeEngine:
    def __init__(self, target_path, draft_path, tokenizer_id):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 2 
        
        self.target_sess = ort.InferenceSession(target_path, sess_options)
        self.draft_sess = ort.InferenceSession(draft_path, sess_options)

    def _get_logits(self, session, input_ids, num_layers, num_heads):
        attention_mask = np.ones(input_ids.shape, dtype=np.int64)
        position_ids = np.arange(input_ids.shape[1]).reshape(1, -1).astype(np.int64)
        
        input_feed = {
            "input_ids": input_ids.astype(np.int64),
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        for i in range(num_layers):
            input_feed[f"past_key_values.{i}.key"] = np.zeros((1, num_heads, 0, 64), dtype=np.float32)
            input_feed[f"past_key_values.{i}.value"] = np.zeros((1, num_heads, 0, 64), dtype=np.float32)
        
        return session.run(None, input_feed)[0]

    def generate(self, prompt, max_new_tokens=20, K=3):
        input_ids = self.tokenizer.encode(prompt, return_tensors="np")
        print(f"\n🚀 Speculative Engine Running...")
        
        while input_ids.shape[1] < 50: # Limit total length for test
            # 1. Draft model suggests K tokens
            prefix_len = input_ids.shape[1]
            draft_ids = input_ids.copy()
            for _ in range(K):
                logits = self._get_logits(self.draft_sess, draft_ids, 12, 12)
                next_id = np.argmax(logits[:, -1, :], axis=-1)
                draft_ids = np.concatenate([draft_ids, next_id.reshape(1, 1)], axis=-1)

            # 2. Target model checks ALL tokens (prefix + K suggestions)
            target_logits = self._get_logits(self.target_sess, draft_ids, 22, 4)
            # Get the most likely tokens for the entire sequence
            target_predictions = np.argmax(target_logits[0, prefix_len-1:-1, :], axis=-1)

            # 3. Compare and Accept
            # We check how many of the draft's K tokens match the target's predictions
            match_found = False
            for i in range(K):
                if draft_ids[0, prefix_len + i] == target_predictions[i]:
                    # Draft was right! Add this token
                    input_ids = np.concatenate([input_ids, draft_ids[0, prefix_len + i].reshape(1,1)], axis=-1)
                else:
                    # Draft was wrong! Add the Target's correction and stop
                    input_ids = np.concatenate([input_ids, target_predictions[i].reshape(1,1)], axis=-1)
                    match_found = True
                    break
            
            # If they all matched perfectly, we still add the Target's NEXT token prediction
            if not match_found:
                 last_target_token = np.argmax(target_logits[0, -1, :])
                 input_ids = np.concatenate([input_ids, last_target_token.reshape(1,1)], axis=-1)

            print(self.tokenizer.decode(input_ids[0], skip_special_tokens=True))
            print("-" * 20) # Divider to see the "jumps"

if __name__ == "__main__":
    engine = SpeculativeEngine("models/target/model_quantized.onnx", "models/draft/model_quantized.onnx", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    engine.generate("The best part of being a developer is")