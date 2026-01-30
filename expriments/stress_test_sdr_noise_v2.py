import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from difflib import SequenceMatcher
import random
import os

# USING v2 
from core.modeling_sdr_qwen_full_v2 import SDRQwenForCausalLM
from core.qwen_sdr_tokeriser_v4 import SDRQwenTokenizer

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
LOCAL_MODEL_PATH = "./sdr_qwen_phase3_math"
BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

# Noise Levels: 0%, 10%, 20%, 30%, 40%, 50%
NOISE_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
MAX_NEW_TOKENS = 60

# We use Questions now, since it's a Chat model
TEST_PROMPTS = [
    "What is the capital of France?",
    "List three healthy foods.",
    "Write a python function to add two numbers.",
    "Tell me a very short story about a cat."
]


# -----------------------------------------------------------------------------
# NOISE INJECTION ENGINE
# -----------------------------------------------------------------------------
def inject_sdr_noise(sdr_tensor, noise_level, sdr_n=2048):
    if noise_level <= 0.0:
        return sdr_tensor

    noisy_sdr = sdr_tensor.clone()
    B, S, N = noisy_sdr.shape

    for b in range(B):
        for s in range(S):
            active_indices = (noisy_sdr[b, s] > 0).nonzero(as_tuple=False).view(-1)
            num_active = len(active_indices)
            num_to_swap = int(num_active * noise_level)

            if num_to_swap > 0:
                # 1. DELETE SIGNAL (1 -> 0)
                perm = torch.randperm(num_active)
                indices_to_drop = active_indices[perm[:num_to_swap]]
                noisy_sdr[b, s, indices_to_drop] = 0.0

                # 2. ADD NOISE (0 -> 1)
                added_count = 0
                while added_count < num_to_swap:
                    rand_idx = random.randint(0, N - 1)
                    if noisy_sdr[b, s, rand_idx] == 0:
                        noisy_sdr[b, s, rand_idx] = 1.0
                        added_count += 1

    return noisy_sdr


# -----------------------------------------------------------------------------
# MAIN TEST LOOP
# -----------------------------------------------------------------------------
def run_stress_test():
    print("=" * 60)
    print("SDR CHAT ROBUSTNESS STRESS TEST (v2)")
    print("=" * 60)

    print(f"Loading Model from {LOCAL_MODEL_PATH}...")
    model = SDRQwenForCausalLM.from_pretrained(LOCAL_MODEL_PATH, torch_dtype=DTYPE).to(DEVICE)
    model.eval()

    raw_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    sdr_tokenizer = SDRQwenTokenizer(pretrained_model_name_or_path=BASE_MODEL_NAME, sdr_n=2048, sdr_w=41)

    results_log = []
    system_prompt = "You are a helpful SDR-based AI assistant."

    for prompt in TEST_PROMPTS:
        print(f"\nUser Question: '{prompt}'")
        print("-" * 60)

        baseline_output = ""

        for noise in NOISE_LEVELS:
            # 1. APPLY CHAT TEMPLATE
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            text_input = raw_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            # 2. Encode
            input_ids = raw_tokenizer.encode(text_input)
            current_ids = torch.tensor(input_ids).unsqueeze(0).to(DEVICE)

            # 3. Convert to SDR
            clean_sdr = sdr_tokenizer._convert_ids_to_sdr(current_ids).to(DEVICE)

            # 4. INJECT NOISE (This now noises the System Prompt AND User Prompt)
            noisy_sdr = inject_sdr_noise(clean_sdr, noise, sdr_n=2048)

            # 5. Generate
            generated_ids = []
            curr_sdr_input = noisy_sdr.clone()

            for _ in range(MAX_NEW_TOKENS):
                with torch.no_grad():
                    with torch.autocast(device_type="cuda" if "cuda" in DEVICE else "cpu", dtype=DTYPE):
                        outputs = model(input_ids=curr_sdr_input, return_dict=True)

                    next_token_logits = outputs.logits[:, -1, :]
                    next_token_id = torch.argmax(next_token_logits, dim=-1)

                    if next_token_id.item() in [raw_tokenizer.eos_token_id, raw_tokenizer.pad_token_id]:
                        break

                    generated_ids.append(next_token_id.item())

                    # Feed back CLEAN history (Model internal state remains stable)
                    next_sdr = sdr_tokenizer._convert_ids_to_sdr(next_token_id.view(-1)).to(DEVICE)
                    next_sdr = next_sdr.unsqueeze(1)
                    curr_sdr_input = torch.cat([curr_sdr_input, next_sdr], dim=1)

            output_text = raw_tokenizer.decode(generated_ids)

            # 6. Metrics
            if noise == 0.0:
                baseline_output = output_text
                similarity = 100.0
            else:
                similarity = SequenceMatcher(None, baseline_output, output_text).ratio() * 100

            status = "✅ PASS" if similarity > 75 else "⚠️ DEGRADED" if similarity > 40 else "❌ FAIL"
            print(f"Noise {int(noise * 100)}% | Sim: {similarity:.1f}% | {status}")
            print(f"   Answer: {output_text.strip()[:80]}...")

            results_log.append(f"Q: {prompt} | Noise: {noise} | Sim: {similarity:.2f} | A: {output_text}")

    with open("sdr_chat_noise_test.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(results_log))
    print(f"\nDetailed logs saved to sdr_chat_noise_test.txt")


if __name__ == "__main__":
    run_stress_test()
