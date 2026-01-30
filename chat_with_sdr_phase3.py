import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from core.modeling_sdr_qwen_full_v2 import SDRQwenForCausalLM
from core.qwen_sdr_tokeriser_v4 import SDRQwenTokenizer

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
# POINTS TO THE NEW PHASE 3 MODEL
LOCAL_MODEL_PATH = "./sdr_qwen_phase3_math"
BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

# GENERATION SETTINGS
TEMP = 0.7
TOP_P = 0.9
TOP_K = 50
REP_PENALTY = 1.1  # Lowered slightly for chat (1.2 can be too strict for coding)
MAX_NEW_TOKENS = 250  # Increased for longer answers


def apply_repetition_penalty(logits, sequence, penalty):
    if penalty == 1.0:
        return logits
    score = torch.gather(logits, 1, sequence)
    score = torch.where(score < 0, score * penalty, score / penalty)
    logits.scatter_(1, sequence, score)
    return logits


def sample_next_token(logits, input_ids, temperature, top_k, top_p, penalty):
    # 1. Apply Repetition Penalty
    logits = apply_repetition_penalty(logits, input_ids, penalty)

    # 2. Apply Temperature
    logits = logits / temperature

    # 3. Filter Top-K
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = -float('Inf')

    # 4. Filter Top-P (Nucleus)
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = -float('Inf')

    # 5. Sample
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token


def chat():
    print(f"Loading Phase 2 SDR Model from {LOCAL_MODEL_PATH}...")

    # 1. Load Tokenizers
    print(f"Loading Tokenizers from {BASE_MODEL_NAME}...")
    raw_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    sdr_tokenizer = SDRQwenTokenizer(
        pretrained_model_name_or_path=BASE_MODEL_NAME,
        sdr_n=2048,
        sdr_w=41
    )

    # 2. Load Model
    model = SDRQwenForCausalLM.from_pretrained(
        LOCAL_MODEL_PATH,
        torch_dtype=DTYPE,
        use_safetensors=True
    ).to(DEVICE)
    model.eval()

    print("\n" + "=" * 50)
    print("PHASE 2 SDR BRAIN ONLINE.")
    print(f"Temp: {TEMP} | Rep_Penalty: {REP_PENALTY}")
    print("Type 'quit' to exit.")
    print("=" * 50)

    # Standard Assistant Prompt
    system_prompt = "You are a helpful SDR-based AI assistant."

    while True:
        try:
            user_input = input("\nUser: ")
            if user_input.lower() in ["quit", "exit"]: break

            # Prepare Prompt
            messages = [{"role": "system", "content": system_prompt}]
            messages.append({"role": "user", "content": user_input})
            text_prompt = raw_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            # --- PREPARE INPUTS ---
            input_ids = raw_tokenizer.encode(text_prompt)
            current_ids = torch.tensor(input_ids).unsqueeze(0).to(DEVICE)  # [1, Seq]
            current_sdr = sdr_tokenizer._convert_ids_to_sdr(current_ids).to(DEVICE)  # [1, Seq, 41]

            print("Assistant: ", end="", flush=True)

            # --- GENERATION LOOP ---
            for _ in range(MAX_NEW_TOKENS):
                with torch.no_grad():
                    with torch.autocast(device_type="cuda" if "cuda" in DEVICE else "cpu", dtype=DTYPE):
                        outputs = model(input_ids=current_sdr, return_dict=True)

                    next_token_logits = outputs.logits[:, -1, :]

                    # Sampling
                    next_token_id = sample_next_token(
                        next_token_logits,
                        current_ids,
                        temperature=TEMP,
                        top_k=TOP_K,
                        top_p=TOP_P,
                        penalty=REP_PENALTY
                    )

                    next_token_item = next_token_id.item()

                    # Stop conditions
                    if next_token_item in [raw_tokenizer.eos_token_id, raw_tokenizer.pad_token_id]:
                        break

                    # Print
                    word = raw_tokenizer.decode(next_token_item)
                    print(word, end="", flush=True)

                    # Update History
                    current_ids = torch.cat([current_ids, next_token_id], dim=1)

                    # Update SDR History
                    next_sdr = sdr_tokenizer._convert_ids_to_sdr(next_token_id.view(-1)).to(DEVICE)
                    next_sdr = next_sdr.unsqueeze(1)  # [1, 1, 41]
                    current_sdr = torch.cat([current_sdr, next_sdr], dim=1)

            print("")

        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    chat()
