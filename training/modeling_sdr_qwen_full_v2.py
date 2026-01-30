import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
import os

# IMPORTS
from core.modeling_sdr_qwen_full_v2 import SDRQwenForCausalLM
from core.qwen_sdr_tokeriser_v4 import SDRQwenTokenizer

# -----------------------------------------------------------------------------
# 1. CONFIGURATION
# -----------------------------------------------------------------------------
CONFIG = {
    # PHASE 1: Retina Alignment (Freeze Body, Train Projection)
    "phase1": {
        "enabled": True,
        "input_model": "./sdr_qwen_modular",  # Raw transplanted model
        "output_dir": "./sdr_qwen_phase1_retina",
        "lr": 1e-3,  # High LR for fresh layer
        "epochs": 1,  # TinyStories is large, 1 epoch is plenty
        "batch_size": 4,
        "grad_accum": 4
    },
    # PHASE 2: General Chat (Unfreeze All, Alpaca)
    "phase2": {
        "enabled": True,
        "input_model": "./sdr_qwen_phase1_retina",  # Loads from Phase 1
        "output_dir": "./sdr_qwen_phase2_chat",
        "lr": 2e-5,  # Standard Fine-Tuning LR
        "epochs": 2,
        "batch_size": 2,  # Lower batch size for full weights
        "grad_accum": 8
    },
    # PHASE 3: Math & Logic Specialist (Unfreeze All, GSM8k)
    "phase3": {
        "enabled": True,
        "input_model": "./sdr_qwen_phase2_chat",  # Loads from Phase 2
        "output_dir": "./sdr_qwen_phase3_math",
        "lr": 1e-5,  # Lower LR for delicate specialist training
        "epochs": 3,
        "batch_size": 2,
        "grad_accum": 8
    },
    "max_length": 1024,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "dtype": torch.bfloat16
}


# -----------------------------------------------------------------------------
# 2. DATASETS
# -----------------------------------------------------------------------------

class BaseSDRDataset(Dataset):
    """Parent class to handle common logic (Masking, SDR Conversion, EOS)"""

    def __init__(self, sdr_tokenizer, raw_tokenizer, max_length):
        self.sdr_tokenizer = sdr_tokenizer
        self.raw_tokenizer = raw_tokenizer
        self.max_length = max_length
        self.system_prompt = "You are a helpful SDR-based AI assistant."

    def process_sample(self, user_text, assistant_text):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text}
        ]

        # 1. Full Conversation (Input)
        full_text = self.raw_tokenizer.apply_chat_template(messages, tokenize=False)
        full_ids = self.raw_tokenizer.encode(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length - 1
        )
        # CRITICAL: Append EOS
        full_ids.append(self.raw_tokenizer.eos_token_id)

        # 2. User Prompt Only (For Masking)
        messages_user = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_text},
        ]
        user_text_fmt = self.raw_tokenizer.apply_chat_template(messages_user, tokenize=False,
                                                               add_generation_prompt=True)
        user_ids = self.raw_tokenizer.encode(user_text_fmt, add_special_tokens=False)

        mask_len = len(user_ids)

        # 3. Create Labels (Mask User = -100)
        labels = torch.tensor(full_ids, dtype=torch.long)
        if mask_len < len(labels):
            labels[:mask_len] = -100

        # 4. Convert to SDR
        sdr_tensor = self.sdr_tokenizer._convert_ids_to_sdr(torch.tensor(full_ids))

        return {
            "input_ids": sdr_tensor,
            "labels": labels
        }


class TinyStoriesSDRDataset(BaseSDRDataset):
    """For Phase 1: Retina Alignment"""

    def __init__(self, sdr_tokenizer, raw_tokenizer, max_length=512):
        super().__init__(sdr_tokenizer, raw_tokenizer, max_length)
        print("Loading TinyStories (Train Split)...")
        # Load first 100k samples to keep alignment fast but effective
        self.ds = load_dataset("roneneldan/TinyStories", split="train[:100000]")
        self.system_prompt = "You are a helpful assistant."  # Keep it simple for Phase 1

    def __len__(self): return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        # TinyStories is raw text, so we treat it as User: "Tell me a story" -> Assistant: [Story]
        # Or simpler: Just completion. But for consistency, let's wrap it lightly.
        return self.process_sample("Tell me a story.", item['text'])


class AlpacaSDRDataset(BaseSDRDataset):
    """For Phase 2: General Chat"""

    def __init__(self, sdr_tokenizer, raw_tokenizer, max_length=512):
        super().__init__(sdr_tokenizer, raw_tokenizer, max_length)
        print("Loading Alpaca-Cleaned Dataset...")
        self.ds = load_dataset("yahma/alpaca-cleaned", split="train")

    def __len__(self): return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        user_text = f"{item['instruction']}\n{item['input']}" if item['input'] else item['instruction']
        return self.process_sample(user_text, item['output'])


class MathSDRDataset(BaseSDRDataset):
    """For Phase 3: Math & Logic (GSM8k)"""

    def __init__(self, sdr_tokenizer, raw_tokenizer, max_length=512):
        super().__init__(sdr_tokenizer, raw_tokenizer, max_length)
        print("Loading GSM8k Math Dataset...")
        self.ds = load_dataset("openai/gsm8k", "main", split="train")
        self.system_prompt = "You are a math and logic expert. Solve this step-by-step."

    def __len__(self): return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        return self.process_sample(item['question'], item['answer'])


# -----------------------------------------------------------------------------
# 3. COLLATOR
# -----------------------------------------------------------------------------
class SDRCollator:
    def __init__(self, pad_sdr):
        self.pad_sdr = pad_sdr

    def __call__(self, batch):
        max_len = max([item['input_ids'].shape[0] for item in batch])
        batch_size = len(batch)
        sdr_w = batch[0]['input_ids'].shape[1]

        padded_inputs = torch.zeros(batch_size, max_len, sdr_w, dtype=torch.long)
        padded_labels = torch.full((batch_size, max_len), -100, dtype=torch.long)

        for i, item in enumerate(batch):
            seq_len = item['input_ids'].shape[0]
            padded_inputs[i, :seq_len, :] = item['input_ids']
            if seq_len < max_len:
                gap = max_len - seq_len
                padded_inputs[i, seq_len:, :] = self.pad_sdr.unsqueeze(0).expand(gap, -1)
            padded_labels[i, :seq_len] = item['labels']

        return padded_inputs, padded_labels


# -----------------------------------------------------------------------------
# 4. TRAINING ENGINE
# -----------------------------------------------------------------------------
def run_training_phase(phase_name, config, tokenizer_sdr, tokenizer_raw):
    print(f"\n" + "=" * 60)
    print(f"STARTING {phase_name.upper()}")
    print("=" * 60)

    device = CONFIG['device']

    # 1. Load Model
    print(f"Loading from {config['input_model']}...")
    model = SDRQwenForCausalLM.from_pretrained(
        config['input_model'],
        torch_dtype=CONFIG['dtype'],
        use_safetensors=True
    ).to(device)

    # 2. Configure Freezing
    trainable_params = 0
    all_params = 0
    print(f"Applying Freeze Strategy for {phase_name}...")

    for name, param in model.named_parameters():
        all_params += param.numel()
        if phase_name == "phase1":
            if "sdr_embed" in name:
                param.requires_grad = True
                trainable_params += param.numel()
            else:
                param.requires_grad = False
        else:  # Phase 2 & 3: Unfreeze All
            param.requires_grad = True
            trainable_params += param.numel()

    print(f"Stats: {trainable_params:,} Trainable / {all_params:,} Total ({100 * trainable_params / all_params:.2f}%)")

    # 3. Select Dataset
    if phase_name == "phase1":
        dataset = TinyStoriesSDRDataset(tokenizer_sdr, tokenizer_raw, CONFIG['max_length'])
    elif phase_name == "phase2":
        dataset = AlpacaSDRDataset(tokenizer_sdr, tokenizer_raw, CONFIG['max_length'])
    elif phase_name == "phase3":
        dataset = MathSDRDataset(tokenizer_sdr, tokenizer_raw, CONFIG['max_length'])

    pad_sdr = tokenizer_sdr._convert_ids_to_sdr(torch.tensor([tokenizer_raw.pad_token_id]))[0]
    collator = SDRCollator(pad_sdr)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collator)

    # 4. Optimizer
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config['lr'])

    # 5. Loop
    model.train()
    total_steps = len(dataloader) * config['epochs']
    current_step = 0

    print(f"Training for {config['epochs']} Epochs ({total_steps} Batches)...")

    for epoch in range(config['epochs']):
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            with torch.autocast(device_type="cuda" if "cuda" in device else "cpu", dtype=CONFIG['dtype']):
                outputs = model(input_ids=inputs, labels=labels, return_dict=True)
                loss = outputs.loss / config['grad_accum']

            loss.backward()

            if (batch_idx + 1) % config['grad_accum'] == 0:
                optimizer.step()
                optimizer.zero_grad()
                current_step += 1

                if current_step % 50 == 0:  # Log every 50 steps
                    raw_loss = loss.item() * config['grad_accum']
                    print(f"[{phase_name}] Ep {epoch + 1} | Step {current_step}/{total_steps} | Loss: {raw_loss:.4f}")

    # 6. Save
    print(f"Saving {phase_name} result to {config['output_dir']}...")
    model.save_pretrained(config['output_dir'])
    tokenizer_sdr.save_pretrained(config['output_dir'])
    tokenizer_raw.save_pretrained(config['output_dir'])
    print(f"{phase_name} Complete.")


# -----------------------------------------------------------------------------
# 5. MAIN ORCHESTRATOR
# -----------------------------------------------------------------------------
def main():
    print("Initializing Tokenizers...")
    base_model = "Qwen/Qwen2.5-0.5B-Instruct"
    raw_tokenizer = AutoTokenizer.from_pretrained(base_model)
    sdr_tokenizer = SDRQwenTokenizer(pretrained_model_name_or_path=base_model, sdr_n=2048, sdr_w=41)

    # Phase 1: Retina Alignment
    if CONFIG["phase1"]["enabled"]:
        run_training_phase("phase1", CONFIG["phase1"], sdr_tokenizer, raw_tokenizer)

    # Phase 2: Chat Fine-Tuning
    if CONFIG["phase2"]["enabled"]:
        run_training_phase("phase2", CONFIG["phase2"], sdr_tokenizer, raw_tokenizer)

    # Phase 3: Math & Logic Specialist
    if CONFIG["phase3"]["enabled"]:
        run_training_phase("phase3", CONFIG["phase3"], sdr_tokenizer, raw_tokenizer)


if __name__ == "__main__":
    main()
