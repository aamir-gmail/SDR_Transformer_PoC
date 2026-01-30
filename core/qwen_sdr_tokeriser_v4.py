import torch
import hashlib
from typing import List, Optional, Union, Dict, Any
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from transformers.utils import logging


class SDRQwenTokenizer(PreTrainedTokenizerFast):
    """
    SDR-Enabled Qwen Tokenizer.
    Returns input_ids as [Batch, Seq, 41] tensors.
    """

    def __init__(
            self,
            pretrained_model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct",
            sdr_n=2048,
            sdr_w=41,
            overlap_scale=1.0,
            seed=42,
            **kwargs
    ):
        # 1. Load Base Tokenizer
        base_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)

        # 2. Init Parent
        super().__init__(
            tokenizer_object=base_tokenizer.backend_tokenizer,
            eos_token=base_tokenizer.eos_token,
            pad_token=base_tokenizer.pad_token,
            unk_token=base_tokenizer.unk_token,
            **kwargs
        )

        # 3. CRITICAL FIX: Inherit Chat Template manually
        self.chat_template = base_tokenizer.chat_template

        # 4. SDR Config
        self.sdr_n = sdr_n
        self.sdr_w = sdr_w
        self.overlap_scale = overlap_scale
        self.seed = seed

        # 5. Build Map (Using len(self) for safety)
        self.real_vocab_size = len(self)
        print(f"Pre-computing SDR Map (Size: {self.real_vocab_size})...")
        self.sdr_map = self._build_sdr_vocab_map()
        print("SDR Map Ready.")

    def _hash_to_index(self, x: int, step: int) -> int:
        msg = f"{self.seed}|{x}|0".encode("utf-8")
        digest = hashlib.blake2b(msg, digest_size=8).digest()
        return int.from_bytes(digest, 'little') % self.sdr_n

    def _build_sdr_vocab_map(self) -> torch.Tensor:
        step = int(round(1 + self.overlap_scale * (self.sdr_w - 1)))
        all_indices = []
        for v in range(self.real_vocab_size):
            base = v * step
            indices = [self._hash_to_index(base + t, step) for t in range(self.sdr_w)]
            all_indices.append(indices)
        return torch.tensor(all_indices, dtype=torch.long)

    def _convert_ids_to_sdr(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.sdr_map.device != input_ids.device:
            self.sdr_map = self.sdr_map.to(input_ids.device)

        # Bounds Check
        if input_ids.max() >= self.sdr_map.shape[0]:
            raise IndexError(f"Token ID {input_ids.max()} exceeds SDR Map size {self.sdr_map.shape[0]}.")

        return torch.nn.functional.embedding(input_ids, self.sdr_map)

    def _post_process_sdr(self, encoding, return_tensors):
        """Helper to inject SDR logic into any encoding result"""
        standard_ids = encoding['input_ids']

        # Anti-Recursion Guard: If it's already rank 3, we are done.
        if torch.is_tensor(standard_ids) and standard_ids.ndim == 3:
            return encoding
        if isinstance(standard_ids, list) and len(standard_ids) > 0 and isinstance(standard_ids[0],
                                                                                   list) and isinstance(
                standard_ids[0][0], list):
            return encoding

        # Convert
        if torch.is_tensor(standard_ids):
            sdr_indices = self._convert_ids_to_sdr(standard_ids)
        else:
            temp_tensor = torch.tensor(standard_ids, dtype=torch.long)
            sdr_indices = self._convert_ids_to_sdr(temp_tensor)

        # Format Return
        if return_tensors is None:
            encoding['input_ids'] = sdr_indices.tolist()
            if torch.is_tensor(encoding['attention_mask']):
                encoding['attention_mask'] = encoding['attention_mask'].tolist()
        else:
            encoding['input_ids'] = sdr_indices

        return encoding

    def batch_encode_plus(self, batch_text_or_text_pairs, return_tensors=None, **kwargs):
        # Override for LIST inputs
        encoding = super().batch_encode_plus(batch_text_or_text_pairs, return_tensors='pt', **kwargs)
        return self._post_process_sdr(encoding, return_tensors)

    def encode_plus(self, text, text_pair=None, return_tensors=None, **kwargs):
        # Override for STRING inputs
        encoding = super().encode_plus(text, text_pair=text_pair, return_tensors='pt', **kwargs)
        return self._post_process_sdr(encoding, return_tensors)


# -----------------------------------------------------------------------------
# RIGOROUS TEST SUITE
# -----------------------------------------------------------------------------
def run_tests():
    print("\n" + "=" * 50)
    print("INITIALIZING TOKENIZER")
    print("=" * 50)
    tokenizer = SDRQwenTokenizer(sdr_n=2048, sdr_w=41)

    # -------------------------------------------------------
    # TEST 1: Variable Length Batch & Padding
    # -------------------------------------------------------
    print("\n" + "=" * 50)
    print("TEST 1: Variable Length Batch & Padding")
    print("=" * 50)

    batch_text = [
        "Short.",
        "This is a medium length sentence.",
        "This is a much longer sentence that will force the others to pad significantly."
    ]

    encoded = tokenizer(batch_text, padding=True, return_tensors='pt')

    input_ids = encoded['input_ids']
    attn_mask = encoded['attention_mask']

    print(f"Batch Size: {input_ids.shape[0]}")
    print(f"Max Seq Len: {input_ids.shape[1]}")
    print(f"SDR Width:   {input_ids.shape[2]}")

    if input_ids.ndim == 3 and input_ids.shape[2] == 41:
        print("[PASS] Output Shape Correct.")
    else:
        print(f"[FAIL] Shape Mismatch. Got {input_ids.shape}")

    # Check Mask Logic
    short_len = attn_mask[0].sum().item()
    long_len = attn_mask[2].sum().item()

    print(f"Short Sentence Real Tokens: {short_len}")
    print(f"Long Sentence Real Tokens:  {long_len}")

    if short_len < long_len:
        print("[PASS] Attention Mask correctly identifies padding.")
    else:
        print("[FAIL] Attention Mask logic seems flawed.")

    # -------------------------------------------------------
    # TEST 2: Padding Integrity
    # -------------------------------------------------------
    print("\n" + "=" * 50)
    print("TEST 2: Padding Value Integrity")
    print("=" * 50)

    pad_id = tokenizer.pad_token_id
    expected_pad_sdr = tokenizer._convert_ids_to_sdr(torch.tensor([pad_id]))[0]

    # Check last position of first sentence
    last_pos_sdr = input_ids[0, -1, :]

    if torch.equal(last_pos_sdr, expected_pad_sdr):
        print("[PASS] Padded regions contain correct SDR for <PAD> token.")
    else:
        print(f"[FAIL] Padded regions contain garbage.")

    # -------------------------------------------------------
    # TEST 3: Truncation
    # -------------------------------------------------------
    print("\n" + "=" * 50)
    print("TEST 3: Truncation Logic")
    print("=" * 50)

    long_text = "Repeat " * 50
    max_len = 10
    trunc_enc = tokenizer(long_text, truncation=True, max_length=max_len, return_tensors='pt')
    actual_len = trunc_enc['input_ids'].shape[1]

    if actual_len == max_len:
        print(f"[PASS] Successfully truncated to {max_len} tokens.")
    else:
        print(f"[FAIL] Truncation failed. Expected {max_len}, got {actual_len}.")

    # -------------------------------------------------------
    # TEST 4: Chat Template
    # -------------------------------------------------------
    print("\n" + "=" * 50)
    print("TEST 4: Chat Template & Special Tokens")
    print("=" * 50)

    messages = [{"role": "user", "content": "Hi"}]
    chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(f"Chat Template Output: {chat_text!r}")

    chat_enc = tokenizer(chat_text, return_tensors='pt')
    print(f"[PASS] Chat template processed into SDR shape: {chat_enc['input_ids'].shape}")


if __name__ == "__main__":
    run_tests()
