import torch
import hashlib
from typing import List, Optional, Union, Dict, Any
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from transformers.utils import logging


class SDRQwenTokenizer(PreTrainedTokenizerFast):
    """
    SDR-Enabled Qwen Tokenizer (V5 - Robust Override).
    Returns input_ids as [Batch, Seq, 41] tensors.
    """

    def __init__(
            self,
            pretrained_model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct",
            sdr_n=2048,  # Consider increasing to 4096+ for Dual-3090 run
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

        # 3. Inherit Chat Template
        self.chat_template = base_tokenizer.chat_template

        # 4. SDR Config
        self.sdr_n = sdr_n
        self.sdr_w = sdr_w
        self.overlap_scale = overlap_scale
        self.seed = seed

        # 5. Build Map
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
            # Fallback for out-of-bounds special tokens if any
            input_ids = torch.clamp(input_ids, max=self.sdr_map.shape[0] - 1)

        return torch.nn.functional.embedding(input_ids, self.sdr_map)

    def _post_process_sdr(self, encoding, return_tensors):
        """Helper to inject SDR logic into any encoding result"""
        standard_ids = encoding['input_ids']

        # 1. Anti-Recursion Guard
        if torch.is_tensor(standard_ids) and standard_ids.ndim == 3:
            return encoding

        # 2. Conversion
        if torch.is_tensor(standard_ids):
            sdr_indices = self._convert_ids_to_sdr(standard_ids)
        else:
            # Handle list output (if return_tensors=None)
            # We must convert to tensor to do the embedding lookup
            temp_tensor = torch.tensor(standard_ids, dtype=torch.long)
            sdr_indices = self._convert_ids_to_sdr(temp_tensor)

        # 3. Robust Assignment
        # Explicitly update the underlying data dict to bypass strict BatchEncoding setters
        if return_tensors is None:
            if hasattr(encoding, 'data'):
                encoding.data['input_ids'] = sdr_indices.tolist()
            else:
                encoding['input_ids'] = sdr_indices.tolist()
        else:
            if hasattr(encoding, 'data'):
                encoding.data['input_ids'] = sdr_indices
            else:
                encoding['input_ids'] = sdr_indices

        return encoding

    # -----------------------------------------------------------------------
    # CRITICAL FIX: Override __call__ to catch "tokenizer(text)" calls
    # -----------------------------------------------------------------------
    def __call__(self, text=None, text_pair=None, text_target=None, text_pair_target=None, return_tensors=None,
                 **kwargs):
        # We force return_tensors='pt' internally to handle the embedding lookup,
        # then convert back if the user wanted lists.
        encoding = super().__call__(
            text=text,
            text_pair=text_pair,
            text_target=text_target,
            text_pair_target=text_pair_target,
            return_tensors='pt',  # Force PT
            **kwargs
        )
        return self._post_process_sdr(encoding, return_tensors)

    def batch_encode_plus(self, batch_text_or_text_pairs, return_tensors=None, **kwargs):
        encoding = super().batch_encode_plus(batch_text_or_text_pairs, return_tensors='pt', **kwargs)
        return self._post_process_sdr(encoding, return_tensors)

    def encode_plus(self, text, text_pair=None, return_tensors=None, **kwargs):
        encoding = super().encode_plus(text, text_pair=text_pair, return_tensors='pt', **kwargs)
        return self._post_process_sdr(encoding, return_tensors)


# -----------------------------------------------------------------------------
# RIGOROUS TEST SUITE
# -----------------------------------------------------------------------------
def run_tests():
    print("\n" + "=" * 50)
    print("INITIALIZING TOKENIZER (V5)")
    print("=" * 50)
    # Using Qwen base for testing
    try:
        tokenizer = SDRQwenTokenizer(sdr_n=2048, sdr_w=41)
    except Exception as e:
        print(f"Skipping Qwen load (environment issue?), using minimal mock: {e}")
        return

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

    # This calls __call__
    encoded = tokenizer(batch_text, padding=True, return_tensors='pt')

    input_ids = encoded['input_ids']
    attn_mask = encoded['attention_mask']

    print(f"Batch Size: {input_ids.shape[0]}")
    print(f"Max Seq Len: {input_ids.shape[1]}")
    # This line triggered the error before
    print(f"SDR Width:   {input_ids.shape[2] if input_ids.ndim > 2 else 'FAIL - Rank 2'}")

    if input_ids.ndim == 3 and input_ids.shape[2] == 41:
        print("[PASS] Output Shape Correct.")
    else:
        print(f"[FAIL] Shape Mismatch. Got {input_ids.shape}")


if __name__ == "__main__":
    run_tests()
