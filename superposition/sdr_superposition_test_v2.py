import torch
from core.qwen_sdr_tokeriser_v4 import SDRQwenTokenizer


def sanitize_shape(tensor, expected_last_dim=41):
    """
    Forces any high-dimensional tensor into [Batch, Seq, LastDim].
    """
    flat = tensor.view(-1, tensor.shape[-1])
    if flat.shape[-1] != expected_last_dim:
        try:
            flat = tensor.view(-1, expected_last_dim)
        except:
            print(f"⚠️ Warning: Could not cleanly reshape {tensor.shape} to [..., {expected_last_dim}]")
            return tensor
    return flat.view(1, 1, expected_last_dim)


def indices_to_dense_sdr(indices_tensor, sdr_n=2048):
    """
    Converts Indices [Batch, Seq, 41] -> Dense Binary [Batch, Seq, 2048]
    """
    indices_tensor = sanitize_shape(indices_tensor, expected_last_dim=41)
    B, S, W = indices_tensor.shape
    binary_sdr = torch.zeros(B, S, sdr_n, device=indices_tensor.device)
    binary_sdr.scatter_(-1, indices_tensor.long(), 1.0)
    return binary_sdr


def rotate_binary_sdr(binary_sdr, shifts=1):
    """Circular shift right"""
    return torch.roll(binary_sdr, shifts=shifts, dims=-1)


def run_superposition_test_3_tokens():
    print("Initializing Tokenizer...")
    tokenizer = SDRQwenTokenizer(pretrained_model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct", sdr_n=2048, sdr_w=41)

    print("\n--- TEST: 3-Token Packing Limit (Tri-gram) ---")

    # 1. Get IDs for 3 distinct words
    tokens = ["Cat", "Sat", "Mat"]
    id_list = [tokenizer.encode(t, add_special_tokens=False) for t in tokens]

    # Convert to Tensors
    tensor_ids = [torch.tensor(i) for i in id_list]

    # Sanitize and Convert to Binary SDRs
    # Note: encode() returns [Batch, Seq, 41] directly in your tokenizer
    binary_sdrs = []
    for i, t_id in enumerate(tensor_ids):
        sanitized = sanitize_shape(t_id, 41)
        binary = indices_to_dense_sdr(sanitized, sdr_n=2048)
        binary_sdrs.append(binary)

    bin_cat, bin_sat, bin_mat = binary_sdrs[0], binary_sdrs[1], binary_sdrs[2]

    # 2. Create Bundles (3-Token Sequence)

    # Bundle A: "Cat Sat Mat"
    # Logic: Cat | Rot(Sat, 1) | Rot(Mat, 2)
    term1 = bin_cat
    term2 = rotate_binary_sdr(bin_sat, shifts=1)
    term3 = rotate_binary_sdr(bin_mat, shifts=2)

    # Combine using Max (Bitwise OR)
    bundle_A = torch.max(term1, torch.max(term2, term3))

    # Bundle B: "Sat Mat Cat" (Shifted Order)
    # Logic: Sat | Rot(Mat, 1) | Rot(Cat, 2)
    term1_b = bin_sat
    term2_b = rotate_binary_sdr(bin_mat, shifts=1)
    term3_b = rotate_binary_sdr(bin_cat, shifts=2)

    bundle_B = torch.max(term1_b, torch.max(term2_b, term3_b))

    # Bundle C: "Cat Mat Sat" (Swapped last two)
    # Logic: Cat | Rot(Mat, 1) | Rot(Sat, 2)
    term1_c = bin_cat
    term2_c = rotate_binary_sdr(bin_mat, shifts=1)
    term3_c = rotate_binary_sdr(bin_sat, shifts=2)

    bundle_C = torch.max(term1_c, torch.max(term2_c, term3_c))

    # 3. Stats & Comparisons
    print(f"\nSingle SDR Active Bits: ~41")
    print(f"Packed SDR (3 Tokens) Active Bits: {int(bundle_A.sum().item())}")

    def compare(name1, sdr1, name2, sdr2):
        intersection = (sdr1 * sdr2).sum().item()
        bits1 = sdr1.sum().item()
        bits2 = sdr2.sum().item()
        union = bits1 + bits2 - intersection
        sim = (intersection / union) * 100
        print(f"\nComparing '{name1}' vs '{name2}':")
        print(f"  Overlap: {int(intersection)} bits")
        print(f"  Similarity: {sim:.2f}%")
        return sim

    # Test 1: Complete Shift ("Cat Sat Mat" vs "Sat Mat Cat")
    sim_AB = compare("Cat Sat Mat", bundle_A, "Sat Mat Cat", bundle_B)

    # Test 2: Partial Swap ("Cat Sat Mat" vs "Cat Mat Sat")
    # This is harder because the first token ("Cat") is identical in both!
    sim_AC = compare("Cat Sat Mat", bundle_A, "Cat Mat Sat", bundle_C)

    print("\n--- CONCLUSION ---")
    limit_check = bundle_A.sum().item()
    sparsity = (limit_check / 2048) * 100
    print(f"SDR Density: {sparsity:.2f}% (Ideal is < 10-15%)")

    if sim_AB < 5.0:
        print("✅ Order is preserved (Totally distinct sequences).")
    elif sim_AC < 40.0:
        print("✅ Partial Order preserved (First token match detected, others distinct).")
    else:
        print("⚠️ Warning: Bundles are becoming too similar.")


if __name__ == "__main__":
    run_superposition_test_3_tokens()
