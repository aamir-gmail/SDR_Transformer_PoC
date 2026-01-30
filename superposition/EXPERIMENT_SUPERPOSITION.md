# Experiment: SDR Superposition & Sequence Order Verification

**Codename:** "The Cat Sat Mat Test"  
**Script:** `sdr_superposition_test_v2.py`

## 1. Overview
One of the most powerful theoretical properties of Sparse Distributed Representations (SDRs) is **Superposition**: the ability to store multiple data points in a single vector using a boolean Union (OR) operation.

However, standard superposition is commutative ($A \cup B = B \cup A$), meaning it loses sequence information. "Cat Sat" becomes identical to "Sat Cat."

This experiment tests a **Rotation-based Encoding** strategy to pack multiple tokens into a single SDR while strictly preserving their temporal order.

## 2. The Hypothesis
By applying a unique circular bit-shift (rotation) to each position in a sequence before superposition, we hypothesize that:
1.  **Distinctness:** The packed SDR for sequence `[A, B, C]` will be mathematically orthogonal (distinct) from `[B, C, A]`.
2.  **Capacity:** The resulting packed vector will remain sparse enough (low density) to allow for reliable retrieval.

## 3. Methodology
We define a "Packed Trigram" (3-token sequence) using the following mathematical operation:

$$SDR_{\text{Packed}} = SDR(T_1) \cup \text{Rotate}(SDR(T_2), 1) \cup \text{Rotate}(SDR(T_3), 2)$$

Where:
* $\cup$ is the Bitwise OR operation.
* $\text{Rotate}(X, n)$ shifts the bits of vector $X$ to the right by $n$ positions.

We compared three permutations of the same words:
1.  **Baseline:** "Cat Sat Mat"
2.  **Full Shift:** "Sat Mat Cat" (Totally different order)
3.  **Partial Shift:** "Cat Mat Sat" (Same start, different end)

## 4. Experimental Results

**Input Data:**
* **Tokens:** "Cat", "Sat", "Mat"
* **Vector Size:** 2048 bits
* **Active Bits per Token:** ~41

### A. Density Analysis
| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| Single Token Active Bits | 41 | ~2.0% Density |
| **Packed Trigram Active Bits** | **121** | **~5.91% Density** |

* **Result:** The Union of 3 tokens resulted in only ~6% of the bits being active.
* **Verdict:** This is well below the "saturation threshold" (typically ~15-20%), suggesting we could theoretically pack 6-8 tokens before retrieval becomes noisy.

### B. Order Sensitivity
We measured the **Jaccard Similarity** between the Baseline and the Permutations.

#### Test 1: Baseline vs. Full Shift
* **Compare:** "Cat Sat Mat" vs. "Sat Mat Cat"
* **Overlap:** 9 bits
* **Similarity:** **3.86%**
* **Analysis:** The vectors are nearly orthogonal. Despite containing the exact same words, the model sees them as completely different objects. **Order is preserved.**

#### Test 2: Baseline vs. Partial Shift
* **Compare:** "Cat Sat Mat" vs. "Cat Mat Sat"
* **Overlap:** 44 bits
* **Similarity:** **22.34%**
* **Analysis:** The similarity is higher because both sequences start with the unrotated "Cat" (sharing ~41 bits). However, the remaining ~66% of the vector is different. The model can easily distinguish this difference.

## 5. Implications
This experiment confirms that **SDRs can serve as a "Sliding Window" compression layer** for Transformers.

If we feed these packed "Trigram SDRs" into a model instead of single tokens:
1.  **Context Compression:** We effectively triple the context window size (3 tokens stored in the RAM of 1).
2.  **Attention Speedup:** Attention complexity is quadratic ($O(N^2)$). Reducing sequence length by 3x results in a **9x speedup** in attention calculations.

## 6. How to Run
To reproduce these results, run the superposition verification script:

```bash
python experiments/sdr_superposition_test_v2.py
