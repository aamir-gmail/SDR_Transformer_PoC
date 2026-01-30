# What is a Sparse Distributed Representation (SDR)?

**Sparse Distributed Representations (SDRs)** are a data format based on the theoretical operations of the mammalian neocortex. Unlike the dense, continuous vector embeddings used in standard Deep Learning (e.g., Word2Vec, BERT), SDRs use massive, highly sparse binary vectors to encode semantic meaning.

## 1. The Representation
An SDR is a long binary vector (typically $N \ge 2048$ bits) where only a tiny percentage of bits (typically $\sim2\%$) are active ($1$) at any given time.

* **Dense Embedding (Standard):** `[0.23, -0.91, 0.55, ...]` (Continuous, Compact)
* **SDR (Neuromorphic):** `[0, 0, 1, 0, 0, 0, 1, 0, ...]` (Binary, Vast, Sparse)

### Why "Distributed"?
The semantic meaning is **distributed** across the pattern of active bits.
* No single bit represents "Cat".
* The *set* of bits $\{5, 120, 900, ...\}$ represents "Cat".
* If a few bits are flipped due to noise, the semantic meaning remains intact because the overall pattern is still recognizable.

## 2. Mathematical Superpowers
SDRs possess unique mathematical properties that make them ideal for robust AI architectures:

### A. Superposition (The "Union" Property)
You can bundle multiple concepts into a single vector using a bitwise **OR** operation. Unlike dense vectors (where averaging mixes concepts into a blur), SDR unions retain the discrete identity of every component.
$$\text{SDR}(\text{Cat}) \cup \text{SDR}(\text{Sat}) = \text{SDR}(\text{Cat, Sat})$$
* **Capacity:** Because the space is so vast ($2^{2048}$), millions of unique patterns can coexist without colliding.
* **Retrieval:** You can mathematically query the union: *"Is 'Cat' inside this bundle?"* by checking if the 'Cat' bits are active.

### B. High-Noise Robustness
SDRs are inherently fault-tolerant.
* **Signal Loss:** You can turn off 50% of the active bits, and the system can still identify the original concept (the "Attractor Basin").
* **Noise Injection:** Random noise bits have little effect because they are unlikely to form a meaningful competing pattern by chance.

## 3. Comparison: Dense vs. SDR

| Feature | Dense Embeddings (Standard Transformers) | Sparse Distributed Representations (SDR) |
| :--- | :--- | :--- |
| **Data Type** | Float32 / Float16 | Binary (0/1) |
| **Dimensions** | Low (e.g., 768, 4096) | High (e.g., 2048, 16384) |
| **Activity** | 100% Dense | ~2% Sparse |
| **Combination** | Vector Addition (Blurs meaning) | Boolean Union (Preserves distinct items) |
| **Noise Tolerance** | Low (Values change significantly) | Extremely High |
| **Biologically Plausible?** | No | Yes |

## 4. Relevance to this Project
In this repository, we replace the standard dense embedding layer of an LLM with an **SDR Retina**. This allows the Transformer to:
1.  **Compute Efficiently:** Processing sparse binaries is computationally cheaper than dense floats.
2.  **Compress Context:** Utilizing the **Superposition** property to pack multiple tokens into a single input step (e.g., 4x Context Compression).
3.  **Resist Noise:** Maintaining high accuracy even when inputs are heavily corrupted.
