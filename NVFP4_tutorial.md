
# A Technical Guide to LLM Pretraining with NVFP4

*An overview of NVIDIA's 4-bit floating point format for efficient and accurate model training, based on the technical report "Pretraining Large Language Models with NVFP4".*

The growing scale of Large Language Models (LLMs) necessitates more efficient training methods. While 8-bit floating point (FP8) training is widely adopted, 4-bit floating point (FP4) formats offer further improvements in computational speed and memory usage. This guide provides a technical summary of **NVFP4**, a 4-bit format from NVIDIA, and the methodology required for its successful implementation in LLM pretraining.

## Background: Key Concepts in Numerical Precision

Before diving into NVFP4, it's essential to understand a few foundational concepts.

You can copy the 3 points below into an AI chatbot for personalized lessons.

*   **Numerical Precision:** In deep learning, numbers are typically stored in floating-point formats (e.g., FP32, FP16, BF16, FP8, FP4). The number in the format name indicates the number of bits used to represent a single number. More bits (like in FP32) allow for a wider range of numbers and higher precision (more detail), but consume more memory and are computationally slower. Fewer bits (like in FP4) are faster and more memory-efficient but have lower precision.

*   **Quantization:** This is the process of converting a tensor from a higher-precision format (e.g., FP32) to a lower-precision one (e.g., FP4). This is the core technique for accelerating model training and inference. However, this process can lead to a loss of information, which, if not managed correctly, can degrade the model's accuracy.

*   **Dynamic Range:** This refers to the range of values that can be represented by a numerical format, from the smallest non-zero number to the largest. When quantizing, we often scale the values in a tensor to fit within the limited dynamic range of the target format (e.g., FP4). A key challenge is that a single very large value (an "outlier") can dominate the entire range, forcing all other smaller values to be quantized to zero or near-zero, effectively erasing their information.

*   **The Outlier Problem in Scaling (An Example):** The presence of outliers - ex. `50` in `[0.5, -0.2, 1.1, -0.8, 50.0]` - is a major challenge in quantization. Because the scaling factor for a block of numbers is determined by the single largest value, one outlier can ruin the precision for all other numbers in its block.
    *   **Scenario:** Imagine we have a small block of numbers: `[0.5, -0.2, 1.1, -0.8, 50.0]`
    *   **The Outlier:** The value `50.0` is a significant outlier.
    *   **Scaling:** To quantize this block into FP4, which has a maximum representable value of `6.0`, we must scale every number down with the same scaling factor. `Scale Factor = 6.0 / 50.0 = 0.12`. The scaling factor is based on the largest number (taken absolute value)
    *   **Result:** After scaling (multiplying), our block becomes `[0.06, -0.024, 0.132, -0.096, 6.0]`.
    Here, the values that can be represented in FP4 (E2M1) are ±0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, and ±6. This means that after scaling, any value in the block will be rounded to the nearest of these discrete numbers. The representable range is thus from -6 to +6, with only these specific values available.
    *   **Information Loss:** When these new values are converted to the closest representable FP4 number (e.g., `±0`, `±0.5`, `±1.0`...), the first four values are so small that they will likely all be rounded to zero. The original information they contained is lost. Only the outlier retains its significance. NVFP4's techniques are designed to mitigate exactly this problem.

## Technical Advantages of NVFP4

Transitioning from FP8 to FP4 can yield a 2-3x increase in **arithmetic performance**—primarily the throughput of General Matrix Multiplication (GEMM) operations, which are the computational core of transformers—and a 50% reduction in memory usage. However, the lower precision introduces challenges. NVFP4 is designed to address these issues through several key features:

*   **Two-Level Scaling for High-Precision Representation:** In short, there are two scaling factors: one that applies to an entire tensor (like weights or activations, often millions of values), and a second that applies to each 16-element block within that tensor.

NVFP4 uses two distinct scaling factors, which is its most critical feature. To understand this, let's define two terms:
    *   A **Tensor** is a large, multi-dimensional array of numbers that holds the model's weights or activations. These are the core data structures in a neural network, and their scale can be immense. Here are some concrete examples based on the models described in the paper:
        *   **Weight Tensor:** In the 12B model, a single FFN weight tensor can have over 104 million values (shape: [5120, 20480]).
        *   **Activation Tensor:** In the 8B model, activations (layer outputs) between layers can form a 2D tensor of shape [6.3M, 4096] (from batch size 768 × sequence length 8192 × model dim 4096).
    *   A **Block** is a small, fixed-size chunk of a tensor. In NVFP4, a block is a group of just 16 contiguous numbers. So, the enormous weight and activation tensors above would be partitioned into thousands or millions of these tiny blocks for quantization.
    
    The two-level scaling works as follows:
    1.  **Coarse, Per-Tensor Scaling (FP32):** First, a single scaling factor is calculated for the *entire tensor* based on its absolute maximum value (max value(abs(tensor))). This factor performs a rough, global adjustment, bringing all the values in the tensor into a more manageable intermediate range.
    2.  **Fine-Grained, Per-Block Scaling (FP8):** After the first scaling is applied, the tensor is divided into thousands of small 16-element blocks. For *each* of these blocks, a second, more precise scaling factor is calculated. This local factor makes a fine-tuned adjustment, perfectly mapping the 16 values to the extremely limited range of FP4. The FP4 format used here (E2M1) can only represent the values ±0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, and ±6.
    
    This dual approach is powerful because it allows for highly localized adaptation. A large outlier in one part of the tensor will only affect the scaling of its tiny 16-element block, leaving the quantization of all other blocks completely unaffected. This preserves significantly more information compared to a single scaling factor.
*   **Reduced Block Size for Better Dynamic Range:** NVFP4 uses a smaller micro-block size of 16 elements. This is crucial for **capturing the local dynamic range**. In simpler terms, if a block of numbers contains one large outlier, only the other 15 numbers in that small block are affected during scaling. In a larger block (e.g., 32 elements), that same outlier would force a less precise scaling for all 31 other numbers, potentially causing more information loss. The smaller block size isolates the impact of outliers.
*   **Native Hardware Support:** The NVIDIA Blackwell GPU architecture includes Tensor Cores with native support for NVFP4, enabling significant hardware acceleration for GEMM operations.

These design choices allow NVFP4 to provide the efficiency of 4-bit precision while **mitigating the typical trade-offs**, namely the loss of model accuracy and potential for training instability that can arise from aggressive quantization.

## Core Methodology for NVFP4 Training

Achieving training outcomes comparable to FP8 requires a specific set of techniques. The following methodology is recommended for stable and accurate pretraining with NVFP4.

### 1. Mixed-Precision Strategy

Quantizing the entire model to FP4 can lead to divergence. A mixed-precision approach is crucial for stability.

**Implementation:**
*   Use NVFP4 for the majority of GEMM operations within the linear (fully-connected) layers.
*   Maintain a small percentage of numerically sensitive layers (approx. 15%, primarily the final layers of the network) in a higher precision format like BF16 or FP8.
*   Keep embeddings, normalization layers, attention mechanisms, and optimizer states in higher precision (BF16 or FP32).

### 2. Outlier Management with Random Hadamard Transforms (RHT)

Large-magnitude outliers can degrade accuracy in low-precision formats. RHT is an orthogonal transformation used to redistribute these outlier values into a more Gaussian-like distribution, making them more representable in FP4.

**Implementation:**
*   Apply RHT to the inputs of the weight gradient GEMMs (`Wgrad`), as this has the most significant impact on training stability.
*   A Hadamard matrix size of 16x16 offers a good trade-off between outlier mitigation and computational overhead for large-scale models.

### 3. Consistent Quantization with 2D Block Scaling

The backward pass in training involves transposing tensors, which can lead to different quantization schemes being applied in the forward and backward passes. This inconsistency can violate the chain rule and negatively affect convergence.

**Implementation:**
*   For weight tensors, use a **two-dimensional (2D) block scaling** strategy with a block size of 16x16. This ensures the same quantization is applied in both passes.
*   For activations and gradients, the standard 1D (1x16) block scaling is sufficient.

### 4. Unbiased Gradient Estimation with Stochastic Rounding

Deterministic rounding methods can introduce systematic bias during quantization, particularly for gradient tensors. Stochastic rounding is a probabilistic technique that mitigates this bias.

**Implementation:**
*   Apply **stochastic rounding** when quantizing all gradient tensors to FP4.
*   Use the deterministic **round-to-nearest-even** method for weights and activations, as stochastic rounding can introduce counterproductive noise in the forward pass.

## Empirical Validation: 12B Model on 10T Tokens

This methodology was validated by training a 12-billion parameter hybrid Mamba-Transformer model on 10 trillion tokens.

**Results:**
*   **Training Loss:** The validation loss for the NVFP4-trained model closely matched the FP8 baseline throughout the 10T token run.
*   **Downstream Task Accuracy:** The NVFP4 model achieved accuracies comparable to the FP8 baseline across a diverse set of downstream tasks, including reasoning, mathematics, and code generation. For example, the NVFP4 model achieved an MMLU-pro accuracy of 62.58%, nearly identical to the FP8 model's 62.62%.

This result constitutes the longest publicly documented 4-bit training run and demonstrates the viability of NVFP4 for large-scale pretraining.

## Format Comparison: NVFP4 vs. MXFP4

In a direct comparison using an 8B parameter model, NVFP4 demonstrated superior convergence over the MXFP4 format.

To achieve the same final training loss as the model trained with NVFP4, the model using MXFP4 required **36% more training tokens**. This suggests that the design of NVFP4 leads to greater sample efficiency.

## Conclusion

NVFP4, when combined with the specified training methodology, enables stable and accurate pretraining of large-scale language models in 4-bit precision. This approach offers significant efficiency gains in terms of computational throughput and memory usage without compromising model performance. Full support for NVFP4 is available in NVIDIA's Transformer Engine.

---

***Source:*** *This guide is a summary of the technical report "[Pretraining Large Language Models with NVFP4](https://placeholder_to_paper.com)". For complete details, please refer to the original publication.*
