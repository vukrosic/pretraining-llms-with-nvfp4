
# %% [markdown]
# # Implementing the NVFP4 Recipe From Scratch: A Developer's Tutorial
#
# This tutorial deconstructs the core algorithms from PR #2177 to teach you how to implement them conceptually. We will build Python/PyTorch reference functions that mirror the logic of the new C++/CUDA kernels.
#
# Our goal is to implement these key components:
#
# 1.  **Core 1D Block Quantization**: The fundamental scaling and casting logic for 1x16 blocks.
# 2.  **2D Block Quantization**: An extension for quantizing 16x16 blocks, ideal for weights.
# 3.  **Random Hadamard Transform (RHT)**: The pre-quantization step to improve accuracy.
# 4.  **The Fused Operation**: Combining everything to produce the final `NVFP4Tensor` components.
#
# We will focus on the *algorithmic logic*, not CUDA-level performance optimizations.

# %%
import torch
import math

# For reproducibility
torch.manual_seed(0)
torch.cuda.manual_seed(0)

# %% [markdown]
# ## Step 1: Understanding the Target - The NVFP4 E2M1 Format
#
# Before we can quantize, we need to know what we're converting *to*. NVFP4 in this PR uses the `E2M1` format (2 exponent bits, 1 mantissa bit). It's a 4-bit floating-point number. We can represent all possible 16 values in a lookup table (LUT). This helps us simulate the casting process.
#
# The C++ code uses native `__nv_fp4_e2m1` types, but this LUT is perfect for a Python reference.

# %%
# The 16 possible values for an E2M1 FP4 number.
# Index corresponds to the 4-bit integer representation.
FP4_E2M1_LUT = torch.tensor([
    # Positive values (first bit 0)
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    # Negative values (first bit 1)
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=torch.float32)

# The maximum absolute value for E2M1 is 6.0. This is a critical constant.
FP4_E2M1_MAX_VAL = 6.0

def find_closest_fp4_val(value):
    """Simulates casting a float to the nearest FP4 value."""
    # Find the value in our LUT that is closest to the input value.
    # The index of this closest value is our 4-bit representation.
    return torch.argmin(torch.abs(value - FP4_E2M1_LUT.to(value.device)))

print(f"FP4 E2M1 Lookup Table:\n{FP4_E2M1_LUT}")
print(f"\nExample: Casting 2.9 to FP4 -> finds value {FP4_E2M1_LUT[find_closest_fp4_val(torch.tensor(2.9))]}")
print(f"Example: Casting -4.2 to FP4 -> finds value {FP4_E2M1_LUT[find_closest_fp4_val(torch.tensor(-4.2))]}")

# %% [markdown]
# ## Step 2: Implementing 1D Block Quantization
#
# This is the core logic. For each 1D block of 16 elements in a tensor row, we perform these steps. This logic is what the reference implementation `quantize_nvfp4_1d` in `test_cast_nvfp4_transpose.cu` performs.
#
# 1.  Find the absolute maximum value (`amax`) in the 16-element block.
# 2.  Calculate a `scaling_factor` for this block. The formula is `amax / FP4_E2M1_MAX_VAL`.
# 3.  **Scale** the original 16 values by dividing by the `scaling_factor`.
# 4.  **Cast** the scaled values to the nearest FP4 value.
# 5.  Store the resulting 16 4-bit integers and the single `scaling_factor`.

# %%
def quantize_1d_block_reference(hp_tensor: torch.Tensor):
    """
    Reference implementation for 1D block quantization (1x16 blocks).
    """
    assert hp_tensor.dim() == 2, "Input must be a 2D tensor"
    rows, cols = hp_tensor.shape
    assert cols % 16 == 0, "Columns must be divisible by 16"

    # Outputs
    num_scale_blocks = cols // 16
    quantized_data = torch.zeros(rows, cols, dtype=torch.int8, device=hp_tensor.device)
    scaling_factors = torch.zeros(rows, num_scale_blocks, dtype=hp_tensor.dtype, device=hp_tensor.device)

    for i in range(rows):
        for j in range(num_scale_blocks):
            # 1. Get the 1x16 block
            start_col, end_col = j * 16, (j + 1) * 16
            block = hp_tensor[i, start_col:end_col]

            # 2. Find amax
            block_amax = torch.max(torch.abs(block))
            if block_amax == 0: # Handle all-zero blocks
                scaling_factors[i, j] = 0.0
                # Quantized data is already 0
                continue

            # 3. Calculate scaling factor
            scaling_factor = block_amax / FP4_E2M1_MAX_VAL
            scaling_factors[i, j] = scaling_factor

            # 4. Scale the block
            scaled_block = block / scaling_factor

            # 5. Cast to FP4 (by finding closest value in LUT)
            for k in range(16):
                quantized_data[i, start_col + k] = find_closest_fp4_val(scaled_block[k])

    return quantized_data, scaling_factors

# --- Test it ---
sample_tensor = torch.randn((2, 32), dtype=torch.bfloat16, device='cuda')
q_data_1d, scales_1d = quantize_1d_block_reference(sample_tensor)

print("--- 1D Quantization Example ---")
print(f"Original Tensor Shape: {sample_tensor.shape}")
print(f"Quantized Data Shape: {q_data_1d.shape} (stores 4-bit integer indices)")
print(f"Scaling Factors Shape: {scales_1d.shape}")
print("\nFirst row's scaling factors:")
print(scales_1d[0])

# %% [markdown]
# ## Step 3: Implementing 2D Block Quantization
#
# The PR enables 2D quantization for weights. The logic is similar, but the block size is now 16x16. There is only **one scaling factor for the entire 256-element block**. This is implemented in the reference function `quantize_nvfp4_2d` in `test_cast_nvfp4_transpose.cu`.

# %%
def quantize_2d_block_reference(hp_tensor: torch.Tensor):
    """
    Reference implementation for 2D block quantization (16x16 blocks).
    """
    assert hp_tensor.dim() == 2, "Input must be a 2D tensor"
    rows, cols = hp_tensor.shape
    assert rows % 16 == 0 and cols % 16 == 0, "Dimensions must be divisible by 16"

    # Outputs
    num_blocks_y, num_blocks_x = rows // 16, cols // 16
    quantized_data = torch.zeros_like(hp_tensor, dtype=torch.int8)
    scaling_factors = torch.zeros(num_blocks_y, num_blocks_x, dtype=hp_tensor.dtype, device=hp_tensor.device)

    for i in range(num_blocks_y):
        for j in range(num_blocks_x):
            # 1. Get the 16x16 block
            start_row, end_row = i * 16, (i + 1) * 16
            start_col, end_col = j * 16, (j + 1) * 16
            block = hp_tensor[start_row:end_row, start_col:end_col]

            # 2. Find amax for the entire 16x16 block
            block_amax = torch.max(torch.abs(block))
            if block_amax == 0:
                scaling_factors[i, j] = 0.0
                continue

            # 3. Calculate scaling factor
            scaling_factor = block_amax / FP4_E2M1_MAX_VAL
            scaling_factors[i, j] = scaling_factor

            # 4. Scale the block
            scaled_block = block / scaling_factor

            # 5. Cast to FP4
            # (Vectorized version for simplicity)
            quantized_block = torch.zeros_like(scaled_block, dtype=torch.int8)
            for y in range(16):
                for x in range(16):
                    quantized_block[y, x] = find_closest_fp4_val(scaled_block[y, x])
            quantized_data[start_row:end_row, start_col:end_col] = quantized_block

    return quantized_data, scaling_factors


# --- Test it ---
sample_tensor_2d = torch.randn((32, 64), dtype=torch.bfloat16, device='cuda')
q_data_2d, scales_2d = quantize_2d_block_reference(sample_tensor_2d)

print("--- 2D Quantization Example ---")
print(f"Original Tensor Shape: {sample_tensor_2d.shape}")
print(f"Quantized Data Shape: {q_data_2d.shape}")
print(f"Scaling Factors Shape: {scales_2d.shape} (2x4 blocks of 16x16)")
print("\nScaling factors for all 16x16 blocks:")
print(scales_2d)

# %% [markdown]
# ## Step 4: Implementing Random Hadamard Transform (RHT)
#
# RHT is a pre-processing step applied to activations before quantization. It's a matrix multiplication with a special "Hadamard" matrix. The goal is to distribute the information across the vector, making quantization less lossy. The PR adds highly optimized kernels for this (`hadamard_transform_cast_fusion.cu`).
#
# Our reference will build the matrix and apply it block-wise.

# %%
def get_hadamard_matrix(size, device):
    """Constructs a Hadamard matrix of a power-of-two size."""
    if size == 1:
        return torch.ones((1, 1), device=device)
    h_prev = get_hadamard_matrix(size // 2, device)
    h_next = torch.cat([
        torch.cat([h_prev, h_prev], dim=1),
        torch.cat([h_prev, -h_prev], dim=1),
    ], dim=0)
    return h_next

def random_hadamard_transform_reference(hp_tensor: torch.Tensor):
    """Applies a 16x16 RHT to the tensor block-wise."""
    rows, cols = hp_tensor.shape
    assert cols % 16 == 0, "Columns must be divisible by 16"

    # The transform matrix includes normalization
    h_matrix = get_hadamard_matrix(16, hp_tensor.device).to(hp_tensor.dtype)
    h_matrix *= (1.0 / math.sqrt(16))

    transformed_tensor = torch.zeros_like(hp_tensor)

    for i in range(rows):
        for j in range(cols // 16):
            start_col, end_col = j * 16, (j + 1) * 16
            block = hp_tensor[i, start_col:end_col]
            # Apply the transform: block @ H
            transformed_block = torch.matmul(block, h_matrix)
            transformed_tensor[i, start_col:end_col] = transformed_block

    return transformed_tensor

# --- Test it ---
sample_tensor_rht = torch.randn((1, 32), dtype=torch.bfloat16, device='cuda')
transformed_tensor = random_hadamard_transform_reference(sample_tensor_rht)

print("--- RHT Example ---")
print("Original first 16 values:\n", sample_tensor_rht[0, :16])
print("\nTransformed first 16 values:\n", transformed_tensor[0, :16])
print(f"Shape remains the same: {transformed_tensor.shape}")


# %% [markdown]
# ## Step 5: The Fused Operation - Putting It All Together
#
# The true power of the PR is fusing all these steps into a single, efficient CUDA kernel. The kernel performs:
# `Cast -> RHT (optional) -> Quantize -> Transpose -> Quantize (again for transposed layout)`
#
# This avoids materializing intermediate tensors in memory and is much faster. Let's create a Python function that orchestrates our reference components to simulate this entire pipeline. This mimics the `compute_ref` function in `test_cast_nvfp4_transpose.cu`.

# %%
def nvfp4_recipe_reference(
    hp_tensor: torch.Tensor,
    use_rht: bool,
    use_2d_quant_for_weights: bool # In TE, this only applies to weights, but we simulate it here
):
    """
    Simulates the full, fused quantization pipeline.
    """
    # --- Process the input for row-wise (activation) usage ---
    processed_tensor = random_hadamard_transform_reference(hp_tensor) if use_rht else hp_tensor
    # Always use 1D quantization for activations/row-wise data
    q_data, scales = quantize_1d_block_reference(processed_tensor)

    # --- Process the input for column-wise (weight) usage ---
    hp_tensor_t = hp_tensor.T.contiguous()
    if use_2d_quant_for_weights:
        # NOTE: Real implementation pads to 16x16 blocks. We'll assume divisible dimensions.
        q_data_t, scales_t = quantize_2d_block_reference(hp_tensor_t)
    else:
        q_data_t, scales_t = quantize_1d_block_reference(hp_tensor_t)

    print("Simulated fused operation successful!")
    return q_data, scales, q_data_t, scales_t

# --- Test it with a realistic shape ---
activation_tensor = torch.randn((128, 2048), dtype=torch.bfloat16, device='cuda')

q_activation, scales_activation, q_weight, scales_weight = nvfp4_recipe_reference(
    activation_tensor,
    use_rht=True,
    use_2d_quant_for_weights=True
)

print("\n--- Outputs of the Fused Pipeline ---")
print(f"Quantized Activation Shape: {q_activation.shape}")
print(f"Activation Scales Shape: {scales_activation.shape}")
print(f"Quantized Weight (Transposed) Shape: {q_weight.shape}")
print(f"Weight Scales (Transposed) Shape: {scales_weight.shape}")


# %% [markdown]
# ## Step 6: The `NVFP4Tensor` Data Structure
#
# Finally, why does the PR introduce a new `NVFP4Tensor` class in Python?
#
# Because the results of the fused operation (`q_data`, `scales`, `q_data_t`, `scales_t`) all belong together. They represent a single high-precision tensor in its quantized form. The `NVFP4Tensor` acts as a container for all these components.
#
# When a TE layer needs the tensor for a forward pass GEMM (activations), it uses `q_data` and `scales`. When it needs the tensor for a wgrad GEMM (weights), it uses `q_data_t` and `scales_t`. This avoids costly re-quantization or transposing of packed 4-bit data on the fly.

# %%
from dataclasses import dataclass

@dataclass
class NVFP4TensorReference:
    """A Python dataclass to represent the real NVFP4Tensor structure."""
    _rowwise_data: torch.Tensor
    _rowwise_scale_inv: torch.Tensor
    _columnwise_data: torch.Tensor
    _columnwise_scale_inv: torch.Tensor
    original_shape: tuple

# Let's package our results into this structure
nvfp4_tensor_ref = NVFP4TensorReference(
    _rowwise_data=q_activation,
    _rowwise_scale_inv=scales_activation,
    _columnwise_data=q_weight,
    _columnwise_scale_inv=scales_weight,
    original_shape=activation_tensor.shape
)

print("Representation of a complete NVFP4Tensor object:")
print(nvfp4_tensor_ref)

# %% [markdown]
# ## Conclusion
#
# You have now implemented the core algorithmic building blocks of the NVFP4 recipe from scratch.
#
# You've learned that the implementation is not just a simple cast, but a sophisticated, fused pipeline that involves:
# 1.  **Block-based Scaling**: Calculating per-block scaling factors (either 1D or 2D).
# 2.  **Optional Pre-processing (RHT)**: Applying a mathematical transform to improve numerical stability.
# 3.  **Fused Operations**: Performing quantization and transposition in a single step to generate layouts for both forward and backward passes efficiently.
# 4.  **A Specialized Data Structure**: Using `NVFP4Tensor` to hold all the necessary components (data, scales, transposed versions) together.
#
# The actual C++/CUDA code in the PR takes these exact algorithms and implements them with extreme performance optimizations, using techniques like shared memory, tensor core instructions, and careful data movement to make 4-bit training feasible at scale.
    