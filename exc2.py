# %% [markdown]
# # Advanced Lessons: Implementing the NVFP4 Recipe
#
# Welcome to the advanced implementation tutorial for the NVFP4 recipe. In the previous session, we built high-level Python models of the core algorithms. Now, we will dissect the engineering principles and low-level details from the PR to understand how this is implemented for maximum performance on a GPU.
#
# ### Learning Path:
# *   **Lesson 1: The "Why" of Fused Kernels** - Why not just call the Python functions in sequence?
# *   **Lesson 2: Anatomy of the CUDA Kernel** - A conceptual breakdown of the C++ `block_scaled_1d_cast_transpose_kernel`.
# *   **Lesson 3: The Nuances of Two-Level Scaling** - Understanding the global (`S_enc`) and local (`S_dec_b`) scaling factors.
# *   **Lesson 4: Distributed Training & Quantized All-Gather** - How to handle custom data types in a multi-GPU setting.
# *   **Lesson 5: The Python API Glue** - How the `NVFP4Quantizer` class orchestrates everything.

# %%
import torch
import math

# For reproducibility
torch.manual_seed(0)
torch.cuda.manual_seed(0)

# Constants from the previous lesson
FP4_E2M1_MAX_VAL = 6.0
# A new constant from the PR: the max value of an FP8 E4M3 number, used for scaling factors.
FP8_E4M3_MAX_VAL = 448.0

# %% [markdown]
# ## Lesson 1: The "Why" of Fused Kernels - The Memory Bottleneck
#
# In our previous tutorial, we implemented each step (RHT, Quantize, Transpose) as a separate Python function. On a real GPU, this would be incredibly inefficient. Why? **Memory Bandwidth**.
#
# A GPU is fastest when it's doing math (computing). It's relatively slow when it's moving data between its main memory (HBM) and its compute cores. Operations like ours are often **memory-bound**, meaning the GPU spends more time waiting for data than computing on it.
#
# Consider the "naive" approach:
# 1.  `hp_tensor` is in Global Memory.
# 2.  **Kernel 1 (RHT)**: Load `hp_tensor`, compute RHT, write `rht_tensor` back to Global Memory.
# 3.  **Kernel 2 (Amax)**: Load `rht_tensor`, compute amax, write `amax_tensor` back to Global Memory.
# 4.  **Kernel 3 (Quantize)**: Load `rht_tensor` and `amax_tensor`, compute scales and quantized data, write `q_tensor` and `scales_tensor` to Global Memory.
# 5.  ...and so on for the transpose.
#
# This involves multiple round-trips to slow global memory. A **fused kernel**, like the one in this PR (`quantize_transpose_vector_blockwise_fp4.cu`), does all of this in a single trip.
#
# ### The Fused Kernel Strategy:
# 1.  **Launch ONE Kernel.**
# 2.  Threads load a small tile of the `hp_tensor` from Global Memory into ultra-fast **Shared Memory**.
# 3.  Perform all operations (RHT, amax reduction, scaling, casting) directly on the data in Shared Memory.
# 4.  Write the final, tiny outputs (`q_tensor` tile, `scales_tensor` tile) back to Global Memory.
#
# This minimizes global memory traffic and maximizes computation, leading to massive speedups. The entire PR is built around this principle.

# %% [markdown]
# ## Lesson 2: Anatomy of a Fused CUDA Kernel
#
# Let's write a "pseudo-code" walkthrough of the main kernel. We can't run CUDA C++ here, but we can model its logic and structure in Python comments to understand how it works. We'll focus on the `block_scaled_1d_cast_transpose_kernel` logic from the new C++ tests.
#
# A CUDA kernel is executed by a grid of *thread blocks*. Each block is responsible for processing one "tile" of the input data. Inside a block, threads cooperate using **Shared Memory**.

# %%
def conceptual_fused_kernel(hp_tensor):
    """A Python simulation of the fused kernel's logic for a single 16x16 tile."""
    # --- Kernel Launch Setup (Done by the CUDA runtime) ---
    # Imagine this function is ONE thread block, given an index (blockIdx.x, blockIdx.y)
    # to identify which 16x16 tile of the hp_tensor it should process.
    # Let's assume this block is responsible for the tile starting at (0, 0).
    TILE_DIM = 16
    block_start_row, block_start_col = 0, 0

    # --- Inside the Kernel (Execution on GPU) ---

    # 1. Cooperative Loading into Shared Memory
    # Each of the 256 threads in the block loads one element from global HBM
    # into the fast, on-chip shared memory scratchpad.
    shared_mem_tile = hp_tensor[
        block_start_row : block_start_row + TILE_DIM,
        block_start_col : block_start_col + TILE_DIM
    ].clone() # .clone() simulates the copy to a new memory space.
    # In CUDA, a `__syncthreads()` barrier would wait for all loads to complete.

    # 2. On-Chip AMAX Reduction (Row-wise)
    # The threads now work on the fast shared memory tile.
    # They cooperatively find the amax for each of the 16 rows in the tile.
    row_amaxes = torch.max(torch.abs(shared_mem_tile), dim=1).values
    # This is a simplified view. In CUDA, this is a multi-step reduction using
    # warp-level primitives (`__shfl_down_sync`) and another `__syncthreads()`.

    # 3. Calculate Row-wise Scaling Factors
    row_scales = row_amaxes / FP4_E2M1_MAX_VAL
    # Handle division by zero for all-zero rows
    row_scales[row_scales == 0] = 1.0

    # 4. Scale and Cast (Row-wise)
    # Each thread scales its value and simulates the cast.
    # The actual CUDA kernel uses a PTX instruction like `cvt.rn.satfinite.e2m1x2.f32`
    # which converts two FP32 numbers to two packed FP4 numbers in one go.
    scaled_tile = shared_mem_tile / row_scales.unsqueeze(1)
    quantized_tile = torch.round(scaled_tile).clamp(-FP4_E2M1_MAX_VAL, FP4_E2M1_MAX_VAL) # Simplified cast logic

    # 5. On-Chip Transposition
    # Threads cooperatively write to a second shared memory buffer in a transposed pattern.
    transposed_shared_mem_tile = shared_mem_tile.T.contiguous()
    # `__syncthreads()` ensures the transpose is complete.

    # 6. AMAX, Scale, and Cast (Column-wise / Transposed)
    # The process is repeated on the transposed tile to get the column-wise outputs.
    col_amaxes = torch.max(torch.abs(transposed_shared_mem_tile), dim=1).values
    col_scales = col_amaxes / FP4_E2M1_MAX_VAL
    col_scales[col_scales == 0] = 1.0
    scaled_transposed_tile = transposed_shared_mem_tile / col_scales.unsqueeze(1)
    quantized_transposed_tile = torch.round(scaled_transposed_tile).clamp(-FP4_E2M1_MAX_VAL, FP4_E2M1_MAX_VAL)

    # 7. Write Final Results to Global Memory
    # The threads write their final results from shared memory back to the final output tensors in HBM.
    # This is the only other time they touch global memory.
    print("Conceptual kernel finished processing one tile.")
    return quantized_tile, row_scales, quantized_transposed_tile, col_scales

# --- Run the conceptual model ---
sample_tile = torch.randn((16, 16), dtype=torch.float32, device='cuda')
q_data, scales, q_data_t, scales_t = conceptual_fused_kernel(sample_tile)

print(f"\nRow-wise quantized data shape: {q_data.shape}")
print(f"Row-wise scales shape: {scales.shape} (One scale per row in the tile)")
print(f"Column-wise quantized data shape: {q_data_t.shape}")
print(f"Column-wise scales shape: {scales_t.shape} (One scale per column in the tile)")

# %% [markdown]
# ## Lesson 3: The Nuances of Two-Level Scaling
#
# The previous lessons used a simplified scaling formula: `scale = amax / 6.0`. The actual implementation in the PR is more sophisticated, as seen in the C++ function `compute_global_encode_scaling_factor_FP4`. It uses a **two-level scaling system**.
#
# 1.  **Global Per-Tensor Scale (`S_enc`)**: A single FP32 scale factor is computed for the *entire tensor*. Its job is to map the tensor's global amax into a range that is friendly to FP8-E4M3, the format used for the *scaling factors themselves*.
#
# 2.  **Local Per-Block Scale (`S_dec_b`)**: This is the scale we've been calculating (`block_amax / 6.0`). It handles local variations.
#
# **The final scaling factor stored in memory is `S_final = S_dec_b * S_enc`**.
#
# Why do this? It improves numerical precision. By pre-scaling the entire tensor with `S_enc`, we ensure that the per-block `S_dec_b` values can be accurately represented by the FP8-E4M3 format.

# %%
def two_level_scaling_reference(hp_tensor: torch.Tensor):
    """Reference implementation for the two-level scaling logic."""
    # -- Level 1: Global Scaling --
    global_amax = torch.max(torch.abs(hp_tensor))

    # This formula is a direct translation of the C++ `compute_global_encode_scaling_factor_FP4`
    # It maps the global amax to the dynamic range of FP8 * FP4
    if global_amax == 0.0:
        S_enc = 1.0
    else:
        S_enc = (FP8_E4M3_MAX_VAL * FP4_E2M1_MAX_VAL) / global_amax
        S_enc = min(S_enc, torch.finfo(torch.float32).max) # Clamp to max float32

    # -- Level 2: Local Scaling (within a 1D block) --
    rows, cols = hp_tensor.shape
    num_scale_blocks = cols // 16
    final_scales = torch.zeros(rows, num_scale_blocks, dtype=torch.float32, device=hp_tensor.device)

    for i in range(rows):
        for j in range(num_scale_blocks):
            block = hp_tensor[i, j*16:(j+1)*16]
            block_amax = torch.max(torch.abs(block))

            # Calculate the local decoding scale
            S_dec_b = block_amax / FP4_E2M1_MAX_VAL

            # Combine with global encoding scale to get the final scale
            S_final = S_dec_b * S_enc

            # The final scale is then cast to FP8 E4M3 for storage.
            # We will just store it as float32 for this reference.
            final_scales[i, j] = S_final

    print(f"Global Amax: {global_amax:.4f}, S_enc (Global Scale): {S_enc:.4f}")
    return final_scales

# --- Test the two-level scaling ---
sample_tensor = torch.randn((2, 32), device='cuda') * 10 # Scale up to see a more interesting amax
final_scaling_factors = two_level_scaling_reference(sample_tensor)
print("\nFinal (two-level) scaling factors for the first row:")
print(final_scaling_factors[0])

# %% [markdown]
# ## Lesson 4: Distributed Training & Quantized All-Gather
#
# Making a new feature work on one GPU is only half the battle. For large models, it must work with tensor parallelism across multiple GPUs. This PR adds a custom `_all_gather_nvfp4` function in `transformer_engine/pytorch/distributed.py`.
#
# **The Problem**: You can't just call `torch.distributed.all_gather` on an `NVFP4Tensor` object. The All-Gather operation only works on single, contiguous `torch.Tensor`s.
#
# **The Solution**:
# 1.  Deconstruct the `NVFP4Tensor` on each GPU into its constituent `torch.Tensor` components (e.g., `_rowwise_data`, `_rowwise_scale_inv`).
# 2.  Perform a separate `all_gather` operation on each component tensor.
# 3.  Reconstruct a new, larger `NVFP4Tensor` on each GPU from the gathered components.
#
# **A New Problem (The "Interleave" Issue)**: When you gather a *transposed* tensor (like `_columnwise_data`) along the batch dimension, the data from different GPUs gets interleaved incorrectly.
#
# Imagine 2 GPUs. GPU0 has `[A0, B0]` and GPU1 has `[A1, B1]`. After gathering, the memory layout isn't `[A0, B0, A1, B1]`. It becomes something like `[A0, A1, B0, B1]`.
#
# To fix this, the PR adds a `swap_first_dims` operation. Let's simulate this.

# %%
def simulate_distributed_gather_and_fix():
    world_size = 4 # Simulate 4 GPUs
    local_dim0, local_dim1 = 2, 8

    # Create dummy transposed data on each GPU
    gpu_data = [torch.arange(local_dim0 * local_dim1, dtype=torch.float32).reshape(local_dim0, local_dim1) + (i*100) for i in range(world_size)]
    print(f"--- Data on GPU 0 (Transposed Layout) ---\n{gpu_data[0]}")

    # Simulate `all_gather` on the first dimension. This creates the interleaved result.
    interleaved_data = torch.cat(gpu_data, dim=0)
    print(f"\n--- Interleaved Data After All-Gather ---\n{interleaved_data}")

    # The `swap_first_dims` logic to fix the layout
    # This is what `tex.swap_first_dims` in the PR does in a highly optimized way.
    total_dim0 = interleaved_data.shape[0]
    fixed_data = interleaved_data.reshape(world_size, total_dim0 // world_size, -1).transpose(0, 1).reshape(total_dim0, -1)

    print(f"\n--- Data After `swap_first_dims` Fix ---\n{fixed_data}")

simulate_distributed_gather_and_fix()

# %% [markdown]
# ## Lesson 5: The Python API Glue - `NVFP4Quantizer`
#
# The `NVFP4Quantizer` class in `transformer_engine/pytorch/tensor/nvfp4_tensor.py` is the high-level orchestrator. It's the bridge between the Python world and the C++/CUDA backend.
#
# Let's break down its key responsibilities based on the PR:
#
# 1.  **Configuration (`__init__`)**: It reads the `Recipe` object and stores flags like `with_rht`, `stochastic_rounding`, and `with_2d_quantization`. It also pre-builds the RHT matrix if needed.
#
# 2.  **State Management**: It holds stateful information. For example, it generates and stores the random sign mask for the RHT matrix.
#
# 3.  **Backend Invocation (`quantize`)**: This is the main method. It takes a high-precision `torch.Tensor` as input.
#     *   It checks the tensor shape and properties.
#     *   It packages all the configuration flags and tensor pointers into a C-compatible structure (`QuantizationConfigWrapper`).
#     *   It calls the core C++ function (e.g., `tex.quantize_fp4`) through the Pybind11 bridge. This is the function that launches the fused CUDA kernel we discussed in Lesson 2.
#
# 4.  **Object Creation**: The C++ function returns raw tensor data. The `NVFP4Quantizer` takes this raw data and uses it to construct and return a proper, user-friendly `NVFP4Tensor` Python object.
#
# This class design cleanly separates the high-level configuration and object management in Python from the low-level, high-performance computations in C++/CUDA.

# %% [markdown]
# ## Grand Conclusion
#
# You have now journeyed from a high-level user of the NVFP4 recipe to understanding the deepest implementation details. You've learned:
#
# -   **Performance is King**: Fused kernels are essential to overcome memory bandwidth limitations, which is the primary motivation for the C++/CUDA implementation.
# -   **CUDA Programming Patterns**: Thread blocks, shared memory, and cooperative execution are the tools used to build these fused kernels.
# -   **Numerical Precision Matters**: The two-level scaling system is a clever trick to maintain accuracy when the scaling factors themselves must be stored in a low-precision format.
# -   **Distributed Systems are Complex**: Features must be designed with multi-GPU execution in mind, often requiring custom communication patterns like the fix for interleaved gathering.
# -   **APIs are Abstractions**: The Python `NVFP4Quantizer` class provides a clean interface that hides the immense complexity of the underlying C++/CUDA/distributed logic.
#
# You are now well-equipped to read through the files in PR #2177, such as `quantize_transpose_vector_blockwise_fp4.cu` and `distributed.py`, and recognize the patterns and algorithms we've discussed here.