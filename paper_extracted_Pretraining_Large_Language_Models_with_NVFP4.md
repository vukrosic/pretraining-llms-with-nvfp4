# Pretraining Large Language Models with NVFP4

*Extracted from PDF document*

---

## Page 1

2 0 2 5-9-3 0
Pretraining Large Language Models with NVFP 4
### Nvidia
Abstract. Large Language Models (LLMs) today are powerful problem solvers across many domains, and they
continue to get stronger as they scale in model size, training set size, and training set quality, as shown by extensive
research and experimentation across the industry. Training a frontier model today requires on the order of tens
to hundreds of yottaflops, which is a massive investment of time, compute, and energy. Improving pretraining
efficiency is therefore essential to enable the next generation of even more capable LLMs. While 8-bit floating point
(FP 8) training is now widely adopted, transitioning to even narrower precision, such as 4-bit floating point (FP 4),
could unlock additional improvements in computational speed and resource utilization. However, quantization at
this level poses challenges to training stability, convergence, and implementation, notably for large-scale models
trained on long token horizons.
In this study, we introduce a novel approach for stable and accurate training of large language models (LLMs)
using the NVFP 4 format. Our method integrates Random Hadamard transforms (RHT) to bound block-level
outliers, employs a two-dimensional quantization scheme for consistent representations across both the forward
and backward passes, utilizes stochastic rounding for unbiased gradient estimation, and incorporates selective
high-precision layers. We validate our approach by training a 1 2-billion-parameter model on 1 0 trillion tokens –
the longest publicly documented training run in 4-bit precision to date. Our results show that the model trained
with our NVFP 4-based pretraining technique achieves training loss and downstream task accuracies comparable to
an FP 8 baseline. For instance, the model attains an MMLU-pro accuracy of 6 2.5 8%, nearly matching the 6 2.6 2%
accuracyachievedthrough FP 8 pretraining. Thesefindingshighlightthat NVFP 4, whencombinedwithourtraining
approach, represents a major step forward in narrow-precision LLM training algorithms.
Code: Transformer Engine support for NVFP 4 training.
1. Introduction
The rapid expansion of large language models (LLMs) has increased the demand for more efficient
numerical formats to lower computational cost, memory demand, and energy consumption during training.
8-bit floating point (FP 8 and MXFP 8) has emerged as a popular data type for accelerated training of
LLMs (Micikevicius et al., 2 0 2 2; Deep Seek-AI et al., 2 0 2 4; Mishra et al., 2 0 2 5). Recent advances in
narrow-precision hardware (NVIDIA Blackwell, 2 0 2 4) have positioned 4-bit floating point (FP 4) as the
next logical step (Tseng et al., 2 0 2 5 b; Chmiel et al., 2 0 2 5; Wang et al., 2 0 2 5; Chen et al., 2 0 2 5; Castro
et al., 2 0 2 5; Zhou et al., 2 0 2 5; Rouhani et al., 2 0 2 3), delivering a two- to three-fold boost in arithmetic
performance and reducing memory usage by half compared to FP 8.
This technical report presents an in-depth analysis of large language model (LLM) pretraining using
NVFP 4(Alvarezetal.,2 0 2 5),a 4-bitdataformatthatextendsthe“microscaling”approach(Rouhanietal.,
2 0 2 3). Unlike 4-bit microscaling formats such as MXFP 4 (Rouhani et al., 2 0 2 3; Open-Compute-Project,
2 0 2 3), NVFP 4 employs a smaller micro-block structure, which more effectively captures the local dynamic
range in the data. NVFP 4 also utilizes an FP 8 scale factor format that incorporates fractional precision
for more accurate microscaling. In addition, NVFP 4 employs a two-level scaling strategy, which combines
a fine-grained FP 8 scale factor with an FP 3 2 scale applied at the tensor level. These design choices allow
for more precise and accurate representation of tensor values during training.
Leveraging the NVFP 4 format, we introduce a 4-bit training methodology that achieves accuracies
comparable to FP 8 on very strong language models. This approach preserves numerically sensitive layers
in higher precision, utilizes two-dimensional (2 D) block scaling to maintain same quantized representations
across forward and backward passes, applies Random Hadamard transforms (Tseng et al., 2 0 2 5 b; Castro
et al., 2 0 2 5) to disperse large-magnitude outliers, and employs stochastic rounding (Tseng et al., 2 0 2 5 b;
Chmiel et al., 2 0 2 5; Chen et al., 2 0 2 5; Castro et al., 2 0 2 5) on gradients to reduce quantization bias.
Ablationstudiesconfirmthateachcomponentofthismethodologyisimportantfor 4-bittraining,especially
© 2 0 2 5 NVIDIA.Allrightsreserved.
5 2 0 2
pe S
9 2
]LC.sc[
1 v 9 4 1 5 2.9 0 5 2:vi Xra

## Page 2

Pretraining Large Language Modelswith NVFP 4
in large-scale models and during long token horizons.
To validate our approach, we train a very strong 1 2-billion parameter LLM (NVIDIA, 2 0 2 5 b) on 1 0 trillion
tokens, demonstrating that its loss curve and accuracies on downstream tasks closely match with those of
an FP 8 baseline. While our work establishes the feasibility of FP 4 training at large scales, this report is
primarily concerned with the underlying algorithms and methodology rather than with runtime efficiency
or system-level optimizations. This marks, to our knowledge, the first successful demonstration of training
billion-parameter language models with 4-bit precision over a multi-trillion-token horizon, laying the
foundation for faster and more efficient training of future frontier models.
The remainder of this technical report is organized as follows: Section 2 describes the NVFP 4 format,
Section 3 presents results for a 1 2 billion model trained on 1 0 trillion tokens with NVFP 4, Section 4
discusses the training methodology for NVFP 4, and Section 5 compares training with NVFP 4 and
MXFP 4. The appendices include details of the training setup (models, datasets, and hyperparameters),
the quantization procedure, and ablation studies analyzing the impact of different technique choices.
2. NVFP 4 Format
Due to the limited range of narrow floating-point formats, microscaling (MX) formats (Open-Compute-
Project, 2 0 2 3) were introduced to balance dynamic range and precision. These formats are characterized
by a block-wise representation where a group of data elements shares a single, common scale factor. MX
formatsinclude 8-bit(MXFP 8), 6-bit(MXFP 6), and 4-bit(MXFP 4)floating-pointtypes. In MXFP 4, each
element is represented as E 2 M 1 1 (Open-Compute-Project, 2 0 2 3), meaning it has 1 sign bit, 2 exponent
bits, and 1 mantissa bit. This allows MXFP 4 to encode the values ±0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, and
±6.
Since original higher-precision values (e.g., FP 3 2 or BF 1 6) often exceed the FP 4 range, they must be
scaled into the representable range during quantization. Scale factors are typically chosen so that the
absolute maximum value (amax) within a block maps to the FP 4 maximum representable, favoring
the prevention of saturations while minimizing small magnitudes being lost to zero. After scaling, high
precision values in a tensor are rounded to the nearest FP 4-representable number and later decoded back
to their original range using the reciprocal of the same scale. To improve hardware efficiency, MX formats
store block scale factors in 8 bits. Each block of 3 2 contiguous elements in a tensor shares a single 8-bit
scale factor, stored in an unsigned E 8 M 0 format (UE 8 M 0), which encodes a power-of-two value ranging
from 2−1 2 7 to 2 1 2 7. Mishra et al. (2 0 2 5) found that it is beneficial to round scale factors up to the next
representable UE 8 M 0 value to avoid saturations.
NVFP 4 is an enhanced 4-bit format that provides improved numerical properties over MXFP 4. First, by
reducing the block size from 3 2 to 1 6 elements, NVFP 4 narrows the dynamic range within each block,
better fitting values into the FP 4 range. Second, block scale factors are stored in E 4 M 3 rather than
UE 8 M 0, trading some exponent range for additional mantissa bits. Third, an FP 3 2 scale is applied at the
tensor level to retain the range of block scales. With such a two-level microscaling approach, NVFP 4
encodes at least 6.2 5% of values in a block (the amax values in each block of 1 6 elements) at near-FP 8
precision, while storing the remaining values in FP 4 (see Figure 1). In contrast, MXFP 4 stores all values
in FP 4, and can potentially lose up to one binade of dynamic range (and four samples: ±4 and ±6)
because of power-of-two scale factor rounding (see Appendix B.4 for details).
For NVFP 4, having more precise scaling with E 4 M 3 reduces the range available for representing the scale
factors. As a result, a second level of FP 3 2 scaling is used to adjust the original tensor’s distribution such
that block scale factors can be represented in E 4 M 3. This two-level scaling scheme works as follows: (1) a
per-tensor FP 3 2 scale remaps all the values within a tensor into representable range of a block (FP 4 ×
FP 8), then (2) a per-block E 4 M 3 scale moves the values within a block into FP 4 representable range.
Appendix B describes the quantization and scaling strategy in more detail.
1 Floating-point types are denoted as E𝑥M𝑦 and consist of one sign bit, 𝑥 exponent bits, and 𝑦 mantissa bits.
© 2 0 2 5 NVIDIA. All rights reserved. 2

## Page 3

Pretraining Large Language Modelswith NVFP 4
Insummary,NVFP 4’sdesignimprovementsover MXFP 4 increasetheaccuracyofoutlierswhileminimizing
the amount of small values being quantized to zero. These numerical advances (smaller block size and
more precise scaling) give NVFP 4 a clear advantage over MXFP 4, resulting in consistently better training
behavior. We discuss training results comparing these two formats in Section 5.
FP 8 scaling
factor 1 6 FP 4 elements
2 8 6 0.5 -2 -4 1 0 3 -1 2 4 -3 0.5 -1 2 0 4
Block amax
NVFP 4 block
Metadata Tensor Data
Figure 1 | A 1 6×3 2 matrix stored in NVFP 4 format. Each block contains sixteen contiguous FP 4 elements
(gray and green) along with a single FP 8 scale factor (yellow). The element with the largest magnitude in
each block (green) is scaled to the FP 4 maximum representable value and can be recovered using the
block scale factor. A per-tensor FP 3 2 scale factor (not shown) is also applied.
Table 1 | NVIDIA Blackwell Tensor Cores.
Format Element Scale Block Speedup vs. BF 1 6
### Gb 2 0 0 Gb 3 0 0
### Mxfp 8 E 5 M 2/E 4 M 3 Ue 8 M 0 3 2 2× 2×
### Mxfp 6 E 3 M 2/E 2 M 3 Ue 8 M 0 3 2 2× 2×
### Mxfp 4 E 2 M 1 Ue 8 M 0 3 2 4× 6×
### Nvfp 4 E 2 M 1 E 4 M 3 1 6 4× 6×
Hardware support via Tensor Cores: NVIDIA Blackwell GPUs provide native support for general
matrix multiplications (GEMMs) for a wide range of microscaling formats – MXFP 8, MXFP 6, MXFP 4,
NVFP 4 – as summarized in Table 1. Tensor Cores read narrow precision inputs along with 8-bit scale
factors for each block of 1 6 or 3 2 elements. Tensor Cores compute partial dot-products over the block,
multiply each partial product by the corresponding scale factors to descale the inputs scaled during
quantization, and accumulate the partial results in higher precision to produce the final dot-product
in FP 3 2. Further, Blackwell GPUs have native support for several rounding modes including round-
to-nearest-even and stochastic rounding for FP 4 conversion instructions. Tensor Cores deliver FP 4
computations at 2× (on GB 2 0 0 chips) and 3× (on GB 3 0 0 chips) higher math throughput rates compared
to FP 8. Memory usage is also approximately halved when using FP 4 operands compared to FP 8. As
a result, FP 4 could offer significant speedups for LLM training when GEMMs make up a substantial
portion of training time.
© 2 0 2 5 NVIDIA. All rights reserved. 3

## Page 4

Pretraining Large Language Modelswith NVFP 4
3. Training with NVFP 4
We report training results for a 1 2 B-parameter hybrid Mamba-Transformer model trained on 1 0 T tokens
with NVFP 4 precision and compare the results against an FP 8 reference model.
Model and training setup: We consider a hybrid Mamba-Transformer model architecture used in the
recently introduced Nemotron-H family of models (NVIDIA, 2 0 2 5 b,a). These models consist of a mixture
of Mamba-2, Self-Attention, and FFN blocks. We use the same architecture as the Nemotron-Nano-1 2 B-
v 2-Base model (a 1 2 B-parameter model from the Nemotron-H family (NVIDIA, 2 0 2 5 b)), which has been
shown to achieve competitive accuracies across multiple benchmarks. We train this model on 1 0 T tokens
with a Warmup-Stable-Decay (Hu et al., 2 0 2 4) learning rate schedule, where the learning rate is constant
through the first 8 0% of training and then decayed over the last 2 0%. Appendix A.1 has more details on
the model configuration.
We pretrain the model in NVFP 4 using the methodology described in Section 4. To compare the loss and
accuracies on downstream tasks, we pretrain an FP 8 baseline following the methodology in (Deep Seek-AI
et al., 2 0 2 4; NVIDIA, 2 0 2 5 b).
1.5
1.4
1.3
1.2
1.1
0 1 2 3 4 5 6 7 8 9 1 0
ssol
noitadila V
### Nvfp 4
### Fp 8
Start of learning rate annealing
(2 0% before end of training)
Transition from Phase 1 to Phase 2 data
Transition from Phase 2 to Phase 3 data
Tokens (in trillions)
Figure 2 | Validation loss of NVFP 4 and FP 8 pretraining for the 1 2 B model using 1 0 T tokens.
Pretraining results: Figure 2 shows that the validation loss of NVFP 4 closely tracks its FP 8 counterpart
throughout training. During the stable phase of training, the relative loss error of NVFP 4 remains
consistently below 1%, and widens to slightly above 1.5% as the learning rate is decayed towards the end
of training. This indicates that the training dynamics of NVFP 4 closely follows FP 8, with only a small
divergence appearing late in training. Note that the change in the slope of the loss curve at 8 T tokens
stems from the learning rate decay. Additionally, the small jump in loss at 9 T tokens corresponds to the
change in the dataset blend. Appendix A.1 has more details on the used dataset blend.
Despite the small gap in loss, downstream task accuracies remain largely unaffected. Figure 3 shows
NVFP 4 matching FP 8 on downstream evaluations over the duration of training. This trend holds across a
wide range of domains, including knowledge-intensive reasoning, mathematics, coding, and commonsense
reasoningtasks. Table 2 providesamorecomprehensiveview, confirmingthat NVFP 4 achievescomparable
accuracy to FP 8 across most individual benchmarks. The exception is the coding task, where NVFP 4 falls
slightly behind. We suspect the difference could be due to noisy evaluations: MBPP+ accuracy drops
on the very final checkpoint evaluation and choosing another checkpoint could potentially lead to better
accuracy for this task.
In scenarios where minimizing loss is critical, the gap can be reduced by transitioning to higher precision
© 2 0 2 5 NVIDIA. All rights reserved. 4

## Page 5

Pretraining Large Language Modelswith NVFP 4
6 5
5 5
4 5
3 5
2 5
0 1 2 3 4 5 6 7 8 9 1 0
ycarucc A
MMLU Pro COT (5-shot)
Tokens (trillions)
8 0
7 5
7 0
6 5
6 0
5 5
0 1 2 3 4 5 6 7 8 9 1 0
ycarucc A
8 5
7 5
6 5
5 5
4 5
3 5
2 5
0 1 2 3 4 5 6 7 8 9 1 0
MMLU (5-shot)
Tokens (trillions)
ycarucc A
### Math 5 0 0
Tokens (trillions)
6 5
5 5
4 5
3 5
2 5
0 1 2 3 4 5 6 7 8 9 1 0
ycarucc A
MBPP+ (Pass@1)
Tokens (trillions)
### Fp 8 Nvfp 4
Figure 3 | Task accuracy of NVFP 4 versus FP 8 measured throughout 1 0 T tokens of pretraining.
during the final stages of training. In particular, changing precision from NVFP 4 to BF 1 6 (or potentially,
MXFP 8) during the decay phase mitigates the loss gap, as explained later in Appendix D. This implies
most of the training can be executed in NVFP 4 (with a small amount of training in higher precision) to
achieve losses that are closer to the FP 8 baseline.
These results confirm that NVFP 4 training remains stable over long token horizons, preserving accuracy
relative to higher-precision baselines, and demonstrate that our NVFP 4 training methodology offers a
practical pathway for scalable 4-bit training.
Table 2 | Accuracy of the 1 2 B model for FP 8 and NVFP 4 pretraining. Evaluations are done in BF 1 6.
Task FP 8 NVFP 4
Task FP 8 NVFP 4
General 6 8.9 9 6 9.8 2
Code 5 9.5 2 5 6.6 7
### Mmlu 7 7.3 6 7 6.5 7
Human Eval+ 5 9.9 3 5 7.4 3
MMLU-Pro 5-shot 6 2.6 2 6 2.5 8
### Mbpp+ 5 9.1 1 5 5.9 1
AGIEval English Co T 6 7.0 1 7 0.3 1
Commonsense Understanding 7 7.2 9 7 6.7 5
Math 8 6.2 0 8 6.8 8
ARC Challenge 9 1.8 1 9 1.8 1
GSM 8 k Co T 8 9.0 8 9 2.2 7
Hella Swag 8 3.8 3 8 3.0 9
### Math 8 3.3 2 8 1.4 8
Open Book QA 4 7.6 0 4 7.4 0
Multilingual 7 7.9 3 8 0.2 4
### Piqa 8 2.6 4 8 2.7 0
Global MMLU 7 4.0 0 7 4.9 4
Winogrande 8 0.5 8 7 8.7 7
### Mgsm 8 1.8 7 8 5.5 3
4. Training Methodology
In addition to the NVFP 4 data type, our approach incorporates several key techniques to enable effective
4-bit training. These include (1) retention of specific numerically sensitive layers in higher precision, (2)
Random Hadamard transforms to manage block-level outliers, (3) two-dimensional (2 D) block scaling
applied to weights for consistency between forward and backward passes, and (4) stochastic rounding
to ensure unbiased quantized gradients. While smaller models trained on shorter token horizons may
© 2 0 2 5 NVIDIA. All rights reserved. 5

## Page 6

Pretraining Large Language Modelswith NVFP 4
1.8 0
1.7 5
1.7 0
1.6 5
0 1 2 3 4 5 6 7
ssol
gniniar T
0.0
-0.5
-1.0
-1.5
-2.0
3.4 3.5 3.6 3.7 3.8 3.9 4
Tokens (in trillions)
)%(
8 pf
morf
ecnereffid
evitale R
### Nvfp 4
NVFP 4 without SR
NVFP 4 without RHT
NVFP 4 without 2 D W
NVFP 4 with last
four blocks in BF 1 6
Tokens (in trillions)
Figure 4 | Ablations on the 1 2 B model trained for 1 0 T tokens. Ablation studies start from the model
trained up to 3.4 3 T tokens using NVFP 4 except in the first two and last eight blocks, and systematically
remove one methodology component at a time: stochastic rounding (SR), Random Hadamard Transforms
(RHT), two-dimensional scaling (2 D), and fewer blocks in BF 1 6. Relative difference is defined as (FP 8 -
experiment) / FP 8, where a negative difference means the experiment is worse.
not require all of these techniques, we find that each component is essential for ensuring convergence
and stability of the 1 2 B model training over the 1 0 T-token horizon. Figure 4 illustrates this via ablation
studies: starting with the full training methodology described below, we remove one component at a time
and observe that eliminating any of them leads to worse convergence.
In short, our recommendation for NVFP 4 training is:
1. Keep a few sensitive linear layers in higher precision (1 5% of the network, with the majority of high
precision layers at the end of the network).
2. Apply Random Hadamard transforms of size 1 6×1 6 to inputs of weight gradient GEMMs.
3. Use two-dimensional (2 D) scaling over 1 6×1 6 blocks for weights, and one-dimensional scaling over
1×1 6 blocks for activations and gradients.
4. Use stochastic rounding for gradients and round-to-nearest-even for weights and activations.
In the rest of this section, we discuss each component of the training methodology in detail and describe
the ablation study presented in Figure 4. Additional ablations are reported in Appendix E to support our
choices.
4.1. Mixed precision
We adopt a mixed-precision strategy for FP 4 training. The majority of computations, specifically the
GEMM operations within linear (fully-connected) layers, are carried out in FP 4. As illustrated in Figure 5,
each linear layer has three underlying GEMMs: a GEMM in the forward pass (Fprop), and separate
GEMMs to compute activation gradients (Dgrad) and weight gradients (Wgrad) in the backward pass.
GEMM operations consume FP 4 tensors as inputs and produce outputs in BF 1 6 or FP 3 2.
Linear layers: Although linear layers are typically computed in narrower precisions, we observe that
some linear layers are more sensitive to FP 4 than others. In particular, training diverges when every
linear layer is quantized to FP 4. We observe from our ablation studies (see Appendix E.2) that the final
few linear layers in our models cause training to diverge, since they require more dynamic range and
© 2 0 2 5 NVIDIA. All rights reserved. 6

## Page 7

Pretraining Large Language Modelswith NVFP 4
From
layer i -1 BF 1 6 Quantize NVFP 4 FPROP BF 1 6 Activation
Activation to NVFP 4 (NVFP 4 GEMM) To layer i + 1
Transpose Transpose
### Nvfp 4
2 D Block Hadamard Hadamard
Quantize to Transform Transform
### Nvfp 4 Bf 1 6 Bf 1 6
Quantize Quantize
Transpose FP 3 2 to NVFP 4 to NVFP 4
Weights with SR
### Nvfp 4 Nvfp 4
### Fp 3 2
NVFP 4 BF 1 6 From
Optimizer BF 1 6
(NV
W
FP
G
4
R
G
A
E
D
### M M)
Activation layer i + 1
Gradient
Quantize
Activation Gradient BF 1 6 DGRAD NVFP 4 to NVFP 4
To layer i -1 (NVFP 4 GEMM) with SR
Figure 5 | Illustration of compute flow for a NVFP 4 quantized linear layer. All GEMM operations quantize
their inputs to NVFP 4.
mantissa than FP 4 provides. Based on these findings, we recommend leaving a small fraction of the final
layers (e.g., fewer than 1 5%) in BF 1 6 or MXFP 8 for better training convergence.
For the 1 2 B model, we chose a conservative configuration, keeping the first two blocks in addition to the
final eight blocks (FFNs or Mamba-2, each has 2 linear layers) in BF 1 6, representing 1 6% of the linear
layers in the network in high precision. However, Figure 4 indicates that convergence remains stable even
when only the final four blocks are left in higher precision, suggesting that a larger portion of the model
could have been safely trained in FP 4.
Attention, embedding, non-linear layers, and other tensors: To ensure numerical stability during
training, we retain the original precision (e.g., BF 1 6 or FP 3 2) for embeddings, the output projection head,
normalization layers, non-linearities, and attention components, including softmax and the query-key and
attention score-value batched GEMMs. The main weights (stored by the optimizer), weight gradients
(used for gradient accumulation across microbatches and across data-parallel replicas), and optimizer
states are also kept in FP 3 2. Tensor parallel reductions are performed in BF 1 6 precision.
4.2. Random Hadamard Transforms
While microscaling reduces the dynamic range needed to represent tensor values, outliers can still have a
disproportionate impact (An et al., 2 0 2 5; Park et al., 2 0 2 5; Raman et al., 2 0 2 5; Dettmers et al., 2 0 2 2;
Xiao et al., 2 0 2 4) on FP 4 formats, degrading model accuracy. Random Hadamard transforms (Shah et al.,
2 0 2 4; Ashkboos et al., 2 0 2 5, 2 0 2 4; Tseng et al., 2 0 2 4, 2 0 2 5 a; Malinovskii et al., 2 0 2 4) address this by
redistributing outliers into an approximately Gaussian distribution, making them easier to represent in
narrower formats. Below we discuss the application of Random Hadamard transforms in FP 4 training.
GEMMs transformed: Random Hadamard transforms are typically applied on both GEMM inputs
so that the dot-product inverts each transform by the other operand due to orthogonality. More details
on their mechanics is discussed in Appendix C. Empirically, we observe that transforming Wgrad inputs
improves training for the 1 2 B model (e.g., Figure 4 shows that loss worsens after removing transformations
from Wgrad). On the other hand, Hadamard transforms show no measurable benefit for Fprop and Dgrad
at smaller scales (see Appendix E.4.1), likely because FP 4 already provides sufficient range. As a result,
we restrict Hadamard transforms to Wgrad inputs, though there may be cases where Fprop and Dgrad
would also benefit.
Hadamard matrix size: Random Hadamard transforms are implemented as matrix multiplications
between 𝑑×𝑑 Hadamard matrices and each tile of the tensor of equal size. The matrix size 𝑑 introduces
a trade-off between accuracy and performance. Larger matrices distribute outliers more effectively, by
© 2 0 2 5 NVIDIA. All rights reserved. 7

## Page 8

Pretraining Large Language Modelswith NVFP 4
spreading them over more values, but increase compute and memory costs. Matrices with too few entries
are less likely to reproduce a Gaussian distribution, harming FP 4 accuracy. At small scales, we observe
no measurable differences in convergence due to matrix size. At larger scales, such as the 1 2 B model, we
observe diminishing gains from Hadamard matrices beyond moderate sizes (see Appendix E.4.2), whereas
having too few matrix entries affects convergence. We believe this is in part due to larger models having
more outliers. We therefore choose a matrix size of 𝑑 = 1 6, which we find to have better convergence than
𝑑 = 4 and similar results as 𝑑 = 1 2 8.
Random sign vector: Random Hadamard transforms introduce randomness by multiplying with a
random diagonal sign vector that flips the signs for entire rows or columns. This reduces the chance that
“structured” outliers (e.g., tensor patterns aligned with the Hadamard basis) survive the transform. At
small scales, randomization has no impact on accuracy, and training remains stable with the standard
Hadamard transform. However, we find that randomization benefits larger models trained over longer
token horizons, as detailed in Appendix E.4.3. In our setup, we use a single random sign vector that is
shared across all linear layers throughout training. Our studies show no measurable impact from increasing
the number of random sign vectors.
4.3. 2 D scaling
During training, transform and scaling operations apply along the dot-product dimension, causing tensors
to be transformed and scaled differently in the forward (along rows) and backward (along columns)
passes. This occurs because the backward pass transposes the tensors, which changes the dot-product
dimension. As a result, the same tensor can have two distinct quantized representations, effectively
breaking the chain rule since backpropagation no longer differentiates the same function used in the
forward pass. More precisely, the backward update 𝜕𝑥 = 𝑤𝑇 𝜕𝑦 computes a gradient for a different
bprop
function 𝑦 = 𝑤 𝑥 than used in the forward pass 𝑦 = 𝑤 𝑥 when 𝑤 ≠ 𝑤 . We
bprop bprop fprop fprop fprop bprop
hypothesize that chain rule violations in the weights contribute to reduced model accuracy.
Block scaling: To mitigate this issue, we propose a two-dimensional (2 D) block scaling method that
ensures consistent quantization in both forward and backward passes. For weights, elements are grouped
and scaled in 1 6×1 6 blocks (i.e., 1 6 input channels by 1 6 output channels) similar to Deep Seek-AI et al.
(2 0 2 4). 2 D block scales are replicated for each of the 1×1 6 blocks when being passed into Tensor Cores,
and continue to leverage an FP 3 2 per-tensor scale. Activations and gradients use the standard NVFP 4
scaling (i.e., 1×1 6 blocks), since finer-grained scaling improves quantization accuracy. While activation
quantization also presents a chain rule concern, we observe that training is less sensitive to inconsistencies
in activation tensors than weight tensors (Appendix E.5 discusses this further). Weights are also more
tolerant to the scale granularity because they can adapt to the FP 4 values. As illustrated in Figure 4,
maintaining consistent quantized weights leads to improved training loss for the 1 2 B model.
Random Hadamard transforms: Similar to scaling, Random Hadamard transforms applied along
the dot-product dimension introduce inconsistency after quantization (i.e., different transformations will
result in different quantized values) and, therefore, are not applied on the weight tensors. As a result,
transformed activations and gradients in weight-related GEMMs can no longer be inverted by transforming
the weight tensor, preventing Fprop and Dgrad from benefiting from the transformation. Therefore, we
restrict Hadamard transforms to the Wgrad tensors, which we find sufficient for training our models
(Appendix E.4.1).
4.4. Stochastic rounding
During quantization to FP 4, deterministic rounding (e.g., round-to-nearest-even) can introduce bias,
producing systematic errors due to mantissa distributions that favor rounding in a particular direction,
values underflowing to zero, or values saturating to the largest representable number. The effect of bias
is typically more pronounced in gradient tensors (Castro et al., 2 0 2 5; Tseng et al., 2 0 2 5 b; Chmiel et al.,
© 2 0 2 5 NVIDIA. All rights reserved. 8

## Page 9

Pretraining Large Language Modelswith NVFP 4
2 0 2 5, 2 0 2 3; Alistarh et al., 2 0 1 7), which can impact training convergence. To address this bias, we adopt
stochastic rounding during quantization of high precision values to FP 4. Stochastic rounding rounds
a value probabilistically to one of its two nearest representable numbers, with probabilities inversely
proportional to their distances. This prevents values from being consistently quantized in the same
direction, thereby reducing bias.
We observe that applying stochastic rounding to gradient tensors is essential for convergence in the 1 2 B
model, as illustrated in Figure 4. Other tensors in the backward pass do not benefit from stochastic
rounding, reinforcing that gradients are the primary source of bias (see Appendix E.3). Moreover, applying
stochastic rounding to the forward pass tensors is detrimental, as it amplifies quantization error relative
to nearest rounding (Castro et al., 2 0 2 5).
5. NVFP 4 and MXFP 4
As discussed earlier, there are two FP 4 microscaling formats on NVIDIA Blackwell – MXFP 4 and NVFP 4.
In this section, we compare training behavior when using these two formats.
0.0
-0.5
-1.0
-1.5
-2.0
-2.5
-3.0
0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1
)%(
6 1 fb
morf
ecnereffid
evitale R
1.2 8 0
1 T tokens
1.2 7 5
1.2 5 T tokens 1.2 7 0
1.3 6 T tokens
1 T tokens
1.2 6 5
### Nvfp 4
MXFP 4 1.5 T tokens
1.2 6 0
0.8 1 1.2 1.4 1.6
Tokens (in trillions)
(a) Relative difference between training loss of BF 1 6
(baseline) and NVFP 4 and MXFP 4 pretraining.
ssol
noitadila V
### Nvfp 4
### Mxfp 4 Mxfp 4
### Nvfp 4
Tokens (in trillions)
(b)Finalvalidationlossfor NVFP 4 and MXFP 4 pretrain-
ing with different number of tokens.
Figure 6 | NVFP 4 vs MXFP 4 comparisons: (a) training-loss difference; (b) validation perplexity across
token budgets.
Model and training setup: We consider an 8-billion parameter (8 B) model based on the hybrid
Mamba-Transformer architecture. The model is trained on 1 trillion tokens with the same dataset as used
for the 1 2 B model. Training consists of two phases of data-blending, between the first 6 0% and last 4 0%
of training. The model and training details are described in Appendix A.2.
The reference model is pretrained in BF 1 6. FP 4 pretraining follows the training methodology described
in Section 4 with MXFP 4 and NVFP 4 as the respective data formats. For MXFP 4, we adopt a Random
Hadamard transform size of 𝑑 = 3 2 for Wgrad inputs, to align with the MXFP 4 block size. In both the
settings, the last eight blocks (either FFNs or Mamba-2) are kept in BF 1 6, comprising about 1 5% of the
model.
Results: Figure 6 a demonstrates that NVFP 4 pretraining converges to a better loss than MXFP 4.
Specifically, MXFP 4 has a relative error of around 2.5% compared to 1.5% for NVFP 4. To close the
gap with NVFP 4, we extend MXFP 4 pretraining with additional tokens (varying between 1 T and 1.5 T
total tokens). Figure 6 b illustrates the final loss obtained as a function of number of tokens used during
pretraining. We observe that MXFP 4 matches NVFP 4 loss when trained on 3 6% more tokens (i.e., using
1.3 6 T instead of 1 T tokens). This translates to a considerable increase in training time for MXFP 4,
highlighting the benefits of NVFP 4. Future studies should evaluate scaling laws for these formats on
different parameter counts and token horizons.
© 2 0 2 5 NVIDIA. All rights reserved. 9

## Page 1 0

Pretraining Large Language Modelswith NVFP 4
6. Conclusions
We have demonstrated that large-scale pretraining with NVFP 4 is both stable and accurate when paired
with a targeted methodology designed to improve training stability and convergence through techniques
such as 2 D weight scaling, Random Hadamard transforms, stochastic rounding, and others described in
this technical report. Using this approach, a 1 2 B hybrid Mamba-Transformer model was trained on 1 0
trillion tokens, with loss and downstream accuracy closely tracking the FP 8 baseline. This establishes the
first public evidence of sustained 4-bit pretraining at multi-trillion-token scale.
In side-by-side experiments, NVFP 4 reached comparable loss with fewer tokens than MXFP 4, indicating
efficiency gains without sacrificing accuracy. These comparisons provide an initial view into the memory
and compute efficiency benefits, as well as the convergence trade-offs, of different FP 4 formats during
pretraining.
Future work will further characterize NVFP 4’s pretraining performance relative to other formats, while
refining the methodology to quantize all linear layers without impacting convergence, reducing remaining
high-precision layers, and extending NVFP 4 to attention and communication paths. We also plan to
explore its use in post-training scenarios and evaluate it on larger models, longer token horizons, and
additional architectures such as mixture-of-experts. NVFP 4 training on Blackwell is now fully supported
via a recent update to Transformer Engine.
© 2 0 2 5 NVIDIA. All rights reserved. 1 0

## Page 1 1

Pretraining Large Language Modelswith NVFP 4
Contributors
Numerics, Evaluations: Anjulie Agrusa, Mike Chrzanowski, Eric Chung, Steve Dai, Bita Darvish
Rouhani, Carlo del Mundo, Brucek Khailany, Mikail Khona, Nick Knight, Ben Lanir, Simon Layton,
Daniel Lo, Paulius Micikevicius, Asit Mishra, Deepak Narayanan, Chao Ni, Mostofa Patwary, Sweta
Priyadarshi, Yigong Qin, Oleg Rybakov, Charbel Sakr, Sanjeev Satheesh, Mohammad Shoeybi, Michael
Siu, Darko Stosic, Dusan Stosic, Bor-Yiing Su, Nima Tajbakhsh, Aditya Vavre, Rangharajan Venkatesan,
Roger Waleffe, Qiyu Wan, Mengdi Wang, Lizzie Wei, Hao Wu, Keith Wyss, Jinze Xue
SW Support, Performance: Felix Abecassis, Anjulie Agrusa, Michael Andersch, Jinhang Choi, Victor
Cui, Carlo del Mundo, Burc Eryilmaz, Abhinav Goel, Oleg Goncharov, Robert Hesse, Herbert Hum,
Ronny Krashinsky, Tim Moon, Yigong Qin, Xiaowei Ren, Kirthi Shankar, Frank Sun, Przemek Tredak,
Evgeny Tsykunov, Qiyu Wan, Lizzie Wei, Evan Wu, Keith Wyss, Jinze Xue, Charlene Yang, Yujia Zhai,
Jingyang Zhu, Zhongbo Zhu
Infrastructure: Dong Ahn, Stefania Alborghetti, Sivakumar Arayandi, Alexis Bjorlin, Aaron Blakeman,
Evan Briones, Carlo del Mundo, Deena Donia, Henry Estela, Yugi Guvvala, Russell J. Hewett, Alex
Kondratenko, Deepak Narayanan, Abhijit Paithankar, Satish Pasumarthi, Ankit Patel, Ashwin Poojary,
Gargi Prasad, Oleg Rybakov, Stas Sergienko, Pasha Shamis, Nishant Sharma, Misha Smelyanskiy, Shelby
Thomas, Evgeny Tsykunov, Gandhi Vaithilingam, Roger Waleffe, Hexin Wang, Ning Xu, Ruoxi Zhang
Leadership: Jonah Alben, Ian Buck, Bryan Catanzaro, Eric Chung, Ujval Kapasi, Michael Lightstone,
Mohammad Shoeybi
© 2 0 2 5 NVIDIA. All rights reserved. 1 1

## Page 1 2

Pretraining Large Language Modelswith NVFP 4
References
Dan Alistarh, Demjan Grubic, Jerry Li, Ryota Tomioka, and Milan Vojnovic. QSGD: Communication-
Efficient SGD via Gradient Quantization and Encoding. In Advances in Neural Information Processing
Systems (Neur IPS), volume 3 0, 2 0 1 7.
Eduardo Alvarez, Omri Almog, Eric Chung, Simon Layton, Dusan Stosic, Ronny
Krashinsky, and Kyle Aubrey. Introducing NVFP 4 for Efficient and Accu-
rate Low-Precision Inference, 2 0 2 5. URL https://developer.nvidia.com/blog/
introducing-nvfp 4-for-efficient-and-accurate-low-precision-inference/.
Yongqi An, Xu Zhao, Tao Yu, Ming Tang, and Jinqiao Wang. Systematic Outliers in Large Language
Models, 2 0 2 5. URL https://arxiv.org/abs/2 5 0 2.0 6 4 1 5.
Saleh Ashkboos, Amirkeivan Mohtashami, Maximilian L. Croci, Bo Li, Pashmina Cameron, Martin Jaggi,
Dan Alistarh, Torsten Hoefler, and James Hensman. Qua Rot: Outlier-Free 4-Bit Inference in Rotated
LLMs, 2 0 2 4. URL https://arxiv.org/abs/2 4 0 4.0 0 4 5 6.
Saleh Ashkboos, Mahdi Nikdan, Soroush Tabesh, Roberto L. Castro, Torsten Hoefler, and Dan Alistarh.
HALO: Hadamard-Assisted Lower-Precision Optimization for LLMs, 2 0 2 5. URL https://arxiv.org/
abs/2 5 0 1.0 2 6 2 5.
Roberto L. Castro, Andrei Panferov, Soroush Tabesh, Oliver Sieberling, Jiale Chen, Mahdi Nikdan, Saleh
Ashkboos, and Dan Alistarh. Quartet: Native FP 4 Training Can Be Optimal for Large Language
Models, 2 0 2 5. URL https://arxiv.org/abs/2 5 0 5.1 4 6 6 9.
Yuxiang Chen, Haocheng Xi, Jun Zhu, and Jianfei Chen. Oscillation-Reduced MXFP 4 Training for Vision
Transformers, 2 0 2 5. URL https://arxiv.org/abs/2 5 0 2.2 0 8 5 3v 2.
Brian Chmiel, Ron Banner, Elad Hoffer, Hilla Ben-Yaacov, and Daniel Soudry. Accurate Neural Training
with 4-Bit Matrix Multiplications at Standard Formats. In Proceedings of the 1 1 th International
Conference on Learning Representations, 2 0 2 3. Poster presentation.
Brian Chmiel, Maxim Fishman, Ron Banner, and Daniel Soudry. FP 4 All the Way: Fully Quantized
Training of LLMs, 2 0 2 5. URL https://arxiv.org/abs/2 5 0 5.1 9 1 1 5.
Deep Seek-AI, Aixin Liu, et al. Deep Seek-V 3 Technical Report. Technical Report, ar Xiv preprint
ar Xiv:2 4 1 2.1 9 4 3 7, 2 0 2 4. URL https://arxiv.org/abs/2 4 1 2.1 9 4 3 7.
Tim Dettmers, Mike Lewis, Younes Belkada, and Luke Zettlemoyer. LLM.int 8(): 8-bit Matrix Multiplica-
tion for Transformers at Scale, 2 0 2 2. URL https://arxiv.org/abs/2 2 0 8.0 7 3 3 9.
Steven Feng, Shrimai Prabhumoye, Kezhi Kong, Dan Su, Mostofa Patwary, Mohammad Shoeybi, and
Bryan Catanzaro. Maximize Your Data’s Potential: Enhancing LLM Accuracy with Two-Phase
Pretraining, 2 0 2 4. URL https://arxiv.org/abs/2 4 1 2.1 5 2 8 5.
Shengding Hu, Yuge Tu, et al. Mini CPM: Unveiling the Potential of Small Language Models with Scalable
Training Strategies, 2 0 2 4. URL https://arxiv.org/abs/2 4 0 4.0 6 3 9 5.
Vladimir Malinovskii, Andrei Panferov, Ivan Ilin, Han Guo, Peter Richtárik, and Dan Alistarh. Pushing
the Limits of Large Language Model Quantization via the Linearity Theorem, 2 0 2 4. URL https:
//arxiv.org/abs/2 4 1 1.1 7 5 2 5.
Paulius Micikevicius, Dusan Stosic, Neil Burgess, Marius Cornea, Pradeep Dubey, Richard Grisenthwaite,
Sangwon Ha, Alexander Heinecke, Patrick Judd, John Kamalu, Naveen Mellempudi, Stuart Oberman,
Mohammad Shoeybi, Michael Siu, and Hao Wu. FP 8 Formats for Deep Learning, 2 0 2 2. URL
https://arxiv.org/abs/2 2 0 9.0 5 4 3 3.
© 2 0 2 5 NVIDIA. All rights reserved. 1 2

## Page 1 3

Pretraining Large Language Modelswith NVFP 4
Asit Mishra, Dusan Stosic, Simon Layton, and Paulius Micikevicius. Recipes for Pre-training LLMs with
MXFP 8, 2 0 2 5. URL https://arxiv.org/abs/2 5 0 6.0 8 0 2 7.
NVIDIA. Nemotron-H: A Family of Accurate and Efficient Hybrid Mamba-Transformer Models, 2 0 2 5 a.
URL https://arxiv.org/abs/2 5 0 4.0 3 6 2 4.
NVIDIA. NVIDIA Nemotron Nano 2: An Accurate and Efficient Hybrid Mamba-Transformer Reasoning
Model, 2 0 2 5 b. URL https://arxiv.org/abs/2 5 0 8.1 4 4 4 4.
NVIDIA Blackwell. Architecture Technical Brief. https://resources.nvidia.com/
en-us-blackwell-architecture, 2 0 2 4.
Open-Compute-Project. OCP Microscaling Formats (MX) Specification Version 1.0, 2 0 2 3. URL https:
//www.opencompute.org/documents/ocp-microscaling-formats-mx-v 1-0-spec-final-pdf.
Jungwoo Park, Taewhoo Lee, Chanwoong Yoon, Hyeon Hwang, and Jaewoo Kang. Outlier-Safe Pre-
Training for Robust 4-Bit Quantization of Large Language Models, 2 0 2 5. URL https://arxiv.org/
abs/2 5 0 6.1 9 6 9 7.
Rahul Raman, Khushi Sharma, and Sai Qian Zhang. Rethinking the Outlier Distribution in Large
Language Models: An In-depth Study, 2 0 2 5. URL https://arxiv.org/abs/2 5 0 5.2 1 6 7 0.
Bita Darvish Rouhani, Ritchie Zhao, Ankit More, Mathew Hall, Alireza Khodamoradi, Summer Deng,
Dhruv Choudhary, Marius Cornea, Eric Dellinger, Kristof Denolf, Stosic Dusan, Venmugil Elango,
Maximilian Golub, Alexander Heinecke, Phil James-Roxby, Dharmesh Jani, Gaurav Kolhe, Martin
Langhammer, Ada Li, Levi Melnick, Maral Mesmakhosroshahi, Andres Rodriguez, Michael Schulte,
Rasoul Shafipour, Lei Shao, Michael Siu, Pradeep Dubey, Paulius Micikevicius, Maxim Naumov, Colin
Verrilli, Ralph Wittig, Doug Burger, and Eric Chung. Microscaling Data Formats for Deep Learning,
2 0 2 3. URL https://arxiv.org/abs/2 3 1 0.1 0 5 3 7.
Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, and Tri Dao. Flash Attention-
3: Fast and Accurate Attention with Asynchrony and Low-precision, 2 0 2 4. URL https://arxiv.org/
abs/2 4 0 7.0 8 6 0 8.
Albert Tseng, Jerry Chee, Qingyao Sun, Volodymyr Kuleshov, and Christopher De Sa. Qu IP#: Even
Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks, 2 0 2 4. URL https:
//arxiv.org/abs/2 4 0 2.0 4 3 9 6.
Albert Tseng, Qingyao Sun, David Hou, and Christopher De Sa. QTIP: Quantization with Trellises and
Incoherence Processing, 2 0 2 5 a. URL https://arxiv.org/abs/2 4 0 6.1 1 2 3 5.
Albert Tseng, Tao Yu, and Youngsuk Park. Training LLMs with MXFP 4, 2 0 2 5 b. URL https://arxiv.
org/abs/2 5 0 2.2 0 5 8 6.
Ruizhe Wang, Yeyun Gong, Xiao Liu, Guoshuai Zhao, Ziyue Yang, Baining Guo, Zhengjun Zha, and
Peng Cheng. Optimizing Large Language Model Training Using FP 4 Quantization, 2 0 2 5. URL
https://arxiv.org/abs/2 5 0 1.1 7 1 1 6.
Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, and Song Han. Smooth Quant:
Accurate and Efficient Post-Training Quantization for Large Language Models, 2 0 2 4. URL https:
//arxiv.org/abs/2 2 1 1.1 0 4 3 8.
Biao Zhang and Rico Sennrich. Root Mean Square Layer Normalization, 2 0 1 9. URL https://arxiv.
org/abs/1 9 1 0.0 7 4 6 7.
Jiecheng Zhou, Ding Tang, Rong Fu, Boni Hu, Haoran Xu, Yi Wang, Zhilin Pei, Zhongling Su, Liang Liu,
Xingcheng Zhang, and Weiming Zhang. Towards Efficient Pre-training: Exploring FP 4 Precision in
Large Language Models, 2 0 2 5. URL https://arxiv.org/abs/2 5 0 2.1 1 4 5 8.
© 2 0 2 5 NVIDIA. All rights reserved. 1 3

## Page 1 4

Pretraining Large Language Modelswith NVFP 4
Appendix
A. Models
We evaluate three model variants throughout this technical report: two hybrid Mamba-Transformer
architectures at 1 2 B and 8 B scales, and a Transformer variant at 1.2 B scale. The 1 2 B model is used as the
primary architecture to validate NVFP 4 training method while the 8 B hybrid model is used to compare
NVFP 4 against MXFP 4. The 1.2 B model is used for several ablation studies. This section describes the
architectural details, datasets, and training schedules used for each model.
A.1. 1 2 B hybrid Mamba-Transformer
Model architecture: Table 3 summarizes the configuration for the 1 2 B hybrid Mamba-Transformer
architecture. The model has 6 2 blocks with 6 Self-Attention, 2 8 FFNs, and 2 8 Mamba-2 blocks (each
block has 2 linear layers). Mamba-2 blocks have 8 groups, state dimension of 1 2 8, head dimension of 6 4,
expansion factor of 2, and convolution window size of 4. Squared Re LU activations are used for FFN
blocks, RMSNorm (Zhang & Sennrich, 2 0 1 9) for the normalization layers, and separate embedding and
output layer weights. The model does not have any position embeddings, dropout, or biases for linear
layers. Residual skip connections are added to each block.
Table 3 | Summary of the 1 2 B Nemotron-H hybrid Mamba–Transformer architecture.
Number of Model FFN Q KV State Mamba
blocks dimension dimension heads heads dimension groups
6 2 5 1 2 0 2 0 4 8 0 4 0 8 1 2 8 8
Dataset: For the pretraining data, we use a corpus of high-quality curated and synthetic dataset
comprising of 1 0 trillion tokens based on NVIDIA (2 0 2 5 b), with data mixtures consisting of general web
crawl data, wikipedia, math, code, academic data, crawl++, multilingual, and synthetic SFT-style data.
Pretraining uses a phased data-blending approach (Feng et al., 2 0 2 4), where the first phase covers 7 0%
of training with a data mixture that promotes diversity in the data, while the second and third phases
primarily consist of high-quality datasets and span the last 2 0% and 1 0% of training, respectively.
Hyperparameters: The model is trained on 1 0 trillion tokens using a sequence length of 8 1 9 2 and batch
size of 7 3 6. The WSD schedule has a constant learning rate of 4.5·1 0−4 that decays to 4.5·1 0−6 over the
last 2 0% of training. Adam parameters are 𝛽 = 0.9 and 𝛽 = 0.9 5, and weight decay is set to 0.1.
1 2
Precisions: The reference model is trained in FP 8 following the methodology in NVIDIA (2 0 2 5 b).
Specifically, all linear layers are computed in E 4 M 3, except the linear layers in the first block and the last
two blocks which are left in BF 1 6. Scale factors apply on 1 2 8×1 2 8 blocks for weights and 1×1 2 8 blocks for
activations and gradients. They are computed online for each block, stored in FP 3 2, and applied before
quantizing the tensor into the FP 8 format. Precisions for other operations are the same as in Section 4.1.
For NVFP 4, we follow the method described in Section 4. All linear layers are computed in NVFP 4,
except the linear layers in the first two blocks and the last eight blocks (FFNs or Mamba-2) which are left
in BF 1 6. This accounts for 1 6% of the total linear layers kept in high precision.
A.2. 8 B hybrid Mamba-Transformer
Model architecture: The 8 B hybrid Mamba-Transformer has a similar architecture as the 1 2 B hybrid
model. Table 4 summarizestheconfigurationforthe 8 Bmodel. Thismodelhas 5 2blocks: 4 Self-Attention,
2 4 FFNs, and 2 4 Mamba-2 blocks. The model hidden dimension is 4 0 9 6, FFN hidden dimension is 2 1 5 0 4,
and Grouped-Query Attention has 3 2 query heads along with 4 key-value heads. Mamba-2 blocks have 8
© 2 0 2 5 NVIDIA. All rights reserved. 1 4

## Page 1 5

Pretraining Large Language Modelswith NVFP 4
groups, state dimension of 1 2 8, head dimension of 6 4, expansion factor of 2, and convolution window size
of 4.
Table 4 | Summary of the 8 B Nemotron-H hybrid Mamba–Transformer architecture.
Number of Model FFN Q KV State Mamba
blocks dimension dimension heads heads dimension groups
5 2 4 0 9 6 2 1 5 0 4 3 2 4 1 2 8 8
Hyperparameters: The model is trained on 1 trillion tokens from the same dataset used for the 1 2 B
model. A batch size of 7 6 8 is used with only two phases of data-blending, split between the first 6 0% and
last 4 0% of training. The sequence length is 8 1 9 2 and the WSD schedule uses a constant learning rate
of 8.0·1 0−4 that decays to 8.0·1 0−6 over the last 1 5% of training. Adam parameters are 𝛽 = 0.9 and
1
𝛽 = 0.9 5, and weight decay is set to 0.1.
2
Precisions: The reference model is trained in BF 1 6. For NVFP 4, we follow the methodology described
in Section 4. All linear layers are computed in NVFP 4, except for the linear layers in the last eight blocks
(FFNs or Mamba-2) which are left in BF 1 6.
A.3. 1.2 B Transformer
Model architecture: The 1.2 B model follows the standard Transformer architecture. Details on the
model configuration are summarized in Table 5. The model has 2 0 transformer blocks, each comprising of
Self-Attention and FFN blocks. The model hidden dimension is 2 0 4 8, FFN hidden dimension is 6 1 4 4, and
Self-Attention has 1 6 query heads and 8 key and value heads. FFN blocks use squared Re LU activations.
The model uses Ro PE embeddings and does not have any dropout or biases for linear layers. Residual
skip connections are added to each of the transformer blocks.
Table 5 | Summary of the 1.2 B Nemotron Transformer architecture.
Number of Model FFN Head Q KV
blocks dimension dimension dimension heads heads
2 0 2 0 4 8 6 1 4 4 1 2 8 1 6 8
Hyperparameters: The model is trained on 1 trillion tokens with the same dataset as for the 8 B model,
using two phases of data-blending. The model is trained with a sequence length of 8 1 9 2 and batch size
of 7 6 8. The WSD schedule starts with a learning rate of 1.2·1 0−3 for 8 5% of training that decays to
1.2·1 0−5 for the last 1 5% of training.
Precisions: Thereferencemodelistrainedin BF 1 6. For NVFP 4,weperformablationsonthemethodology
from Section 4. Linear layer tensors are converted to NVFP 4. Precisions for other operations are the
same as in Section 4.1.
B. NVFP 4 Quantization Procedure
The procedure for converting a tensor from higher precision (FP 3 2 or BF 1 6) to NVFP 4 is described below.
Given a tensor 𝑥, each block 𝑏 of contiguous high-precision values 𝑥 , 𝑖 ∈ 𝑏 is quantized to FP 4. Prior to
𝑖
quantization, values are scaled using a two-level scaling strategy: first, a global FP 3 2 tensor-level scale
factor moves all the values within a tensor into representable range of a block (FP 4 × FP 8); second, a
local block-level scale factor moves the values 𝑥 within a block into FP 4 representable range.
𝑖
© 2 0 2 5 NVIDIA. All rights reserved. 1 5

## Page 1 6

Pretraining Large Language Modelswith NVFP 4
B.1. Global tensor-level scaling
The global encode scale is computed as:
6·4 4 8
𝑠 = (1)
𝑒𝑛𝑐
𝑎𝑚𝑎𝑥
𝑥
where 𝑎𝑚𝑎𝑥 = max(|𝑥 |) represents the absolute maximum value across the entire tensor 𝑥, 6 and 4 4 8 are
𝑥 𝑖
𝑖
the maximum representable magnitudes in the E 2 M 1 and E 4 M 3 formats, respectively. The corresponding
decode scale, 𝑠 = 1/𝑠 , gets stored in FP 3 2 for decoding the resulting values after the NVFP 4 GEMM
𝑑𝑒𝑐 𝑒𝑛𝑐
operation. Since the global scale is computed dynamically across the entire tensor, it induces an extra
pass through device memory: once to compute the global amax, and once to scale prior to conversion to
FP 4, as described later. However, the global scale could potentially span a smaller granularity (e.g., a row
or block of elements) to avoid additional round-trips through device memory.
B.2. Local block-level scaling
The local decode scales are chosen so the largest absolute value in each block, 𝑎𝑚𝑎𝑥 = max(|𝑥 |),
𝑏 𝑖
𝑖∈𝑏
normalizes to the FP 4 maximum representable:
𝑎𝑚𝑎𝑥
𝑏
### 𝑆 = (2)
𝑑𝑒𝑐,𝑏
6
Since the local decode scales must be stored in FP 8 for Tensor Cores, they are first multiplied by the
global encode scale before quantization:
𝑠 = e 4 m 3(𝑠 ·𝑠 ), (3)
𝑑𝑒𝑐,𝑏,𝑒 4 𝑚 3 𝑑𝑒𝑐,𝑏 𝑒𝑛𝑐
where the goal of 𝑠 is to remap the largest local decode scale, i.e., 𝑚𝑎𝑥(𝑠 ) = 𝑎𝑚𝑎𝑥 /6, to the
𝑒𝑛𝑐 𝑑𝑒𝑐,𝑏 𝑥
FP 8 maximum representable. We obtain the real local encode scale factor by inverting the quantized
local decode scale in higher precision and scaling it back to its original representable range, 𝑠 =
𝑒𝑛𝑐,𝑏
1/(fp 3 2(𝑠 )·𝑠 ). Inthisway, wetrytoensurethattheoriginalvaluecanberecoveredafterscaling
𝑑𝑒𝑐,𝑏,𝑒 4 𝑚 3 𝑑𝑒𝑐
with 𝑠 ·𝑠 ·𝑠 ≈ 1, since failing to do so can impact model accuracy. Round-to-nearest-even
𝑒𝑛𝑐,𝑏 𝑑𝑒𝑐 𝑑𝑒𝑐,𝑏,𝑒 4 𝑚 3
is used when computing the decode scale factor in Equation 3.
B.3. Conversion
Combining all of these together, each element 𝑥 in the block gets scaled by the local encode scale and
𝑖
quantized as
𝑥^ = 𝑞(𝑥 ·𝑠 ), (4)
𝑖 𝑖 𝑒𝑛𝑐,𝑏
where 𝑞(·) denotes the FP 4 quantization function. Beyond storing the quantized values 𝑥^ , the local
𝑖
and global decode scales, 𝑠 and 𝑠 , are also stored in memory and used during the matrix
𝑑𝑒𝑐,𝑏,𝑒 4 𝑚 3 𝑑𝑒𝑐
multiplication.
Tensor Core reads the local decode scales and applies them to partial dot-products computed over 𝑏
elements:
𝑠𝑥 ·𝑠𝑦 · ∑︁ (𝑥 ·𝑦 ),
𝑑𝑒𝑐,𝑏,𝑒 4 𝑚 3 𝑑𝑒𝑐,𝑏,𝑒 4 𝑚 3 𝑘 𝑘 (5)
𝑘∈𝑏
where 𝑥 and 𝑦 denote the two input operands. After the GEMM operation, the global decode scales 𝑠𝑥
𝑑𝑒𝑐
and 𝑠𝑦 are applied on the final output in a similar fashion.
𝑑𝑒𝑐
© 2 0 2 5 NVIDIA. All rights reserved. 1 6

## Page 1 7

Pretraining Large Language Modelswith NVFP 4
B.4. Remarks on MXFP 4 and NVFP 4 scale factor
MXFP 4 scale factors are restricted to powers-of-two, meaning values can not be scaled to fit perfectly
into the FP 4 representable range. After scaling, the block amax will either overflow the FP 4 maximum
representable and saturate, or round down to a smaller FP 4 sample. Since saturations have been observed
to cause convergence issues for MXFP 8 training (Mishra et al., 2 0 2 5), we typically round decode scale
factors up to prevent saturations.
This scalingstrategy can resultin some FP 4 samples being wasted while alsoreducing theutilized dynamic
range. As an example, consider a block of values with an absolute maximum value of 𝑎𝑚𝑎𝑥 = 3+𝛿, where
𝛿 represents a small increment. In order to move the block amax to the FP 4 maximum representable
number (i.e., ±6 for E 2 M 1), the decode scale factor is computed as 𝑠 = 𝑎𝑚𝑎𝑥/6 = 0.5 + 𝛿/6,
𝑑𝑒𝑐,𝑏
which rounds up to the next power-of-two, to 𝑠 = 1. After scaling, the block’s amax becomes
𝑑𝑒𝑐,𝑏,𝑢𝑒 8 𝑚 0
𝑎𝑚𝑎𝑥/𝑠 = 3+𝛿, which quantizes to 3 in FP 4. As a result, in the worst case, FP 4 is unable to
𝑑𝑒𝑐,𝑏,𝑢𝑒 8 𝑚 0
represent the samples at ±4 and ±6. This also reduces the dynamic range by nearly one binade, where
only log (3/0.5) = 2.5 8 binades are utilized instead of the full log (6/0.5) = 3.5 8 binades, where 0.5
2 2
represents the minimum positive non-zero magnitude in FP 4.
NVFP 4 overcomes this limitation with a more precise E 4 M 3 block scale, which maps the block amax
much closer to the FP 4 maximum representable number. This maximizes the FP 4 samples utilization and
preserves more of the dynamic range of FP 4.
C. Hadamard Transform Mechanics
Random Hadamard transforms applies an orthogonal rotation to the tensor being quantized, i.e., 𝑥′ =
𝑞(𝑥𝐻 ·𝑠), where 𝐻 is the Hadamard matrix, 𝑞(·) is the quantization function, and 𝑠 is the scale factor
computed in the rotated space 𝑥𝐻. The Hadamard matrix is defined by normalized matrices of the form
√
𝐻 = (1/ 2)𝐻 ⊗𝐻 with elements constrained to ±1. Given their orthogonal nature, they can be
𝑑 2 𝑑/2
applied to both operands of a matrix multiplication:
### 𝐶 = (𝐴𝐻)(𝐻𝑇𝐵) = 𝐴𝐵, (6)
where the transform from each operand gets inverted within the dot-product by 𝐻𝐻𝑇 = 𝐼.
Random Hadamard transforms introduce randomization into the transformation by left-hand multiplying
a 𝑑-dimensional diagonal random matrix, 𝑆 , with a Hadamard matrix, resulting in 𝐻 = 𝑆 𝐻 , where
𝑑 𝑑 𝑑
diagonal entries of 𝑆 are randomly chosen from {−1,1}. The entries in 𝑆 will flip the signs for different
𝑑 𝑑
rows of 𝐻 .
𝑑
We perform Hadamard transforms in a tiled approach by multiplying 𝐻, which has 𝑑×𝑑 matrix entries,
with an 𝑚×𝑘 tensor, where every 𝑑×𝑑 elements of the tensor are multiplied by 𝐻. The transform
involves 𝑚𝑘𝑑 multiply-adds and 𝑑 2 reads for the Hadamard matrix, which is a small cost when 𝑑 is much
smaller than the tensor dimensions 𝑚 or 𝑘. In this case, Hadamard transforms can be implemented as
batched matrix multiplications, which are limited by memory traffic from reading the input tensor when
using Tensor Cores, and can be fused with other layers to reduce round-trips to device memory.
D. Switching to Higher Precision
For situations where FP 4 training does not completely match the loss of higher precision training, we
observe that switching from FP 4 to higher precision towards the end of training can close the loss gap.
Figure 7 shows that loss matches the FP 8 baseline when precisions are switched after 8.2 T tokens (e.g., for
1 8% of training) and only slightly worse when switched after 1 0 T tokens (e.g., for less than 1% of training).
While switching precisions later in training fails to fully recover, presumably because the learning rate is
too low for the weight updates, it significantly reduces the portion of training not performed in FP 4. We
therefore recommend switching to high precision shortly before the onset of learning rate decay for full
loss recovery, or at the very end for notable loss improvements with minimal effect on training runtime.
© 2 0 2 5 NVIDIA. All rights reserved. 1 7

## Page 1 8

Pretraining Large Language Modelswith NVFP 4
0.0 0
-0.2 5
-0.5 0
-0.7 5
-1.0 0
-1.2 5
-1.5 0
0 1 2 3 4 5 6 7 8 9 1 0
)%(
ledom
8 pf
morf
ecnereffid
evitale R
### Nvfp 4
NVFP 4 switch to BF 1 6 for forward and backward
NVFP 4 switch to BF 1 6 in forward
NVFP 4 switch to BF 1 6 in backward
NVFP 4 switch to BF 1 6 for forward and backward (1 0 T tokens)
Start of
transition from
NVFP 4 to BF 1 6
Tokens (in trillions)
Figure 7 | Switching to higher precision towards end of training. Plot shows relative difference in validation
loss for a 1 2 B model trained on 1 0 T tokens. NVFP 4 uses the method specified in Section 4 during all
of the training period (Green). The precision for tensors in forward and backward pass (Blue), tensors
only in the forward pass (Orange), and tensors only in the backward pass (Purple) are switched from
NVFP 4 to BF 1 6 at 8.2 T tokens until remainder of training. A run where the switch to high precision
occurs around 1 0 T tokens is also shown (Red). 1 D weight scaling is used when switching precision for the
backward pass, since doing so is marginally better than 2 D weight scaling in such a setup.
We find that most of FP 4 training’s loss gap arises from quantizing tensors in the forward pass (Castro
et al., 2 0 2 5). More specifically, most of the loss in the 1 2 B model is recovered (from 1.5% to 0.5% relative
error) by switching to higher precision for the forward pass starting at 8.2 T tokens. In contrast to Chmiel
et al. (2 0 2 5), which reports loss recovery from switching precision in the backward pass, we observe no
such improvement in our models. Focusing on the forward pass minimizes the overhead of switching
precision, as only about 6% of the total computations (roughly one-third of the final 1 8% of training) are
performed in higher precision.
E. Ablation of Training Methodology
0.0
-0.5
-1.0
-1.5
-2.0
-2.5
-3.0
-3.5
0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1
)%(
ledom
6 1 fb
morf
ecnereffid
evitale R
LLaasstt 4 4 i nin B BFF 1 1 6 6
LLaasstt 4 4 i nin B BFF 1 1 6 6 + + 2 2 DD W
LLaasstt 4 4 i nin B BFF 1 1 6 6 + + 2 2 DD + W H +a d Haamdaarmdard
LLaasstt 4 4 i nin B BFF 1 1 6 6 + + 2 2 DD + W H +a d Haamdaarmda +rd S +R SR
Tokens (in trillions)
Figure 8 | Combining NVFP 4 training techniques: linear layers in last four blocks in BF 1 6, 2 D weight
scaling, Random Hadamard transforms on Wgrad, and stochastic rounding on gradients. Plot shows
relative difference in validation loss for a 1.2 B model trained on 1 T tokens.
© 2 0 2 5 NVIDIA. All rights reserved. 1 8

## Page 1 9

Pretraining Large Language Modelswith NVFP 4
E.1. Combining techniques
Given FP 4 training requires a suite of techniques, we explore the effects of combining them. We start
from a base method that quantizes all of the layers to NVFP 4, applies the standard NVFP 4 scaling (i.e.,
1×1 6 E 4 M 3 per-block scales with FP 3 2 per-tensor scales) to all tensors, and uses round-to-nearest-even
on all tensors. This base method is used throughout the appendix and combined with other techniques,
unless specified otherwise. Our models diverge early in training when using this base method without
any of the additional techniques. We find that maintaining some linear layers in higher precision plays a
key role in training stability, as elaborated in the following section. While techniques such as stochastic
rounding can improve training stability, they eventually diverge when used in isolation. Figure 8 shows
that combining the techniques leads to improvements in the loss. The relative benefit of each technique
depends on the order in which the components are added. Combining all of the components together
reduces the loss gap compared to a single technique.
E.2. Layer sensitivity
While training diverges with the base method when not using any of the techniques, some layers seem to
be more sensitive to FP 4 than others. Figure 9 shows loss converges when the linear layers in the last four
blocks remain in BF 1 6, which implies the final layers are more sensitive to FP 4 quantization. Maintaining
the first few blocks in higher precision does not improve stability unless combined with the last blocks
(e.g., training is stable when the first two and last two blocks are in BF 1 6, but not when the first four
blocks are in high precision).
2.0 5
1.9 5
1.8 5
1.7 5
1.6 5
1.5 5
1.4 5
0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1
ssol
noitadila V
### Bf 1 6
Last 1 in BF 1 6
Last 2 in BF 1 6
Last 4 in BF 1 6
First 4 in BF 1 6
First 2 and last 2 in BF 1 6
Tokens (in trillions)
Figure 9 | Sensitivity of linear layers to quantization. NVFP 4 for all linear layers except in a few of the
first and last blocks in the model. Plot shows validation loss for a 1.2 B model trained on 1 T tokens.
Based on tensor analysis, we observe the last layers tend to have larger quantization errors in the weight
gradients (i.e., Wgrad output from its inputs being FP 4). Quantization error metrics could potentially
serve as a mechanism to determine which linear layers should remain in higher precision during training.
E.3. Stochastic rounding on tensors
Since stochastic rounding is important for FP 4 training, we investigate its effect on various tensors during
training. As shown in Figure 1 0, applying stochastic rounding to gradients leads to stable convergence
of the training loss for the 1.2 B model, whereas using it on activations or weights causes divergence. A
potential cause of divergence due to stochastic rounding of activation and weight tensors is that this form
of rounding introduces more quantization error than nearest rounding (Chmiel et al., 2 0 2 5). This aligns
with prior findings that stochastic rounding mitigates gradient bias arising from quantization (Tseng
© 2 0 2 5 NVIDIA. All rights reserved. 1 9

## Page 2 0

Pretraining Large Language Modelswith NVFP 4
et al., 2 0 2 5 b; Chmiel et al., 2 0 2 5; Chen et al., 2 0 2 5; Castro et al., 2 0 2 5). Additionally, stochastic rounding
of all tensors in the backward pass shows little improvement over stochastic rounding of gradients only.
This suggests that divergence arises from stochastically rounding tensors in the forward pass. For the
1 2 B model, we observe that stochastic rounding must be applied to gradients going into both Dgrad and
Wgrad to achieve proper convergence.
1.9 5
1.8 5
1.7 5
1.6 5
1.5 5
1.4 5
0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1
ssol
noitadila V
### Bf 1 6
SR on Wgrad and Dgrad
SR on gradients
SR on activations
SR on weights
Tokens (in trillions)
Figure 1 0 | Stochastic rounding applied to different tensors: gradients, activations, weights, and backward-
pass tensors. NVFP 4 is applied on all linear layers except in the last four blocks. Plot shows validation
loss for a 1.2 B model trained on 1 T tokens.
E.4. Random Hadamard transforms
0
-1
-2
-3
-4
0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1
)%(
ledom
6 1 fb
morf
ecnereffid
evitale R
No Hadamard
Hadamard on Wgrad
Hadamard on Fprop
Hadamard on Dgrad
Tokens (in trillions)
Figure 1 1 | Impact of applying Random Hadamard Transforms (RHT) to different GEMMs (Fprop, Dgrad
and Wgrad) during training, compared to no RHT. For RHT runs, each transform uses a fixed random
seed across the entire training. NVFP 4 quantization is applied to all linear layers except in the last
four blocks. The plot shows the relative change in validation loss compared to the BF 1 6 baseline for a
1.2 B-parameter model trained on 1 T tokens.
E.4.1. GEMMs to apply RHT: We evaluate the impact of applying Random Hadamard Transforms
(RHT) to different GEMMs (Fprop, Dgrad and Wgrad) during FP 4 training. As shown in Figure 1 1,
applying RHT to Wgrad inputs improves validation loss for the 1.2 B model, while transforming Fprop or
Dgrad inputs degrades model quality. We hypothesize that RHT introduces additional quantization error
© 2 0 2 5 NVIDIA. All rights reserved. 2 0

## Page 2 1

Pretraining Large Language Modelswith NVFP 4
that offsets the benefit of outlier removal. Thus, although RHT reduces the dynamic range required to
represent outliers, its application can negatively affect training when used on certain GEMMs.
0.0 0
-0.2 5
-0.5 0
-0.7 5
-1.0 0
-1.2 5
-1.5 0
3.4 3.5 3.6 3.7 3.8 3.9 4
)%(
ledom
8 pf
morf
ecnereffid
evitale R
4 x 4 Hadamard matrix
1 6 x 1 6 Hadamard matrix
1 2 8x 1 2 8 Hadamard matrix
Tokens (in trillions)
Figure 1 2 | Effect of varying Hadamard Matrix Size. Wgrad tensors use 1 6×1 6 transforms for the first
3.4 T tokens, then switch to 4×4 or 1 2 8×1 2 8 for the remainder of training. Plot shows relative difference
in training loss for the 1 2 B model trained on 4 T tokens. NVFP 4 is applied on linear layers using the
methodology specified in Section 4.
E.4.2. Hadamardmatrix size: Sincethe Hadamardmatrixsizeimpactstheextentofoutliermitigation,
we consider different choices of matrix sizes to transform Wgrad inputs. For the 1.2 B model, we observe
virtually no difference in loss between 2×2, 4×4, 1 6×1 6 and 1 2 8×1 2 8 matrices. To validate this trend
at scale, we take the 1 2 B model trained up to 3.4 T tokens, switch the matrix size from 1 6×1 6 to 4×4 or
1 2 8×1 2 8, and continue training.
Figure 1 2 shows that 4×4 matrices induce an increase in loss and 1 2 8×1 2 8 matrices result in a minor
benefit to model quality. This follows the intuition that larger Hadamard matrices can better distribute
outliers, whereas matrices with too few entries are less likely to reproduce a Gaussian distribution.
The results validate our choice of a 1 6×1 6 matrix, which reduces the cost of the transform without
compromising model accuracy. It also highlights the need to experiment with larger models trained on
longer token horizons, since conclusions from smaller scales may not always hold for larger models.
E.4.3. Role of randomization: Random Hadamard transforms introduce randomness into the trans-
formation, so we study the importance of this randomization during training. Figure 1 3 illustrates loss
when training using different degrees of randomization: (1) “seed per instance,” a new random sign vector
for every transformation, (2) “single fixed seed,” a single random sign vector used for all transformations
during training, and (3) no random sign vector. We observe lower model quality in the absence of random
sign vectors and no improvements from inducing randomness at every transform instance. A a result, we
find it sufficient to use a single fixed seed for all transforms for our 1 2 B model. Interestingly, there are no
noticeable differences in model quality between the randomization strategies on the 1.2 B model, further
confirming that techniques become more critical at larger models and longer token horizons.
E.5. Consistent representations between tensors
Applying scaling and Hadamard transforms on a weight or activation tensor typically results in different
quantized representations in the forward and backward pass. We therefore study the impact of inconsistent
representations for tensors during model training. In particular, we consider different choices for scale
factors: (1) 1×1 6 block scales along the same dimension (i.e., input channels) in the forward and backward
pass, (2) 1×1 6 block scales along different dimensions (i.e., dot-product dimension, which changes from
© 2 0 2 5 NVIDIA. All rights reserved. 2 1

## Page 2 2

Pretraining Large Language Modelswith NVFP 4
0.0 0
-0.2 5
-0.5 0
-0.7 5
-1.0 0
-1.2 5
3.4 3.5 3.6 3.7 3.8 3.9 4
)%(
ledom
8 pf
morf
ecnereffid
evitale R
Non-random
Single fixed seed
Random seed for each transform
Tokens (in trillions)
Figure 1 3 | Effect of randomization for the Hadamard transform. A single fixed seed is used for all
transforms during the first 3.4 tokens and switched to one of the following randomization options for the
remainder of training: a single fixed seed for all layers, a unique seed for every transform, and not using a
random sign vector. Plot shows relative difference in training loss from the FP 8 baseline for a 1 2 B model
trained on 4 T tokens. NVFP 4 training uses the training methodology specified in Section 4.
0
-1
-2
-3
-4
0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1
)%(
ledom
6 1 fb
morf
ecnereffid
evitale R
Weights with 1 x 1 6 scales along different dims
Weights with 1 x 1 6 scales along same dim
Weights with 1 6 x 1 6 scales
Activations with 1 x 1 6 scales along different dims
Activations with 1 x 1 6 scales along same dim
Tokens (in trillions)
Figure 1 4 | Effect of consistency in tensors. Relative difference in validation loss from the BF 1 6 baseline
for a 1.2 B model trained on 1 T tokens. NVFP 4 is applied on either weights or activations. Different
choices of scaling factors are applied: 1×1 6 block scales along the same dimension, 1×1 6 block scales
along different dimensions, and 1 6×1 6 block scales, along with a global FP 3 2 per-tensor scale.
input channels in forward to output channels in backward) , and (3) 1 6×1 6 block scale factors. While
(1) and (3) maintain the same quantized representation in both stages of training, (2) will have different
quantizations between forward and backward. Only (2) and (3) can be implemented in practice, as Tensor
Cores require scaling factors along the dot-product dimension, which is transposed in the backward pass.
In Figure 1 4, we observe that having different quantized weight tensors negatively impacts the loss
throughout training of the 1.2 B model, where (1) achieves better accuracy than (2). Scaling using 2 D
blocks in (3) also improves the loss over (2), despite having a larger block granularity. On the other
hand, activations are less sensitive to consistency between tensors in the forward and backward pass, and
only impacted in the later stages during the learning rate decay. We hypothesize that weights are more
impacted than activations because errors induced from inconsistent weights materialize in the activation
gradients, which flow through the model layers during backpropagation. We also suspect that applying
Hadamard transforms exacerbates the inconsistency and further impacts model accuracy.
© 2 0 2 5 NVIDIA. All rights reserved. 2 2



---

*This document was automatically converted from PDF to Markdown.*
