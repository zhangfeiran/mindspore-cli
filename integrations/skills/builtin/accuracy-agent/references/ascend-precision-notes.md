# Ascend Precision Notes

Read this file only when the diagnosis involves Ascend backend behavior.

## Why This Matters

Two runs can differ even when the model code looks equivalent, because the
backend may choose different precision or kernel paths.

If the baseline is PyTorch on Ascend, first verify that the script actually
imports `torch_npu` and executes tensors and modules on NPU. A plain CPU
`torch` script is not an Ascend baseline.

## Shared Kernel Library: torch_npu and mindspore

When comparing `torch_npu` vs `mindspore` on Ascend, both frameworks call the
same CANN/aclnn operator library underneath. This means:

- Kernel-level numerical behavior should be identical for the same operator with
  the same inputs, dtype, and configuration.
- Small deltas between the two frameworks are not explained by "different
  kernels." They indicate a callsite mismatch, parameter mismatch, or wrong
  operator mapping.
- A fully aligned `mindspore` operator path should exist for any `torch_npu`
  operator. Do not accept residual deltas without evidence that operator mapping
  and all parameters are fully aligned.

When a small persistent delta remains between `torch_npu` and `mindspore` for
the same operation, the most likely causes are:

- different implicit default parameters
- different operator variant (e.g., legacy `mindspore.ops` vs aligned
  `mindspore.mint`)
- dtype or cast-path differences
- device placement mismatch

## High-Value Checks

### HF32 or Other Special Precision Modes

Check whether a special precision mode is enabled on one side but not the
other. Backend precision policy differences can produce real numerical deltas
without indicating a semantic bug in the model code.

### Accumulation Path Differences

Pay special attention to MatMul and Conv-like paths:

- accumulation precision may differ from input precision
- mixed-precision execution may use different internal accumulation behavior
- small output deltas can grow through deep networks

### Precision Control Settings

Check whether global or local precision settings force different execution
paths. A mismatch here can explain:

- step1 loss mismatch
- cross-platform output mismatch
- sensitivity to graph optimization changes

### Alignment and Kernel Path Differences

Shape, padding, layout, and alignment can route execution through different
kernel paths. When this happens:

- compare a smaller deterministic case first
- confirm whether the two sides are really exercising the same shape pattern

### BF16 and Mixed Precision Context

Avoid raw tensor comparison without context:

- compare in a precision-aware way
- note when the baseline and current run use different reduced-precision paths
- if needed, temporarily raise precision for diagnosis only

## When to Escalate These Checks

Look here early for:

- step1 loss mismatch on Ascend
- PyTorch vs MindSpore comparison on Ascend
- output mismatch that disappears when precision settings change

Do not use these notes as a blanket excuse for every mismatch. They are useful
only when evidence points to backend precision behavior.
