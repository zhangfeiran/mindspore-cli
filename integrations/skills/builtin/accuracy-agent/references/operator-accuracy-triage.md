# Operator Accuracy Triage

Use this file only after the first stable mismatch has already been narrowed to
one operator or to a very small operator cluster. Do not start here from a
full-model symptom.

## Goal

Avoid false operator bug reports. Cross-framework accuracy gaps often come from
callsite mismatch rather than from the operator implementation itself.

## Step 1: Recheck the Operator Callsite Before Blaming the Operator

Compare how both model scripts call the operator, not just the operator name.

Complete this checklist before escalating to an operator bug:

- [ ] explicit arguments
- [ ] implicit default arguments
- [ ] positional versus keyword argument mapping
- [ ] dtype, cast, and shape expectations
- [ ] actual device placement
- [ ] reduction, eps, axis, mask, keepdim, align_corners, and similar semantic
  knobs

Treat implicit defaults as first-class evidence. If the baseline and target
scripts do not pass every relevant argument explicitly, inspect the actual API
defaults on both sides before blaming the operator implementation.

Why this matters:

- operators with similar names across frameworks may not share the same API
  schema
- default values may differ even when explicit arguments look aligned
- a callsite mismatch is usually cheaper to fix than an operator rewrite

Do not claim an operator bug until this callsite check is clean.

## Step 2: Build a Minimal Single-Operator Repro

If the callsite is aligned but the mismatch remains, construct a script that
only exercises the suspected operator.

Keep the repro focused:

- use the same input values, shapes, dtype, and attributes that exposed the
  mismatch
- remove unrelated model logic, optimizer logic, and dataloader noise
- keep determinism controls consistent when randomness is involved

Interpretation:

- if the isolated repro still shows the mismatch, the operator path is a strong
  root-cause candidate
- if the isolated repro does not reproduce the gap, return to the surrounding
  module context instead of escalating an operator claim

## Step 3: Try Direct Replacement Before Reimplementation

Once the single-operator repro confirms the issue, try a safer equivalent path
before proposing large changes.

Use this repair-priority checklist in order:

1. If the mismatching MindSpore path is under `mindspore.nn` or
   `mindspore.ops`, first check whether there is a same-name or officially
   mapped `mindspore.mint` operator for the same semantics.
2. Prefer a direct `mindspore.mint` replacement over a handwritten
   reimplementation from smaller `mint` operators. A direct replacement is the
   shortest and usually the highest-confidence fix path.
3. If no direct `mindspore.mint` replacement exists, prefer another documented
   non-legacy aligned operator variant over a legacy path when available.
4. Reimplement the operator from smaller operators only if direct replacement
   is unavailable, semantically incompatible, or proven insufficient by
   validation.

Why this order matters:

- some `mindspore.nn` and `mindspore.ops` entries are legacy paths in migration
  work
- `mindspore.mint` operators often align more closely with the corresponding
  `torch` or `torch_npu` operator in both API schema and underlying NPU kernel
  path
- a direct `mindspore.mint` replacement is usually cheaper, clearer, and more
  reliable than a handwritten small-operator rewrite

Before picking the replacement, check the official PyTorch-to-MindSpore API
mapping table:

- https://www.mindspore.cn/docs/zh-CN/master/note/api_mapping/pytorch_api_mapping.html

Use it to confirm:

- whether the intended MindSpore API is marked as aligned
- whether the mapping points to `mindspore.mint`, `mindspore.ops`, or
  `mindspore.nn`
- whether the mapped API has documented parameter-name, default-value, or other
  semantic differences

Do not:

- jump to a handwritten `mint` formula or a small-operator rewrite before trying a
  direct `mindspore.mint` replacement when one exists
- propose a mathematical decomposition (e.g., rewriting softmax from
  exp/sum/div, or layernorm from mean/var/scale) when a direct aligned API
  already covers the semantics — decompositions are fragile, hard to maintain,
  and usually less precise than a single aligned kernel
- assume explicit arguments are enough without checking implicit defaults
- treat a legacy `mindspore.nn` or `mindspore.ops` path as the preferred fix
  target when an aligned `mindspore.mint` operator is available

## Step 4: Validate at the Right Scope

After a replacement or reimplementation attempt:

1. rerun the single-operator repro
2. rerun the lowest validation-ladder rung that originally failed
3. return to the full module or model compare only after the smaller scope is
   clean

Do not report the operator fix as confirmed until the user validates the change
at the task-relevant scope.
