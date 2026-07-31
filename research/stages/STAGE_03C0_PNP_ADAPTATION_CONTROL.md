# Stage 3C-0 — Official Patch-PnP Adaptation Control

Status: `LOCAL PILOT PASS — CONDITIONAL B CONTROL`

## Question

Before adding quality/coverage aggregation, test the simpler explanation:
perhaps the official Patch-PnP architecture has enough capacity, but its
official frozen weights are not adapted to the controlled XYZ intervention
diagnosed in Stage 3B.

The local run remains useful engineering evidence. Under the final C1 design
all official parameters are frozen, so this formal control is required only
if a later C2 experiment unfreezes Patch-PnP.

## Frozen intervention

- Initialize from the official LM-O ConvNeXt-Base checkpoint.
- Keep the official direct `R,t` Patch-PnP output.
- Freeze the ConvNeXt backbone and complete geometry head.
- Fine-tune only the existing `pnp_net`; add no layer, loss, depth input, or
  RANSAC deployment path.
- Render XYZ targets online with the existing C++ renderer.
- Use all 50 LM-PBR scenes for training.
- Evaluate direct `R,t` on LM-O with GT boxes every five completed epochs.
- Use seed `20260731`.

## Compute

Formal L40 control:

```text
epochs:          40
physical batch:  48
effective batch: 48
optimizer:       Ranger
learning rate:   8e-5
weight decay:    0.01
precision:       FP32
```

Local RTX 4060 pilot:

```text
instances:       8,192 (1,024 per LM-O object)
scenes:          00–02
visibility:      visib_fract > 0.3
epochs:          1
physical batch:  4
accumulation:    12
effective batch: 48
```

The local pilot is only a feasibility and isolation check.  It cannot select a
method or establish pose accuracy.

## Conditional comparison

If C2 unfreezes Patch-PnP, B and C2 start from the same official checkpoint,
use all 50 PBR scenes, the same 40-epoch schedule, the same seed, and
Patch-PnP learning rate `8e-5`. The architecture contribution is then
measured by C2 minus B.

## Local result

The pilot completed 2,048 micro-batches / 171 optimizer updates on the RTX
4060, with 4,820 MiB peak allocated memory.

- All recorded losses were finite.
- Total-loss first-to-last-quarter median changed from `0.038747` to
  `0.036650` (`-5.41%`).
- Rotation loss changed by `+1.29%`; centroid and depth losses changed by
  `-2.17%` and `-3.27%`.  This mixed component behavior is not accuracy
  evidence.
- The final checkpoint strictly reloaded with zero missing/unexpected keys.
- Exactly 17 trainable tensors were present, all under `pnp_net`.
- Checkpoint comparison found 17 changed `pnp_net` tensors and zero changed
  frozen tensors out of 392 tensors.

Therefore:

```text
LOCAL_PIPELINE_AND_ISOLATION = PASS
FORMAL_MODEL_DECISION = PENDING
```
