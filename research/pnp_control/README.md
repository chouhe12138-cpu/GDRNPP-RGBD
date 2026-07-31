# Stage 3C-0 Patch-PnP Adaptation Control

This control fine-tunes the existing official direct Patch-PnP head.  The
ConvNeXt backbone and geometry head remain frozen.  It adds no deployment-time
solver and no new architecture.

## Final conditional protocol

- Training: all 50 LM-PBR scenes.
- Evaluation: LM-O GT boxes after every five completed epochs.
- Local pilot: 8,192 instances from scenes `00–02`, exactly 1,024 per LM-O
  object after `visib_fract > 0.3`.
- Run this formal control only if C2 unfreezes Patch-PnP.

## Local pilot

```bash
conda activate pytorch22
research/pnp_control/run_local.sh
```

The RTX 4060 configuration uses physical batch 4 and 12-step gradient
accumulation for effective batch 48.

## L40 control

Inside the user's Docker container, mount the repository, BOP datasets, VOC
data, and official checkpoint so that the repository-relative links resolve.
Then run:

```bash
conda activate pytorch22
DEEP_PREFLIGHT=1 research/pnp_control/run_full.sh
```

The L40 configuration uses physical/effective batch 48, 40 epochs, Ranger
with learning rate `8e-5`, weight decay `0.01`, and FP32 inherited from the
official configuration.

## Required post-run checks

```bash
python -m research.pnp_control.summarize_local \
  output/EXP-20260731-005/pnp_only_local/metrics.json

python -m research.pnp_control.verify_checkpoint_isolation \
  --official pretrained_models/lmo_pbr/model_final_wo_optim.pth \
  --trained <trained-checkpoint>
```

Passing isolation requires at least one `pnp_net.*` tensor to change and every
backbone/geometry tensor to remain bit-identical.  A decreasing training loss
only proves pipeline feasibility; it is not pose-accuracy evidence.
