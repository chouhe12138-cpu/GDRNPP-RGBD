# Stage 0 — Repository Bootstrap

## Goal

Create an independent, auditable GDRNPP-RGBD source repository and write the
staged research plan. This stage makes no model or environment changes.

## Inputs

```text
Source: /mnt/d/6D姿态估计/gdrnpp_bop2022-main.zip
Target: /home/wsluser/GDRNPP-RGBD
Branch: main
```

## Allowed Work

- Verify and extract the source archive.
- Record source provenance and broken external links.
- Initialize local Git.
- Add repository ignore rules and research documents.
- Create the two baseline commits.

## Explicitly Forbidden

- Installing or upgrading dependencies.
- Downloading or repairing `bop_renderer`.
- Linking or copying datasets and model weights.
- Importing old RDPN6D/RRF implementation changes.
- Modifying model, loader, loss, or evaluation code.
- Running training or inference.
- Configuring a Gitee remote.
- Accessing or changing the L40 server.

## Acceptance Checklist

- [x] Source archive exists and passes zip integrity validation.
- [x] Source SHA-256 and inventory are recorded.
- [x] Target path was confirmed absent before migration.
- [x] Expected upstream directories and the main GDRNPP model file exist.
- [x] Broken `bop_renderer` link is recorded without modification.
- [x] Independent target directory has been created.
- [x] Upstream snapshot commit has been created.
- [x] Research documentation commit has been created.
- [x] No dataset, weights, checkpoints, large logs, or credentials are tracked.
- [x] `git status` is clean.
- [x] Repository integrity checks pass.

## PASS Rule

Stage 0 is PASS only after every checklist item is satisfied and the verification
evidence below is filled in. A PASS does not authorize Stage 1.

## Verification Evidence

To be completed after migration:

```text
Status: PASS
Snapshot commit: 8a3c7d70c8eadb36180c4a7e5b199c27971d1a39
Documentation commit: docs: add staged RGB-D research plan (this commit)
Tracked files: 1,451
Largest tracked file: 6,196,215 bytes
Git fsck: PASS
Git status: clean on main
```

## Next Stage

Stage 1 will plan environment reconstruction, native extensions, the renderer,
dataset path configuration, and official baseline inference. It remains
unauthorized until explicitly requested.
