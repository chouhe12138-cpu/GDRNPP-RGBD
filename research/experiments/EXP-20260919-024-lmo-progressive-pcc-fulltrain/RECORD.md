# EXP024 — LM-O Progressive PCC ImageNet 全主干训练

## Status

- lifecycle：`SUPERSEDED`
- execution：`FORMAL_NOT_STARTED`
- superseded by：EXP025 `imagenet_full`
- final historical snapshot：`8c6ca86aa0668777548d4f30b8cc6f6ad9864067`
- lab1 batch4 EGL smoke：PASS；40-epoch formal 和评价指标：未生成。

## Research question

在 EXP022 的 LM-O PBR40/GT-box 与 Progressive PCC 上，从 ImageNet ConvNeXt 初始化并训练
完整 backbone 与 PCC，检查该组合策略的可行性和姿态表现。相对 EXP022 frozen arm，初始化
和训练范围同时变化，不能解释为单一解冻效应。

## Intended protocol

- 历史配置：`train_lmo_full_imagenet.py`；smoke：`smoke_lmo_full_imagenet.py`。
- seed42；`lmo_pbr_train → lmo_bop_test`；training EGL、evaluation CPP；GT box。
- ImageNet ConvNeXt 初始化，`MODEL.WEIGHTS=""`；geometry head frozen，backbone/PCC trainable。
- 40 epoch；physical batch4、accumulation12、effective batch48。
- AdamW PCC 3e-4、backbone 3e-5；4% warmup 后 cosine，target factor .01；FP16 AMP。
- E5/E10/E15/E20/E25/E30/E35/E40 checkpoint/evaluation。
- hierarchy 为 `reused_v1.npz`；该旧 contract 后来由 EXP025 取代。

## Canonical observations

- 本地 CPU preflight PASS：LM-O 8 类 `[1,5,6,8,9,10,11,12]`，ImageNet backbone 340 tensors，
  `91,292,872` trainable parameters，accumulation12，finite loss/gradient，symmetry counts
  `[1,1,1,1,1,2,2,1]`。
- 当时完整 research tests 的 3 个失败来自只读 dataset 目录无法写 `models_*.pkl` cache；
  EXP024 config/launcher/protocol 定向检查无失败。没有改外部权限规避该环境问题。
- 用户在 lab1 完成 GPU/ownership/mount/resource 只读检查；物理 GPU1 空闲，项目容器 label、
  ImageNet weights、LM PBR、LM-O test、VOC 与 hierarchy 均匹配。
- source `a3570178799cbda943805344645e658f93b774fe` 的 lab1 release 完成 CPU preflight 和
  真实 EGL batch4、8-step FP16 smoke：status PASS、0 skipped step、finite loss/gradient、
  optimizer step 成功。
- smoke 固定 batch 分段中位数：forward 37.10 ms、backward 47.06 ms、gradient audit
  25.29 ms、optimizer 15.47 ms；peak allocated/reserved 2.826/3.012 GB。计时不含逐步
  DataLoader/EGL render，只是 advisory。
- smoke 通过后历史配置曾设 `FORMAL_READY=True`，但新的 clean formal release 和正式 run
  从未启动；没有 run ID、checkpoint、exit code 或 E5–E40 指标。

## Decision

本实验只证明旧 Progressive PCC + ImageNet full-training 组合在 lab1 batch4 EGL 上工程可行。
它没有证明真实 batch48、正式收敛或姿态精度。统一 T3 CAD attention、真实 batch48 gate 和
最终主干 LR 决策转入 EXP025，因此 EXP024 关闭，旧配置/profile 不恢复到 HEAD。

## Evidence / artifacts

- tracked evidence：无；服务器截图与本地 smoke 输出未形成 canonical JSON。
- 配置、runner 和 profile 由 final snapshot 的 Git history 恢复。
- 完整日志、checkpoint（无 formal）和诊断产物保持外置。

## Known gaps / boundaries

- 未运行真实 batch48 gate、formal、固定评价点或 matched PnP。
- smoke 只有 batch4，不能外推服务器端到端吞吐或精度。
- backbone 3e-5 没有 formal 证据；EXP025 后续将最终 full arm 改为 backbone/head 均 3e-4。
