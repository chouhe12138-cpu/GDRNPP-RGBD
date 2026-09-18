# EXP024 LM-O Progressive PCC ImageNet 全主干训练

- `experiment_id`: `EXP-20260919-024-lmo-progressive-pcc-fulltrain`
- 状态：`LOCAL_CPU_PREFLIGHT_PASS / SERVER_READONLY_RESOURCES_PASS /
  SERVER_EGL_PENDING / FORMAL_NOT_STARTED`
- 正式 `run_id`、checkpoint、评估指标：未生成。
- 源码 commit：待本地提交后填写；不以未提交工作树作为服务器 source。

## 研究问题与变量

在 LM-O 的 EXP022 PBR40/GT-box、固定 `reused_v1.npz` 层级与相同 PCC 方法上，
检查从 ConvNeXt-Base ImageNet 权重开始训练完整 backbone 与 PCC 的可行性和姿态表现。
EXP022 冻结臂从官方 LM-O checkpoint 取 backbone，而本实验从 ImageNet 取 backbone
并解冻，因此**初始化与训练范围同时变化**；两臂差值不能单独解释为解冻效应。
本实验不改变 PCC 网络结构，不代替 EXP023 的 LM13 数据协议。

## 固定协议与预定 gate

- 配置：`configs/gdrn/lmo_pbr/research/exp022_progressive_pcc/train_lmo_full_imagenet.py`；
  接线 smoke：同目录 `smoke_lmo_full_imagenet.py`。
- seed 42；`lmo_pbr_train` → `lmo_bop_test`；LM-O GT bbox；训练 EGL / 评估 CPP；
  冻结 geometry head，仅训练 ConvNeXt backbone 和 PCC；初始化权重由
  `GDRN_CONVNEXT_BASE_WEIGHTS` 指向容器内只读 ConvNeXt-Base ImageNet 文件，
  `MODEL.WEIGHTS=""`，不载入官方 LM-O 模型 checkpoint。
- 40 epoch；物理 batch 4，梯度累积 12 次，effective batch 48；AdamW PCC
  `3e-4`、backbone `3e-5`；4% linear warmup 后 cosine、终点 factor `0.01`；
  FP16 AMP；E5/E10/E15/E20/E25/E30/E35/E40 保存 checkpoint 并评估。
- Integrity：配置与对象顺序、层级、340 个 ImageNet 主干张量、可训练参数与优化器、
  真实 EGL batch 的有限 loss/梯度、AMP 无跳步以及正式 run 的 checkpoint/退出状态。
  任一训练链路失败时不启动或继续 formal。
- Mechanism：正式固定点的 BOP AR、ADD(-S)0.1d、reS、teS 与逐物体结果；
  训练后以相同 fixed support、2D 点和 RANSAC 设置与 EXP022 冻结臂比较。
  单次 seed 只支持该运行的结论。
- Advisory：lab1 batch4 显存和步骤时间、实际端到端吞吐；资源不足时先调整并记录
  实际物理 batch/累积口径，不静默改变有效 batch。

## 2026-09-19 本地准备（Observed）

- 新配置与独立 `TRAIN_PROTOCOL.NAME=lmo_full_imagenet` 已接入 launcher 的资源门；
  该门检查 LM PBR、LM-O test、VOC、ImageNet checkpoint、`reused_v1.npz`，并在容器内
  运行 `research.exp022.preflight`。不要求本实验未使用的官方 LM-O 训练权重。
- `pytorch22` 定向配置/launcher 测试 `56 passed`；完整 `pytest -q research`
  为 `288 passed, 3 failed`。这 3 项均为既有 LM/DeepIM loader 测试，失败点是在
  工作区外的只读 `datasets/BOP_DATASETS/lm/models/` 写入 `models_*.pkl` 缓存时
  得到 `OSError: [Errno 30] Read-only file system`；未出现 EXP024 配置、launcher
  或 protocol 断言失败。没有为通过测试而改写外部数据目录。
- 本地 CPU preflight `PASS`：8 类 BOP ID `[1,5,6,8,9,10,11,12]`，340 个 ImageNet
  主干张量精确匹配，`91,292,872` 个可训练参数，12 次梯度累积、有限 loss/梯度，
  symmetry counts `[1,1,1,1,1,2,2,1]`。
- 本会话本地 `torch.cuda.is_available()` 为 false，CUDA+CPP smoke 未运行；
  不把 CPU preflight 视为 EGL/AMP 通过。
- 用户在 lab1 执行服务器安全只读检查并回传终端截图：2026-09-19 00:30 本地服务器时间，
  物理 GPU 1 L40 空闲显存 `46068 MiB`、仅 Xorg 占用；项目容器
  `gdrnpp_chx_lab1` 标签为 `GDRNPP-RGBD` / `lab1`，镜像
  `gdrnpp-research:torch220-cu121-sm89-c0be1ade7ea9`，源码挂载
  `.../releases/GDRNPP-RGBD-effc99b`，容器内仅 `sleep infinity`。
  同一 `set -Eeuo pipefail` 检查块的 ImageNet 权重、LM PBR、LM-O test、VOC 与
  `reused_v1.npz` 读取检查均通过。尚未创建新 release、运行 EGL 或训练。

## 待完成 / Decision

lab1 必须先由用户执行只读 GPU/容器/数据/权重检查，再用新的确定 commit 建立唯一
release。此时正式配置的 `FORMAL_READY=False`；在服务器容器中运行真实 EGL + AMP
smoke、记录显存/步骤时间、确认 optimizer step 后，回本地开启 formal gate、提交并
生成第二段 release，最后启动正式训练。EXP023 LM13 仍按用户顺序在 EXP022 结束后进行。
