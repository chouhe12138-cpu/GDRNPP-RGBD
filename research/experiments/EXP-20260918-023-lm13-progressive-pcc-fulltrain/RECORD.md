# EXP023 LM13 Progressive PCC 全 backbone 训练准备

- `experiment_id`: `EXP-20260918-023-lm13-progressive-pcc-fulltrain`
- 状态：`CONFIG_AND_LOCAL_SMOKE_PASS / FORMAL_NOT_STARTED`
- 日期：2026-09-18；源码基准：`373298273bf14953634b60b1f2d18ddfab3dbb05` 加本工作区未提交修改
- 正式 `run_id`、checkpoint、epoch 与评估指标：未生成

## 问题与协议

在标准 LINEMOD 13 类上检验 EXP022 的渐进式 CAD 对应方法，训练完整 ConvNeXt backbone 与 PCC。与 LM-O 冻结官方 backbone 的第一阶段相比，本实验同时改变数据集、CAD 层级来源和 backbone 训练策略，因此不能将两者性能差异解释成单一因素效应。

准备配置为 `configs/gdrn/research/exp022_progressive_pcc/train_lm13.py`，PBR 在线 XYZ、GT box 测试、13 类联合训练；ImageNet ConvNeXt 初始化由 `GDRN_CONVNEXT_BASE_WEIGHTS` 显式提供，backbone 学习率倍率 `0.1`，PCC 用基础学习率。当前计划参数为 seed 42、40 epoch、batch 4、AdamW `3e-4`、AMP FP16、每 5 epoch 评估和保存。正式协议 `FORMAL_READY=False`；这些参数是准备值，正式 gate 尚未预注册，训练未启动。

## 本地 Observed（工程验证）

- 训练 split `lm_pbr_13_online_train` 与测试 split `lm_bop_test_13` 的对象 ID 顺序均为 `[1,2,4,5,6,8,9,10,11,12,13,14,15]`；排除 bowl/cup。训练 split 不要求缺失的 `xyz_crop`，使用在线 renderer。
- 忽略 Git 的层级 `.local/dataset_cache/exp022/lm13/independent_v2.npz`：从 `ref.lm_full` 的 CAD 模型独立生成；`generator_version=2`、seed `20260916`、每物体采样 `200000`；13 个对象各有 8/64/512/4096 个节点。`symmetry_counts=[1,1,1,1,1,1,1,2,2,1,1,1,1]`；artifact 的 `object_ids` 与 train/test metadata 顺序一致。
- CPU preflight PASS：ImageNet checkpoint 的 340 个 ConvNeXt 张量一一对应且加载值相同；模型可训练参数 `91,292,872`，三项 loss、反传梯度和推理 finite。
- 本机 RTX 4060 Laptop、CUDA+CPP、真实 PBR 场景 0 的 batch 1、seed 42、FP16 AMP 两步 smoke PASS，无跳步。类别索引 7（BOP ID 10）且为对称实例；第二步 route/residual/mask loss 为 `1.739060/0.330845/0.559275`。峰值 allocated/reserved `1.831321/2.292187 GB`；第二步固定 batch 模型步耗时 `197.64 ms`，未计每步读取和渲染。两步不证明收敛或正式吞吐。
- BOP target 协议检查 PASS：`lm_bop_test_13` 对应 `test_targets_bop19.json` 共 2600 个目标；实际实例化测试集得到 2600 张图、2600 个实例，类别索引覆盖 `0..12`。matched evaluator 尚未运行模型评价，因为 LM13 reference checkpoint 与 PCC 正式 checkpoint 均未产生。

## Decision / 待完成

数据集、层级、ImageNet 初始化和本地在线训练链路可以进入后续短训练准备。正式训练前仍需确定 LM13 reference 模型与 matched fixed-support 协议，预注册 gate，并将 `FORMAL_READY` 显式开启。T-LESS 数据尚未准备，其 variable-S 对称监督是单独的后续工作。
