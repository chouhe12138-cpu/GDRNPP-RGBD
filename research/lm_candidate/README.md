# LM13 CAD candidate（维护项，不是 EXP 实验）

本候选将现有 LM13 real + DeepIM/ImageNet-render 协议与 LM-O 的执行入口分开。
切换数据集只选择配置；模型 `GDRN_CAD`、CAD head、loss、decode 无 LM/LM-O 分支。
LM 与 LM-O 配置分别继承 `configs/gdrn/research/cad/_base_/common.py`；该 base
只保存跨数据集相同的 CAD 模型结构。LM 配置不再继承
`lmo_pbr/research/exp025_hierarchical_cad_attention/`。

| 用途 | 配置 |
|---|---|
| LM13 主候选 | `configs/gdrn/lm/research/candidate_cad/train_imagenet_full.py` |
| LM13 本地短测 | `configs/gdrn/lm/research/candidate_cad/smoke.py` |
| LM-O 正式已锁定配置 | `configs/gdrn/lmo_pbr/research/exp025_hierarchical_cad_attention/train_imagenet_full.py` |

旧 `train_lm13_imagenet_full.py` 保留为 EXP025-era 兼容入口，不用于新候选。
候选保持 LM13 13 类、real+render、ImageNet full、backbone LR×0.1、Ranger
1e-4、物理 batch4/参考 batch24、160 epoch、1000-step warmup、72% cosine、每
20 epoch checkpoint/eval、GT box、EGL 训练/CPP legacy 评价。它没有服务器 profile；
`RESEARCH_PROTOCOL.STAGE=candidate` 永不允许 formal。

公共的 backbone 配置、ImageNet/optimizer 审计、AMP、批次与遥测位于
`research/cad_common/`。旧 `research.exp025.configuration/preflight/runtime`
仍导出历史 API；EXP025 的 hierarchy SHA、official weight、arm 与 metadata 约束
继续留在 EXP025 私有模块。共享批次读取按当前 dataset context 校验身份，只有
EXP025 wrapper 兼容旧的未标记 LM-O 保存批次。

## 本地检查（2026-09-22）

- 2026-09-23 结构收口回归：LM、EXP025 两臂、EXP026 两臂及旧 LM13 兼容入口的完整
  effective config 与 `d0bd434` 基线一致；共享 helper compatibility、LM/LM-O saved-batch
  身份、CPU preflight、CPP CUDA 两步更新及 legacy evaluator 均通过。

- 两个 `consistent_v3.npz` 的实际文件均有四级 `8/64/512/4096` 数组；CAD head 只读取
  T1–T3。契约如实描述 artifact，EXP025 仍保持原 SHA 锁，不重建正式 artifact。
- LM13 hierarchy SHA `322cd3778f0325838675a7dc6bae1a4e1cf107bfa95e05c006dcee0836a66417`；
  object IDs `1,2,4,5,6,8,9,10,11,12,13,14,15`。real/render/test 三个 online split
  的 smoke 样本各取到 8 条；主配置完整 split 分别解析为 2375、13000、13425 条，
  `xyz_crop` 不需要存在。
- CPU preflight PASS：340 个 ImageNet tensor 精确加载；head/backbone LR 为
  `1e-4/1e-5`；loss/backward、有限 XYZ、严格 state_dict 往返 PASS。
- 本地 RTX 4060、CPP、batch4、AMP scale16384、两步更新 PASS；确定性混合 real/render
  批次的 class/shape 正常，0 skipped step，checkpoint 往返 PASS。最终报告/权重位于忽略的
  `output/candidates/lm13_cad/cuda_cpp_20260922_a02/`。
- 本地 EGL 在创建 renderer 时以 `Bindless Textures not supported` 阻塞，尚无
  EGL 训练步；记录见 `output/candidates/lm13_cad/cuda_egl_20260922_a02/report.json`。
- 8 张真实 test 图的 GT-oracle legacy evaluator 指标链 PASS（AD、rete、re、te、proj）；
  输出位于 `output/candidates/lm13_cad/eval_smoke_20260922_a02/`。oracle 的 100%/0
  误差只是接线自检，不是模型精度或科学结果；该 evaluator 的这些指标不调用 CPP
  渲染器，CPP 已在本地训练 batch 短测中运行。

`DATASET_CONTEXT.BOP_TARGETS_FILENAME=test_targets_bop19.json` 只在共享 context 中
形成 BOP 路径；当前 `VAL.USE_BOP=False` 选择 `GDRN_EvaluatorCustom`，直接从
`lm_13_test_online` 标注读 GT，不消费 `VAL.TARGETS_FILENAME=lm_test_targets_bb8.json`。
本机缺少后者，故 BOP-style LM evaluator 未验证，未来需要独立配置与对应目标文件。

## 可复现的本地命令

在仓库根目录、`pytorch22` 环境中，先设置本机
`GDRN_CONVNEXT_BASE_WEIGHTS` 指向 ImageNet ConvNeXt 权重。以下命令不运行完整训练：

```bash
python -m research.lm_candidate.preflight
python -m research.lm_candidate.cuda_smoke \
  --output output/candidates/lm13_cad/cuda_cpp_UNIQUE_RUN --renderer cpp
python -m research.lm_candidate.eval_smoke \
  --output output/candidates/lm13_cad/eval_smoke_UNIQUE_RUN
pytest -q research/lm_candidate/tests research/exp025/tests research/exp026/tests
```

运行目录必须使用新的唯一名字。CUDA smoke 的 CPP 结果不能替代未来服务器 EGL
真实 batch4、资源和 formal gate；本候选无正式姿态精度结论。
