# EXP023 LM13 Progressive PCC 全 backbone 训练

- `experiment_id`: `EXP-20260918-023-lm13-progressive-pcc-fulltrain`
- 状态：`PROTOCOL_OVERHAULED / LOCAL_TRAIN_AND_DATA_CHECK_PASS / FORMAL_NOT_STARTED`
- 日期：2026-09-18（协议重整同日）；源码基准：`1e2ad1266ef5ab0feec9e005fd700af332b39219`
  加本工作区未提交修改
- 正式 `run_id`、epoch 与评估指标：未生成。唯一的 checkpoint 来自下面记录的本机接线短训练，
  不是正式结果。

## 问题与范围

在标准 LINEMOD 13 类上检验 EXP022 的渐进式 CAD 对应方法，训练完整 ConvNeXt backbone 与 PCC。
与 LM-O 冻结官方 backbone 的第一阶段相比，本实验同时改变数据集、CAD 层级来源和 backbone
训练策略，因此不能将两者性能差异解释成单一因素效应。

## 2026-09-18 协议重整（Observed / Derived / Decision）

按 `GDRNPP_LM_dataset_training_evaluation_adjustment_plan.md` 做实验协议层整理，不动
`GDRN_PCC` / `ProgressivePCCHead` / `DatasetContext` 的模型结构。原 `train_lm13.py` 名义上是
LM13、实际数据是 `lm_pbr_13_online_train`（GDRNPP 的 PBR/BOP 协议），本轮把它拆成三条显式协议。

### 数据（Observed）

- `lm_imgn` 由用户提供的 `lm_imgn.zip` 搬到 WSL 原生盘并解压到 `/home/wsluser/Datasets/lm_imgn`
  （`datasets/lm_imgn` 软链接，不进 Git）。校验：**505,191 个文件 / 6.1 GB**、16 个对象目录、
  13 个 `image_set/train_{obj}.txt`。GDR-Net README 记载该数据是 DeepIM 的 OpenGL 渲染集
  （1k/obj，网盘下载），仓库内没有生成脚本。
- `xyz_crop_imgn/` **存在**（13 类，每类约 1860 个 pkl），因此原版需要预生成 XYZ 的 split 也能运行；
  主实验仍用在线 XYZ 以与 PBR / real 两条 arm 一致，预生成 xyz 留作交叉核对。
- 渲染图约定：`imgn/{obj}/{im_id}-color.png`、`-depth.png`、`-pose.txt`（首行可跳过，3×4，
  **单位是米**）；深度为 uint16 毫米（实测 max 1101，对应 pose `t_z≈1.086`）；无 mask 文件，
  mask 由 `depth > 0` 导出；`benchvise` 与 `benchviseblue` 是同一条序列（互为软链接）。
- **`lm/train/` 不等于 `image_set/{obj}_train.txt` 的选片**：`benchvise_train.txt` 有 183 个 id，
  而 `lm/train/000002/rgb` 只有 181 张（缺 891、892）。因此 loader 一律从 `lm/test/{scene}`
  按 `image_set` 取片；若改读 `lm/train/` 会静默少图。该不变量已固化为单测。
- 13 类其余逐类计数一致：train 2,375 张、test 13,425 张、lm_imgn 13,000 张。

### 实现（Observed）

- 新增 `core/gdrn_modeling/datasets/lm_dataset_d2.py`（LM real）与 `lm_syn_imgn.py`（渲染图），
  按仓库既有 `SPLITS_*` / `ref_key` / `DATASET_CACHE_ROOT` 约定编写，未从 `third_party/`
  引入或复制代码，并已在 `dataset_factory.py` 注册。
- split：`lm_13_train`、`lm_13_train_online`、`lm_13_test`、`lm_13_test_online`、`lm_13_train_smoke`；
  `lm_imgn_13_train_1k_per_obj`、`_online`、`_smoke`。
- `data_loader.py` / `data_loader_online.py`：`img_type` 背景替换改为显式分流
  （`syn→1.0`、`real→CHANGE_BG_PROB`、`syn_pbr→PBR_CHANGE_BG_PROB`、其他报错），策略抽成两处
  共用的 `background_replace_probability`；修复原先两支都调用 `_color_aug`、导致
  `COLOR_AUG_SYN_ONLY` 完全失效的问题。确定性域不额外抽随机数，**RNG 流与修改前一致**。
- `configs/_base_/common_base.py` 新增 `PBR_CHANGE_BG_PROB=0.5`（默认值与原行为相同）。
- `lm_pbr.py` 模型加载改用该 split 自己的 `ref_key`，去掉残留的 `ref.lm_full`。
- `DatasetContext` 支持多个训练 split：额外 split 必须与第一个的对象顺序一致，否则报错。

### 协议配置（Observed）

| 配置 | 协议 | TRAIN | 增强 | Solver |
|---|---|---|---|---|
| `train_lm13_gdrn.py` | `lm13_gdrn` 主实验 | `lm_13_train_online` + `lm_imgn_13_train_1k_per_obj_online` | COLOR_AUG 0、VOC 背景 0.5、DZI 1.5/0.25/0.25 | Ranger 1e-4、wd 0、160 ep、effective batch 24 |
| `train_lm13_real_only.py` | `lm13_real_only` 数据消融 | `lm_13_train_online` | 同主实验 | 同主实验 |
| `train_lm13_pbr.py` | `lm13_pbr` | `lm_pbr_13_online_train` | COLOR_AUG 0.8 | AdamW 3e-4、40 ep |
| `eval_lm13_bop.py` | `bop_official` 评估 | 继承主实验 | — | — |

两处实现约束：mmcv 禁止兄弟 base 间键重复，所以协议文件 `lm13_gdrn_protocol.py` 放在实验目录
而非 `_base_/`；`solver_utils` 会优先用 `WARMUP_RATIO` 并把它当作 `ANNEAL_POINT`，因此该配置
显式将 `WARMUP_RATIO` 置空，否则平坦阶段会退化成几乎立刻 cosine。

### P1 数值检查（Observed）

`research/exp022/check_lm_data.py`，`--limit 12`，本机 CUDA + CPP 渲染 batch4：

| 量 | `lm_13_train_online` | `lm_imgn_13_train_1k_per_obj_online` |
|---|---|---|
| `img_type` | real | syn |
| 相机与 LM 标准内参最大绝对误差 | 0.0 px | 0.0 px |
| `bbox` 与分割 mask 的 bbox 最大误差 | 0.0 px | 0.0 px |
| `bbox3d_and_center` 边长 vs `models_info` size | 0.0（精确恒等） | 0.0 |
| 平移模长 | 0.88–1.08 m | 0.96–1.10 m |
| 对称类（类别索引） | 7、8 | 7、8 |

在线渲染一致：batch4 的源 mask 被渲染覆盖 `0.9639/0.9657/0.9537/0.9891`；归一化 XYZ
`[-0.0086, 1.0171]`、均值 `0.5067`，反归一化 ±0.10 m。位姿单位、相机或 CAD 映射任一有误时
该覆盖率会趋近 0，因此这是对三者的联合校验。

### P1 训练与接线（Observed）

- `real_smoke`：`smoke_lm13_gdrn.py`、batch4、3 步、FP16 AMP PASS，无跳步；loss
  route `2.08546→2.08498`、residual `0.33615→0.33525`、mask `0.71746→0.71602`；
  340 个 ImageNet 张量、可训练参数 `91,292,872`、峰值 allocated/reserved `2.820/2.999 GB`。
- 真实训练入口 `core/gdrn_modeling/main_gdrn.py`（`--num-gpus 1`、`XYZ_RENDERER=cpp`）
  用 `smoke_lm13_gdrn.py` 跑完 1 epoch / 4 iteration，**退出码 0**，并保存
  `output/experiments/EXP-20260918-023-lm13-progressive-pcc-fulltrain/RUN-20260918-lm13-gdrn-smoke-s42/checkpoints/model_epoch_001.pth`。
  日志确认 real 与 imgn 两个 smoke split 都被该入口加载。这是接线证据，**不是**收敛或精度结果。
- P2 接线：新增 `research/exp022/eval_manifest.py`（doc §14 要求的 provenance 字段）与
  `eval_lm13_bop.py`；`matched_pnp_eval.py` 完成时会写 `eval_manifest.json`，并显式标为
  `matched_diagnostic`。两个协议都已在上述 checkpoint 上成功生成 manifest。
- 回归：完整 `pytest -q research` **243 passed**；三个 LM13 配置 CPU preflight 均 PASS。
- `third_party/` 已加入 `.gitignore`（只读参考代码，不上传服务器）。

## Decision / 待完成

- 本轮只做协议整理与接线，**未产生任何精度结论**，未运行正式训练，未运行完整评估。
- 两条评估链（legacy ADD(-S)、BOP AR）都只能用 **GT bbox** 跑：官方 Faster R-CNN 的
  `lm/test/test_bboxes/bbox_faster_all.json` 全盘不存在（GDR-Net README 说明它需从单独的
  `image_sets`/`test_bboxes` 网盘包补齐）。因此 Protocol A/B 当前都是诊断口径，
  **不能**与 GDR-Net 论文的 detector-bbox 数字直接比较。
- LM13 没有 reference 模型，`matched_pnp_eval.py` 目前只能做协议校验；正式 matched 对比
  需等待 EXP021 B/C 结果完整后确定 comparator。
- 正式训练前仍需：预注册 gate、显式打开 `RESEARCH_PROTOCOL.FORMAL_READY`、确定服务器 EGL
  资源检查。T-LESS 数据尚未准备，其 variable-S 对称监督是单独的后续工作。
