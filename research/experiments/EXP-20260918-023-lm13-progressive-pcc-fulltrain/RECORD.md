# EXP023 LM13 Progressive PCC 全 backbone 训练

- `experiment_id`: `EXP-20260918-023-lm13-progressive-pcc-fulltrain`
- 状态：`PRETRAIN_CHECKS_PASS / SERVER_INTEGRATION_ADDED / LAUNCHER_CLOSURE_ADDED /
  EGL_SMOKE_PENDING / FORMAL_NOT_STARTED`
- 日期：2026-09-18（协议重整同日，随后按
  `EXP022_LM13_Pretraining_Modification_Task.md` 收口、按
  `EXP023_LM13_Server_Integration_Modification_Task.md` 接入服务器）；源码基准：
  `8304264bf77f8dd04ccf9a63b2de270a5e0bbd35` 加本工作区未提交修改
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

## 2026-09-18 正式训练前收口（Observed / Derived）

按 `EXP022_LM13_Pretraining_Modification_Task.md` 收口，未改 Progressive PCC 网络主体。
四处修改：

- `lm13_gdrn_protocol.py` 显式加 `SOLVER.TARGET_LR_FACTOR=0.0`。此前继承
  `research_runtime` 的 `0.01`，160 epoch 的 `flat_and_anneal` 最终只衰减到 `1e-6`，
  而 GDR-Net LM13 基线衰减到 0。只覆盖本协议，不动全局默认值（derive：
  `solver_utils` 用 `base_lr * target_lr_factor` 作为 cosine 终点）。
- `preflight.py` 把 `TARGET_LR_FACTOR` 加进 `LM13_GDRN_SOLVER`，继承回 `0.01` 会被拒绝。
  另修一处既存缺陷：smoke 配置从正式配置继承 `TRAIN_PROTOCOL.NAME="lm13_gdrn"` 但换成
  `*_smoke` split，因此 `preflight --config smoke_lm13_gdrn.py` 会被正式协议检查拒绝自己的
  smoke；现在只有非 smoke 模式做严格协议校验。
- `train_lm13_gdrn.py` 的 `TEST` 由 `("lm_13_test",)` 改为 `("lm_13_test_online",)`，并显式设
  `EVAL_PERIOD=20`（原继承 5，160 epoch 会跑 32 次全量 13k 图 RANSAC-PnP）。
  `BEST_CHECKPOINT.ENABLED=False`，每 20 epoch 共 8 次评估。
- `lmo_bop_test.py` 的 `obj2label` 反向映射修正为 `(obj, obj_id) for obj_id, obj in
  enumerate(...)`。该成员全仓库只赋值、无读取，所以是纯 cleanup；同形写法在约 40 个其他
  loader（含 `det/yolox/`）中仍未改，本次按范围限定只动该文件。

### 验证（Observed）

| 检查 | 结果 |
|---|---|
| 5 个 LM13 配置 mmcv 加载 | PASS，无 duplicate key |
| effective config | `IMS_PER_BATCH=4`、`REFERENCE_BS=24`、`TOTAL_EPOCHS=160`、Ranger `lr=1e-4` `wd=0`、`WARMUP_ITERS=1000`、`ANNEAL_METHOD=cosine`、`ANNEAL_POINT=0.72`、`TARGET_LR_FACTOR=0.0`、`EVAL_PERIOD=20`、`TEST_BBOX_TYPE=gt` |
| CPU preflight（gdrn / smoke / bop-eval） | 三者 `status=PASS`，340 backbone 张量、`91,292,872` 可训练参数、13 objects |
| `check_lm_data` | 相机与内参误差 `0.0 px`、`bbox` 与 mask 误差 `0.0 px`、`bbox3d` 与 `models_info` 相对误差 `3.06e-08`、平移 0.88–1.10 m、在线渲染覆盖源 mask `0.954–0.989` |
| lm_imgn 13 类覆盖 | 逐类 3 个样本，`benchvise→benchviseblue` 别名生效、固定内参一致、`t_z` 0.54–1.09 m |
| test split 去 xyz_crop | `lm_13_test_online` 在 `lm/test/xyz_crop` 不存在的机器上加载成功；同一机器上旧 `lm_13_test` 因 `osp.exists(xyz_path)` 断言失败（对照） |
| 20 步 CUDA+CPP smoke | PASS，三项 loss 单调下降，`amp_skipped_steps=0`，峰值 allocated `2.820 GB` |
| 真实入口 1 epoch（`main_gdrn.py`，CPP） | 退出码 0，写 `model_epoch_001.pth`（1.46 GB）；checkpoint 内 `iteration=3`（4 步）、optimizer `step=4`、436 个状态张量、`base_lrs=[1e-4, 1e-5]`（backbone `LR_MULT=0.1`）、scheduler `_last_lr=4.996e-7`（warmup 第 4 步的理论值）、GradScaler `scale=65536` 且 `_growth_tracker=4`（无跳步）、backbone 张量与 ImageNet 源不再逐位相等（确有优化步写入） |
| 回归 | `pytest -q research` `248 passed` |
| `run_contract` formal 模式 | `FORMAL_READY=False` → 拒绝；置 `True` 后 PASS（`training_renderer=egl`、`evaluation_renderer=cpp`、`evaluation_period=20`） |

### 最终 CUDA + EGL smoke：BLOCKED

本机 EGL 实测不可用：`lib/egl_renderer/glutils/egl_offscreen_context.py:226`
抛 `RuntimeError: Bindless Textures not supported`（与 EXP021 记录的本机限制一致）。
按任务书 §6 不伪造 PASS，因此：

```text
BLOCKED: final CUDA/EGL smoke must be run on the training server.
```

正式配置的 `RESEARCH_PROTOCOL.FORMAL_READY` 保持 `False`；`train_lm13_gdrn.py` 末尾以注释
形式留下 override（`SCHEDULE="configurable", FORMAL_READY=True`），需在服务器 EGL smoke
通过后才打开。本轮结论：`NO-GO`，唯一 blocker 是服务器 EGL 渲染器。

## 2026-09-18 服务器集成与 launcher 收口（Observed）

按 `EXP023_LM13_Server_Integration_Modification_Task.md` 补齐服务器侧数据、权重、cache、
mount 与 runtime gate；未改 Progressive PCC 主体、LM13 protocol 或训练数学。

- `docker/l40/experiment.sh`：`create` 新增 `${root}/datasets/lm_imgn` 只读 mount 与
  `GDRN_CONVNEXT_BASE_WEIGHTS` 注入；host `${root}/datasets/lm_imgn` 与 repo 侧
  `datasets/lm_imgn` 在 create 前建成目录，使旧 LMO 容器不被空目录阻塞（LM13 gate 仍会
  因空数据拒绝）。`verify_required_mounts` 增加该 mount 的强制核对。
- `check_host()` 拆成通用项（user/Docker/GPU/`${root}` 基础目录），数据集与权重移入
  profile gate。profile 从容器内 `mmcv.Config.fromfile` 读 `TRAIN_PROTOCOL.NAME` 得到
  （`lm13_gdrn`/`lm13_real_only` → `lm13`，`lm13_pbr` → `lm13_pbr`，缺失/其他 →
  `legacy_lmo`），不按文件名或 EXP 编号猜。
- 新增 `research/exp022/server_preflight.py`：在容器内报告 profile、train/test split、
  object IDs、hierarchy、BOP targets、LM real、`lm_imgn`、VOC、ConvNeXt 以及各 split
  记录数；路径失败时以 JSON FAIL 退出。LM13 两档的 runtime gate 会在
  `validate_run_config` 之前调用它，因此失败发生在 run 目录创建之前。
- `research/tests/test_experiment_launcher.py` 由 16 增至 40 个用例（含参数化；旧测试保留，
  只同步更新了两个断言），`research/exp022/tests/test_lm_protocol.py` 由 24 增至 27。

### 本地验证（Observed）

`server_preflight` 在本机实测（非服务器容器）：`lm13_gdrn` → `profile=lm13`，
`lm_13_train_online=2375`、`lm_imgn=13000`、`lm_13_test_online=13425`（与预期一致，冷启动
47 s）；`lm13_pbr` → `profile=lm13_pbr`、`lm_pbr_root=.../train_pbr`，不查 `lm_imgn`；
未设 `GDRN_CONVNEXT_BASE_WEIGHTS` 与传入非 LM13 配置都返回 `status=FAIL` 并给出原因。

launcher 侧全部使用模拟容器（`present` 路径表 + 假的 `docker exec`/`printenv`/python 入口），
覆盖：`lm_imgn` mount 必须只读、ConvNeXt env 注入、四种 profile 映射、各 profile 的资源
分派、LM13 缺 `lm_imgn`/ConvNeXt/hierarchy/LM real 时拒绝、`server_preflight` 非零退出时
拒绝、`lm13_pbr` 不要求 `lm_imgn` 但仍要求 `train_pbr`、legacy 门保持
`lm/train_pbr`+`lmo/test`+VOC+`lmo_pbr/model_final_wo_optim.pth`、以及资源门在 run 目录
创建之前。`pytest -q research` **275 passed**。

### 未验证项（不伪造）

本地 Agent 不连接服务器，**没有**在 lab0/lab1 上创建容器或运行 runtime gate。以下必须由
用户在服务器完成，本轮不标 PASS：真实 bind mount（含 `lm_imgn`）、容器内
`GDRN_CONVNEXT_BASE_WEIGHTS`、`server_preflight` 在容器缓存下的行为、以及 §EGL smoke。

## 2026-09-18 launcher 收口（Observed / Derived）

按 `EXP023_Final_Server_Code_Closure_Task.md` 修四处剩余工程问题，未动 PCC 网络主体、LM13
protocol 数学与路径约定。

- **`check_host` / `create` 职责冲突**（Derived：`create` 先调 `check_host`，再 `mkdir -p`，
  而 `check_host` 当时要求 `datasets/weights/outputs/cache/home` 已存在，干净 profile 的
  第一次 `create` 必然提前失败）。现在 `check_host` 只查 user / Docker / GPU / `${root}`
  本身；`${root}` 下的运行目录（含 `datasets`、`weights`、`cache/gdrnpp_datasets`、
  `home/.cache`）由 `create` 建立。`create` 仍不生成真实数据内容（`lm/test`、
  `VOC2012/JPEGImages`、`lm_imgn/imgn` 一律不建），这些由 profile 资源门验证。
- **`lm13_real_only` 不再被要求提供 `lm_imgn`**：该臂与主实验共用 `lm13` profile，但
  `DATASETS.TRAIN` 只有 `lm_13_train_online`。新增 `container_config_list()`（读容器内
  `mmcv.Config.fromfile` 的序列值）与 `lm13_uses_deepim_renders()`，`require_lm13_resources`
  按实际 TRAIN split 决定是否调用 `require_lm_imgn_data`，与 `server_preflight.py` 的口径一致。
- **未知 `TRAIN_PROTOCOL.NAME` 改为 fail-closed**：原先 `*) → legacy_lmo` 会把
  `lm13_typo` 之类的名字静默当成旧 LMO 契约，用错资源清单。现在只有空名映射到
  `legacy_lmo`，其他非空名直接 `unknown TRAIN_PROTOCOL.NAME` 退出。仓库扫描确认现存合法
  非空名只有 `lm13_gdrn`、`lm13_real_only`、`lm13_pbr`（`eval_lm13_bop.py` 与两个 smoke
  配置继承其中之一），并加了一条测试断言"仓库声明的每个名字都能解析到 profile"。
- **RUNBOOK 的 hierarchy 准备改为 container-safe**：不再指导在 host release 目录直接跑
  项目 Python，改为（A）复制本地已验证的 `independent_v2.npz`，或（B）`create` 之后在容器内
  `docker exec ... python -m research.exp022.build_hierarchy` 生成，并明确方案 B 之后必须
  重跑 `server_preflight` 与 `preflight`。

### 验证（Observed）

| 检查 | 结果 |
|---|---|
| `pytest -q research` | `285 passed`（上轮 275；launcher 测试 40 → 50 用例） |
| 三个配置的真实 `TRAIN_PROTOCOL.NAME` / `DATASETS.TRAIN` | 用 launcher 内嵌的同一段 Python 直跑：`lm13_gdrn`→两个 split、`lm13_real_only`→仅 `lm_13_train_online`、`lm13_pbr`→`lm_pbr_13_online_train`；EXP017 legacy 配置→空名 |
| `lm13_real_only` 无 `lm_imgn` | PASS；同条件下 `lm13_gdrn` 仍 FAIL |
| `lm13_pbr` 无 `lm_imgn` | PASS；缺 `train_pbr` 仍 FAIL |
| legacy LMO 无 `lm_imgn` 内容 | PASS；缺 `lmo_pbr/model_final_wo_optim.pth` 仍 FAIL |
| 未知非空名 | `lm13_typo` / `some_future_protocol` / `lm13_gdrn_v2` 均 FAIL 且报 `unknown TRAIN_PROTOCOL.NAME` |
| 干净 profile 的 `check_host` | `${root}` 存在但 `datasets/weights/outputs/cache/home` 全不存在时 PASS；`${root}` 缺失时仍 FAIL |

### 未验证项（不伪造）

同上：本轮仍未连接服务器。真实 bind mount、容器内 runtime gate、容器内
`server_preflight`/`preflight` 与 EGL smoke 都未执行，不标 PASS。

## Decision / 待完成

- 本轮只做服务器集成，**未产生任何精度结论**，未运行正式训练，未在服务器上运行任何命令。
- 两条评估链（legacy ADD(-S)、BOP AR）都只能用 **GT bbox** 跑：官方 Faster R-CNN 的
  `lm/test/test_bboxes/bbox_faster_all.json` 全盘不存在（GDR-Net README 说明它需从单独的
  `image_sets`/`test_bboxes` 网盘包补齐）。因此 Protocol A/B 当前都是诊断口径，
  **不能**与 GDR-Net 论文的 detector-bbox 数字直接比较。
- LM13 没有 reference 模型，`matched_pnp_eval.py` 目前只能做协议校验；正式 matched 对比
  需等待 EXP021 B/C 结果完整后确定 comparator。
- 正式训练前只剩服务器侧事项：按 `RUNBOOK_CN.md` 的 “EXP023 LM13 server profile” 在
  lab0/lab1 准备 host 数据与 ConvNeXt 权重，`create` 新容器后确认 runtime gate 与
  `server_preflight` 通过，再用 EGL 跑通 smoke（forward / loss / backward /
  optimizer step / 梯度有限 / checkpoint 写入）。通过后回本地打开
  `train_lm13_gdrn.py` 的 `RESEARCH_PROTOCOL.FORMAL_READY=True`，重新 bundle/release，
  再启动 formal。T-LESS 数据尚未准备，其 variable-S 对称监督是单独的后续工作。
- 已发现但本轮未处理：`obj2label` 的反向映射写法在约 40 个其他 loader 中依旧（全仓库无读取
  点，故无行为影响）；`det/yolox/` 下同名文件同样未动。需要时另开 cleanup。
