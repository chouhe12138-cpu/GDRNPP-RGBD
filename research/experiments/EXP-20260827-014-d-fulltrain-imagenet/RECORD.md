# EXP014-D 记录：ImageNet 全量端到端训练

| 项 | 值 |
|---|---|
| 实验 ID | `EXP-20260827-014-d-fulltrain-imagenet` |
| 分支 | `exp014-d-fulltrain` |
| 状态 | `PAUSED / FORMAL_A01_INVALIDATED / EGL_FIX_RETAINED / NO_ACTIVE_RETRAIN` |
| 服务器 | lab1 |
| 环境镜像 | `sha256:f3055cb660032bbb4c1b7cfd9b1840a6c98359d0562a3a4f0601f7238f7291ee`（沿用 EXP013） |
| 交付方式 | 完整 Git bundle（服务器无法连 GitHub/Gitee） |

## 本地 gate（2026-08-27，结果见 LOCAL_GATE.json）

- [x] ImageNet 权重预下载 + 校验：354,476,359 bytes（dl.fbaipublicfiles.com）
- [x] preflight CPU：PASS（全模型 95,159,884 参数全部可训练，三桶梯度/优化器更新验证，strict roundtrip）
- [x] preflight CUDA：PASS（pnp 头 batch1 3.37ms，峰值 2.06GB）
- [x] smoke（cpp 渲染器，batch 4，1 epoch = 2048 iters）：PASS，几何监督激活
- [ ] audit48：**移交服务器 L40 的 managed gate 步骤**（WSL 8GB 显存无法承载 batch 48；与 EXP013 先例一致）

smoke 摘要：loss_region 21.2→20.97（平滑均值）、loss_PM_R 0.097→0.069、loss_z 1.90→1.79，
迭代 0.292s、峰值显存 6274MB；checkpoint `model_epoch_001.pth`（iter 2047）
SHA-256 `04d86dc4c10d35eb688a22eedff61c75e8e1827a69d735177c1dd9f7be4f2a89`，
`geometry_scale_r=0.100107`、`geometry_scale_t=0.098307`（均从 0.1 获得有限更新）。

## 服务器 smoke 记录（2026-08-27）

- RUN-20260827-151207-smoke-s42-a01（source commit `310476b`）：**训练本身 PASS**——
  timm ImageNet 权重从缓存加载、95.16M 参数全量解冻确认（MODEL_SUMMARY）、2048 iters 无错误、
  checkpoint 保存成功，loss 曲线与本地 gate 一致。
- managed 的 **checkpoint-isolation 关卡 FAIL**：`verify_checkpoint_isolation.py` 按
  PNP_REPLACEMENT 冻结语义与官方 ckpt 对比，全解冻设计必然被判"冻结张量变化"（语义不适用，非训练错误）。
- **修复**：新增 `FULL_TRAIN` 隔离角色（仅要求训练产生真实张量变化，不再对照官方 ckpt 判冻结）；
  `research/managed_runtime/run.py`、`research/stage3c_runtime/verify_checkpoint_isolation.py`、
  `docker/l40/managed_experiment.sh`（EXP013D isolation_role）同步更新，bundle 重新生成。

## 正式运行计划

服务器执行顺序（用户侧）：bundle 上传 `/data/labs/lab1/docker_data/chx/transfer/`
→ clone 到 `releases/GDRNPP-RGBD-<short>` → detached checkout 40 位 commit →
`prepare_release.sh` → 先 gate/smoke（egl 渲染器验证）→ formal 40ep。

## 决策规则（预注册）

- 主对照 D E40 vs **0.690399**（官方 ckpt GT 框直评）；
- 次对照 vs **0.683956**（EXP013A E40）；
- D < 0.686 → 先查环境（timm 权重/渲染器/解冻是否真生效），不直接判结构失败；
- 决策点固定 `epoch_040`，评估点 ckpt（E5..E40）全部保留；
- seed 42 首轮，正式结论前补第二 seed。

## 2026-08-28 渲染器覆盖事故与重训决定

- formal a01（RUN-20260827-160112-formal-s42-a01）实际使用 CppTrainingRenderer：
  console.log 崩溃转储中的 renderer 对象与 `engine_utils.get_renderer` 的唯一消费
  路径 `cfg.MODEL.POSE_NET.XYZ_RENDERER`（engine_utils.py:361）均证实。D train.py
  的顶层 `XYZ_RENDERER="egl"` 是无效键；cpp 实际继承自官方 LM-O 基座配置
  `convnext_a6_..._lmo.py` 的 `POSE_NET.XYZ_RENDERER="cpp"`（含 WSL 兼容注释）。
- 旧 preflight 校验的正是同一顶层死键，因此本地 gate 假通过；smoke/audit48 的
  顶层 cpp 覆盖同样无效（碰巧与继承值一致）。egl 在本仓库服务器上从未真实执行，
  此前"egl 已验证"的记录作废。
- a01 于 2026-08-28 06:58 UTC 在 E5 评估与 checkpoint 保存后因共享 L40 被其他
  进程占满显存而 CUDA OOM 崩溃。E5 BOP AR 0.5517 仅作为中断证据保留，不进入
  任何 gate 或决策。E5 checkpoint、console.log 与 score 已归档至
  `E:\6D姿态估计\EXP-014\`。
- 修复（本 commit）：`XYZ_RENDERER` 覆盖移入 `MODEL.POSE_NET`（train.py）；
  smoke/audit48 的 cpp 覆盖同步移入 POSE_NET（否则修复后本地 smoke 将继承 egl
  并在无 GL 的 WSL 崩溃）；preflight 改为校验嵌套键并断言 `egl`。合并值已在
  本地 conda pytorch22 验证（train=egl, smoke/audit48=cpp），EXP013 回归测试通过。
- a01 继续判为无效并保留证据。EGL 修复后的从头重训方案及预注册决策规则保留，
  但用户现已暂停 D；当前没有活动重训授权，不得据旧授权自动创建或启动新 run。

## 最终结果

（E40 后填写：BOP AR / ADD(-S) / 逐物体 / 与 0.690399、0.683956 对比）
