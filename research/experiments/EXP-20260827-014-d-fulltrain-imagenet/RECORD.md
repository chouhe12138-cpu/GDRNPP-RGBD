# EXP014-D 记录：ImageNet 全量端到端训练

| 项 | 值 |
|---|---|
| 实验 ID | `EXP-20260827-014-d-fulltrain-imagenet` |
| 分支 | `exp014-d-fulltrain` |
| 状态 | `AUTHORIZED / LOCAL_GATE_PASS / FORMAL_NOT_STARTED` |
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

## 最终结果

（E40 后填写：BOP AR / ADD(-S) / 逐物体 / 与 0.690399、0.683956 对比）
