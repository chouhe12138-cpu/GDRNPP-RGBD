# EXP021 — 全局引导的层级 CAD 对应预测

EXP021 检验两个机制：固定的 `64×64` CAD 层级与球形受限残差能否降低 dense
XYZ 的切向漂移，以及 8×8 全局图像—CAD 交互能否进一步改善 matched
RANSAC-PnP。V1 只做冻结阶段：RGB 输入，官方 ConvNeXt、原 geometry decoder、
mask 分支和 Patch-PnP 全部冻结；只训练新增 CAD head。

## A/B/C 协议

| 臂 | 配置 | 作用 |
|---|---|---|
| A | `a_official_eval.py` | 官方 checkpoint、GT box、matched PnP 基线 |
| B | `b_hierarchical.py` | 共享 CAD token、粗/细路由和受限残差 |
| C | `c_global.py` | B + 两层 8×8 Transformer、全局粗区域偏置与零初始化残差注入 |

B/C 除 `use_global_guidance` 外配置相同。C 的共享模块与 B 按同一 seed 初始化；
全局注入的最后一层为零初始化，因此初始 decoder 输入与冻结官方路径一致。
V1 不解冻 backbone，也不训练原有 XYZ/Mask/Region/Pose loss。正式 batch-48 训练使用
16 个 DataLoader workers；batch-4 smoke 单独覆盖为 2。B 固定在 lab0/GPU 0，C 固定
在 lab1/GPU 1，并使用相同源码、镜像、配置、seed、数据和权重并行运行。B/C 正式
训练显式启用 FP16 AMP；几何 target、nearest-anchor `cdist` 与分支聚合保留 FP32。

EXP021 的训练专用路径在 backbone 后先做可选全局增强，再只调用一次
`geo_head.forward_features()`。它不计算零权重的旧 mask/XYZ/region 输出，也不进入
Patch-PnP；C 的梯度仍经冻结 decoder 的运算回到 global-guidance 参数。推理路径保持
原 dense outputs 和 K-beam 行为。

## CAD 层级与解码

`build_cad_hierarchy.py` 对 LM-O 八个对象做确定性面积加权表面采样。粗层沿用仓库
`fps64` anchor，将表面分到 64 个 Voronoi 区域；每个粗区域内再做 FPS64，形成
4096 个叶子。artifact 保存 anchor、平均法向、`1.05 ×` 区域最大距离半径、extent、
diameter 和 BOP 完整 SE(3) 对称变换。它属于 dataset cache，不进入 Git。

共享 `10→128→256` MLP 编码
`anchor/extent + normal + relative-parent/extent + radius/diameter`。训练使用 GT 父区域
计算子区域 CE；推理在粗区域 Top-K 内计算父子联合 log probability，默认 K=4。
叶子输出为 `anchor + bounded_residual`，残差范数不会超过叶子半径。

对称监督按实例选择一条完整等价路径：对每个 BOP 对称变换同时计算 coarse CE、
GT-parent fine CE 和 XYZ Smooth-L1，选择加权总损失最小的单一分支，三项 loss 共享
该分支。这样不会逐像素拼接互不一致的对称身份。

## 训练与评价入口

先生成 hierarchy 并完成本地契约检查：

```bash
source /home/wsluser/miniconda3/etc/profile.d/conda.sh
conda activate pytorch22
export PYTHONPATH="$PWD"

python -m research.exp021.build_cad_hierarchy
pytest -q research/exp021/tests
python -m research.exp021.preflight --arm B --device cpu
python -m research.exp021.preflight --arm C --device cpu
```

服务器正式运行前，在同一真实 CUDA/EGL batch 上标定三项 loss 的 CAD-head 梯度，
再执行 B/C one-step smoke。标定器会除去当前配置权重后测量原始梯度；只在原始
梯度比超出 `[0.1, 10]` 时建议 2 的幂次权重，并要求加权后各项相对中位数位于
`[0.25, 4]`。本机 CUDA+CPP batch 的建议为 coarse/fine/XYZ = `0.125/1/16`；随后
lab0/L40 的正式 EGL batch 建议 `0.25/1/16`，因此正式 B/C 共享配置采用服务器建议。
权重变化后的 release 必须再次通过服务器 EGL 标定和 B/C one-step smoke。

```bash
python -m research.exp021.calibrate_loss_weights --device cuda:0 \
  --precision config \
  --output output/experiments/EXP021-calibration.json
python -m research.exp021.real_smoke --arm both --device cuda:0 \
  --precision config \
  --output output/experiments/EXP021-egl-smoke.json
```

若本机 EGL 驱动缺少所需 OpenGL 扩展，可显式加 `--renderer-type cpp` 做本地 CUDA
诊断。报告会写出 `formal_renderer_match=false`；该结果不能替代服务器 EGL smoke，
正式训练配置仍固定为 EGL。

训练性能使用真实 online-geometry batch 分阶段测量。诊断报告分别记录 DataLoader、
renderer、forward、backward、optimizer、总吞吐和峰值 allocated memory；本机 EGL
不可用时以 CPP 隔离模型计算，最终仍须在服务器复核 EGL 总耗时。

```bash
python -m research.exp021.profile_training --arm B --device cuda:0 \
  --batch-size 48 --renderer-type cpp --num-workers 16 --warmup 5 --steps 20 \
  --precision config
python -m research.exp021.profile_training --arm C --device cuda:0 \
  --batch-size 48 --renderer-type cpp --num-workers 16 --warmup 5 --steps 20 \
  --precision config
```

`--precision fp32` 可在相同代码和 batch 序列上生成 matched control；报告包含实际
precision、GradScaler scale、非有限/跳过 step 计数及各阶段耗时。profile 会与正式
入口一样提高文件描述符上限，以支持 16-worker DataLoader。

正式训练使用 `b_hierarchical.py` 和 `c_global.py`，唯一 run 目录由 launcher 设置。
训练 checkpoint 不含 CAD 几何 buffer，加载时仍必须提供同版本 hierarchy artifact。

主评价由 `matched_pnp_eval.py` 完成。它使用官方 A 的
`pred_visible ∩ gt_visible ∩ valid_depth` 构造跨 checkpoint 固定 support，A/B/C 只
替换 XYZ；同一 checkpoint 同时解码 K=1/2/4/8，并记录切向/法向/欧氏/重投影误差、
路由准确率、粗层错误恢复和 RANSAC 时间。完整 run 可加 `--bop-eval` 生成
BOP AR、ADD(-S)、reS、teS 与 mechanism gate。

```bash
python -m research.exp021.matched_pnp_eval \
  --checkpoint-b /path/to/B/model_epoch_040.pth \
  --checkpoint-c /path/to/C/model_epoch_040.pth \
  --output /unique/output/exp021-matched-e40 \
  --device cuda:0 --bop-eval

python -m research.exp021.profile_inference \
  --checkpoint-b /path/to/B/model_epoch_040.pth \
  --checkpoint-c /path/to/C/model_epoch_040.pth \
  --output /unique/output/exp021-profile-e40.json
```

## 预注册 gate

- B 相对 A：切向中位误差至少降低 5%；欧氏对应误差和重投影误差各不得恶化超过
  3%。这用于判断层级表示是否改善几何，而不是只改善分类。
- C 相对 B：BOP AR 与 ADD(-S) 均不下降，其中至少一项提高 3%；欧氏、切向、
  重投影三项至少一项改善 3%，且任一项不得恶化超过 3%。
- 资源：batch-1、K=4、50 次 warmup/200 次 CUDA timing 下，C 相对 A 的中位延迟
  增幅不超过 25%，峰值 allocated memory 增幅不超过 20%。参数量同时如实记录。

当前 AMP + feature-only 实现已通过本机 CUDA+CPP 与 lab0/L40 EGL smoke、梯度标定和
matched FP32/AMP batch-48 profile。首次服务器标定把 coarse 权重从 `0.125` 调整为
`0.25`，新 release 尚需复核。不能据随机初始化输出、工程性能或待替换 run 作科学
机制判断。
