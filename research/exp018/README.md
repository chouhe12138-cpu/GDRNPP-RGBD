# EXP018 — Geometry-Consistency Residual Pose Head

EXP013A 不变，增加一次 camera-frame pose-conditioned XYZ–ROI2D neural correction。
正式状态与证据见 [RECORD](../experiments/EXP-20260906-018-geometry-consistency-residual/RECORD.md)。

## 修改文件范围

- 新增 `core/gdrn_modeling/models/heads/gcr_pose_corrector.py`；
  `models/net_factory.py` 注册 post-decode corrector；`models/GDRN_double_mask.py`
  增加可选模块、optimizer 参数、final pose/loss 接线与 debug。
- `core/gdrn_modeling/datasets/data_loader.py`、`data_loader_online.py` 仅增加可选
  image_hw 元数据；`engine/engine_utils.py`、`engine.py`、`gdrn_evaluator.py`
  仅传递该参数，其他模型不收到新增 keyword。
- 新增 `configs/gdrn/lmo_pbr/research/exp018/gcr_pose/{train,smoke,eval}.py`。
- 新增 `research/exp018/{__init__.py,preflight.py,real_smoke.py,README.md}`、
  `research/exp018/tests/test_gcr.py` 与唯一 EXP018 `RECORD.md`；
  更新 `research/STATUS_CN.md`、`research/EXPERIMENT_INDEX.md` 的 EXP018 导航。
- EXP013A/EXP017 实现文件、既有 decode/loss 函数、Backbone/Geometry Head 均未改；
  保留任务开始前已有的 EXP017 收口文档修改；随后用户授权一并提交并推送 GitHub。

## 本地语义与集成决策

已对照用户压缩包的 `EXP018_STRUCTURE.md`、`INTEGRATION_NOTES.md`、
`code/gcr_pose_corrector.py`，以及本地 EXP013A、EXP017、GDRN_double_mask、decode、
loss、mapper、batcher、evaluator 和解析后的完整 config。

| 项目 | 本地实际语义与处理 |
|---|---|
| XYZ | L1 regression；class-aware 切片后 XYZ3；`(xyz-.5)*extent` 恢复 object-frame 米制坐标；不重复 denormalize，不钳制预测 XYZ 到 [0,1] |
| ROI2D | `get_2d_coord_np(..., endpoint=False)`；`(u/W,v/H)` 经过原有 affine crop；乘增强后完整图像 `(W,H)`，不是 `(W-1,H-1)`，不是 ROI 宽高 |
| 图像尺寸 | train 从实际 mapper 增强后 image.shape 取值；`XYZ_ONLINE=True` 实际使用 `data_loader_online.py`，即使训练 renderer 已关闭；offline mapper 也保留同一元数据契约；test 复用已有 im_H/im_W |
| 相机 | 使用 mapper 已随 resize 缩放的完整图像 K，与 ROI2D 同一坐标系；不假设主点是图像中心 |
| initial pose | 保留 A raw allo_rot6d + centroid_z/REL；复用 `get_rot_mat` 和 `pose_from_pred_centroid_z`；correction 放在解码后、loss/output 前 |
| train/test decode | 原有 train 使用可微 torch、test 使用 numpy allo→ego；不替换两条分支。test 返回 CPU R，correction 显式搬回输入 device；两条路径分别保持初始化等价，不声称彼此 bitwise 相同 |
| mask | 复用 `get_mask_prob`；A 的 L1 mask 为逐 ROI min/max 归一化置信度，不是校准概率。NaN/Inf→0；`confidence>.5` 为新增分支 hard support，同时保留 confidence token |
| symmetry | 继续 `PyPMLoss(PM_LOSS_SYM=True)`，根据 final R 选择等价 GT rotation；不增加逐像素 reprojection loss，也不重新定义对称对应关系 |
| evaluation | 原 evaluator 消费 final `rot/trans`；仍 GT-box、无 PnP/depth refine，输出 BOP 时由原函数把米转毫米；BOP CPP renderer 仅用于原有评估，不属于 correction 或训练 |

旧诊断包提到的 `normalized_image_points` 不在当前树；不恢复已删除诊断代码。
这里按当前 mapper/evaluator 的相同乘法语义直接实现可微张量转换。

## 张量流

```text
class-selected XYZ [B,3,64,64] + ROI2D [B,2,64,64] + Region + mask
    └─ EXP013A 原始双路径 → raw rotation [B,6], centroid/z [B,3]
         └─ 原有 decode → R0 [B,3,3], t0 [B,3]（camera frame，米）

predicted XYZ + ROI2D + visible confidence + K + actual image_hw + extent + R0/t0
    └─ 每点投影 → uv_hat；residual = uv_observed - uv_hat
         └─ 4096 个 9-D token（XYZ3/extent、uv2、clipped residual2、zcam/mean_extent、confidence）
              └─ point MLP 9→64→64；score 64→32→1；support 内 softmax
                   └─ 64-D descriptor + pose context 9→32
                        └─ 96→64→6；最后一层 zero-init
                             └─ R1=Exp(delta_rotvec)@R0；t1=t0+delta_t（一次）
                                  └─ 原 pose loss / 原 evaluator
```

新增参数 13,831，低于 0.1M；没有 sampling、跨点 convolution、Transformer、
Region feature 输入、EXP017 adapter、PnP/RANSAC/LM、depth stats、renderer 或循环迭代。
完整 64×64 对应保留，逐点激活开销不等同于参数开销；GPU 时间/显存尚待 smoke。

rotation 每分量为 `tanh(raw)*15°`，因此向量范数上限为 `sqrt(3)*15°`，
不是严格总旋转角 15°；translation 每分量为 `tanh(raw)*0.15*mean(extent)` 米。
使用 `torch.matrix_exp(skew)`：既有 `lie_vec_to_rot` 在零点计算未被保护的 sqrt，
Taylor 分支也只是一阶近似；不修改旧函数，不增加依赖。geometry/correction 用 FP32 island。

## Loss 接入（不是增加损失）

A 实际开启的只有 `loss_PM_R`、`loss_centroid`、`loss_z`，权重都是 1；
PM 是 rotation-only，`TRANS_LW=0`。因此仅把 final t 传给原 loss 不会训练平移 correction。

- `loss_PM_R` 接 final R，并保留原 symmetry 逻辑。
- final t 重新表达成与 A 完全相同的 centroid offset / relative-z，接原两项 L1 loss。
- 不叠加 initial auxiliary loss、reprojection loss 或 delta regularizer。
- 初始 R0/t0 不 detach，final loss 同时训练 A 与 correction；上游参数继续冻结。

为避免零 correction 时 inverse-project 数值舍入改变 base loss，使用与反解等价的增量公式：

```text
c0 = raw_xy * roi_wh + roi_center
ray_xy = (c0 - principal_point) / focal
raw_xy_final = raw_xy + focal * (delta_xy - ray_xy*delta_z) / t1_z / roi_wh
raw_z_final = raw_z + delta_z / resize_ratio
```

对 `|t1_z|<1e-6` 使用带符号 denominator safeguard；相机平面处 centroid 本身无定义。
这不是额外 loss，但训练早期接近零深度会放大 translation 梯度，是必须观察的风险。
zero-init 时 final pose、centroid/z 和三项 loss 均与相同权重的 A value-exact。
这里的 base 指 A **结构与初始化协议**，formal 不从 A E40 热启动，避免多训练 40 epoch。

## Support 与诊断边界

有效点 = confidence>.5 AND 有限 XYZ/ROI2D AND 有限 camera XYZ AND zcam>1e-6。
无效点在 projection、MLP **之前**以 `torch.where` 清理；不是先编码再乘 0。
learned softmax 在 support 外权重严格为零。全部无效时 descriptor/weights/delta 为零，
强制 identity correction，包括学习出非零 bias 后；A 本身仍正常接受 pose 监督。

support 外 XYZ 污染不影响 **固定 R0/t0 时的 correction**。它不保证整个网络都不受影响：
A 保持原有 soft-mask 消费方式，改变其输入可能改变 initial pose。同理 Region-free
只表示 correction 不直接读取 Region；通过 R0/t0 的间接依赖是设计的一部分。

可调用 `model(..., return_pose_debug=True)`，从 `out_dict["pose_debug"]` 读取：
`init_R/init_t`、`init_raw_rotation/init_raw_translation`、`projected_uv/observed_uv`、
pixel `reprojection_residual`、clipped `residual_norm`、`support/token_weights`、
`descriptor/raw_delta/delta_rotvec/delta_R/delta_t`、`final_R/final_t/final_centroid_z`、
`empty_support`。格点按 H×W row-major 展平。无效点的投影和 residual debug 置零，须结合 support 解读。
默认不返回 debug，不在 model 属性中缓存计算图；如需持久保存，调用方 detach。
已有只捕获 `pnp_net` 后直接 decode 的 α-sweep 会遗漏 correction，不能原样视为 EXP018 final。
之后的机制干预应调用完整 model 或独立 corrector 的公开接口，明确是否固定 initial pose。

## 手动 smoke

先做本地受限 smoke（一个真实 PBR batch、2 步 optimizer、一个 LM-O 图像前向），
检查 R/t correction 梯度、第二步内部梯度、上游冻结、checkpoint round-trip 和实际 test mapper。
它不输出正式指标，也不替代服务器一 epoch smoke。失败 run 不复用输出目录。

在本地 WSL 执行，先确认显存可用：

```bash
(
set -Eeuo pipefail
cd /home/wsluser/GDRNPP-RGBD
source /home/wsluser/miniconda3/etc/profile.d/conda.sh
conda activate pytorch22
nvidia-smi
run_id="RUN-$(date +%Y%m%d-%H%M%S)-local-smoke-s42-${RANDOM}"
run_dir="output/experiments/EXP-20260906-018-geometry-consistency-residual/${run_id}"
python -m research.exp018.real_smoke --device cuda:0 --batch-size 2 --output-dir "${run_dir}"
echo "RESULT: ${run_dir}/result.json"
)
```

产物只有 `result.json` 和 `pose_debug.pt`，均为 ignored local outputs；报错时保存 FAIL 原因。
不要用 `PYTHONOPTIMIZE` / `python -O` 禁用检查。smoke 确认后，再经用户授权提交/bundle、
新只读 release 和 `docker/l40/experiment.sh` 进入服务器流程；本次不提供虚构 commit/release，
也不提前启动 formal。服务器 `smoke.py` 保持原单 epoch 协议，`train.py` 才是 40 epoch。

## 风险与证据边界

- hard support / front-camera 筛选在边界不连续，初始坏 pose 可能令 branch 暂时无梯度；记录每样本有效点数。
- 近零初始深度、残差 clipping 饱和、attention 过度集中可能导致学习不稳定；先看 smoke，不擅自新增 warmup/loss/多步 refinement。
- symmetric 对象的 correspondence 与 initial pose 可能属于不同等价表示，显式 residual 不一定小；不要求 residual 与 ADD 线性相关。
- pose context 仍可能成为主要信号。具备显式 residual 输入不等于已证明高效消费几何，正式增益与后续有限机制对照需分开报告。
- CPU 单测不证明真实数据收敛、CUDA/AMP 正确性、batch48 显存或 BOP 性能。
