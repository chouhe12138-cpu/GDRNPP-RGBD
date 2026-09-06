# EXP018 — Geometry-Consistency Residual Pose Head

## 问题与唯一变量

在 EXP013A initial camera-frame pose 后，用 predicted metric XYZ–ROI2D 在当前 pose 下的
显式 2D reprojection residual 驱动一次 Region-free、support-masked、轻量 6DoF correction。
不改变 Backbone、Geometry Head、数据增强、optimizer、训练时长与正式 evaluation。
细节与手动命令见 [实现说明](../../exp018/README.md)。

## 协议与运行

- experiment_id：`EXP-20260906-018-geometry-consistency-residual`。
- 当前状态：`IMPLEMENTED / CPU_PREFLIGHT_PASS / AWAITING_USER_SMOKE`；2026-09-06。
- formal/local real smoke run_id：尚未创建；真实数据 smoke 与 formal 均未执行。
- 源码：集成提交由标签 `exp018-integration-cpu-pass` 标识；起点 `5b53304`。
  用户已授权本地提交并推送 GitHub；实际训练 run 的 source commit 待启动时记录。
- 配置：`configs/gdrn/lmo_pbr/research/exp018/gcr_pose/{train,smoke,eval}.py`。
- formal seed 42；LM-PBR，40 epoch，batch48，Ranger lr 8e-4 / wd .01，warmup200。
- 与 A 相同 official `pretrained_models/lmo_pbr/model_final_wo_optim.pth` 提供冻结上游；
  A initial pose head 随机初始化；不从 A E40 或 EXP017 checkpoint 继续训练。
- 训练参数仅 `pnp_net.*` 与新增 `pose_corrector.*`；新增 13,831 参数。
- 预定正式 checkpoint/evaluation：E5/10/15/20/25/30/35/40；GT-box LM-O BOP19；
  BOP AR、ADD(-S)0.1d、AR_reS、AR_teS 保持原聚合口径；无 PnP/depth refine。
- 不创建训练 renderer；原 BOP evaluation CPP renderer 保留。
- checkpoint 文件名/epoch：尚未生成；全部正式聚合及逐物体指标：尚未生成。

## 本地验证证据

Conda pytorch22，CPU，无真实训练样本、无 renderer、无正式评估。

- 新增 21 项单测全部通过，包括实际 online mapper 的合成 resize/相机/ROI2D
  元数据链路，以及替换为合成 loader 的 smoke 编排单测。
- correction 最后一层 weight/bias 全零，初始化 final pose 与相同 A state 精确相等。
- 合成完整 `GDRN_DoubleMask` 流程的三项 loss 在零 correction 时与 A 精确相等；
  final PM_R 与 centroid/z 分别向 correction rotation / translation 输出层传入非零梯度。
- B1/B3，support 外 XYZ 为大值/NaN/±Inf，学习后仍不影响固定 initial pose 下的
  descriptor/weights/delta/final；梯度有限且无效 XYZ 梯度为零。
- empty/mixed/behind-camera support 强制 identity；非零 bias 不绕过 empty guard。
- 固定 initial pose 的 Region shuffle 不影响 correction；新增 API 不包含 Region。
- 对应关系变化会改变 residual；改变 current pose 会改变 projection/residual。
- SO(3) zero gradcheck、正交性与行列式；metric/ROI2D 解析投影；centroid/z inverse；
  CPU autocast FP32 island；batch 独立性；已更新分支和完整模型 strict checkpoint round-trip。
- 原有 99 项回归测试通过（research、EXP012/013/017/017B、pose_structure）。
  与新增测试合并执行最终为 `120 passed`；`git diff --check` 通过。
- 实际 ConvNeXt/Geometry/A/GCR CPU preflight 通过，加载官方 producer 全部张量精确一致；
  optimizer 参数集合与 trainable 集合精确一致，真实 loss forward/backward/Ranger step 有限。
  该合成 batch 有效点117；loss_PM_R=0.1651128381、loss_centroid=0.0401103832、
  loss_z=1.1669696569，仅工程检查，不是性能指标。
- 手动 smoke 脚本只完成 CLI/合成 loader 编排检查，不记作真实 smoke PASS。

## 待正式训练前确认的 gate

比较点固定 A E40：BOP AR 0.683956、ADD(-S)0.510727、reS0.498039、teS0.797693。
参考包建议 BOP 至少 +0.003、ADD 不明显下降、R/t 不显著互换、逐物体检查；
其中“不明显”的数值阈值未给定，当前不擅自写成已预注册 gate。
在 smoke 收口、formal 授权前明确 ADD/reS/teS 非劣阈值及逐物体判据，不看结果后补门槛。

## 证据边界与下一步

只证明集成与 CPU 工程正确性，不证明 geometry utilization 或正式精度改善。
support/Region 独立性是固定 initial pose 下的分支性质，不是全网络不依赖 Region/背景的声明。
重点真实风险：近零深度 inverse-centroid 梯度、empty support、clipping/weight 饱和、
对称对象 correspondence mismatch 与 batch48 激活开销。
等待用户手动 smoke 的 `result.json`；本次仅按后续授权提交/push，不启动服务器或 formal。
