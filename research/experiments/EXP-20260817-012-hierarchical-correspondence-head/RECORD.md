# EXP-20260817-012 层级密集 Correspondence Pose Head

## 当前状态

- 状态：`AUTHORIZED / LOCAL_PREFLIGHT_PASS / NO_SERVER_RUN`。
- 本地只完成代码、配置、official warm-start、CPU/CUDA forward/backward、单步 optimizer 与
  checkpoint 严格重载验证；没有 smoke、audit48 或 formal 服务器运行证据。
- 当前结构合理性属于待固定 Epoch 40 证伪的研究假设，不是已验证性能结论。

## 研究动机

EXP007 表明 dense XYZ 包含姿态信息，但 official Patch-PnP 对 ROI2D 空间对应利用
很弱，且更好的 XYZ 不能稳定改善 pose。EXP009 的 CPM 在输入端把 correspondence
压缩为 Region-conditioned 低阶 moments，固定 Epoch 40 screen fail。EXP011 进一步
证明 XYZ–Region 不一致是 GT-XYZ oracle 恶化的重要污染因素，但没有证明 Region
应完全删除，也没有证明 mismatch 是 CPM 欠佳的唯一原因。

EXP012 因此只检验：在任何全局压缩前保留逐像素 XYZ↔ROI2D 配对，先学习局部
关系、再进行层级空间聚合，能否改善冻结 geometry 输出的姿态信息消费。

## 冻结结构

```text
metric XYZ3 + absolute ROI2D2, 64x64
  -> predicted visible mask（任何卷积/归一化之前）
  -> 64-channel fine local residual block
  -> 96-channel stride-2 mid local residual block
  -> 128-channel stride-2 high local residual block
  -> fine global mean + mid global mean + high 4x4 coarse grid
  -> shared 2208-256-256 pose representation
  -> allo rot6d + centroid-z translation
```

Region 只通过 `64→16→64` 的零启动标量残差进入 fine stream，不定义分组或
pooling。它可能保留 predicted Region 的有用信息，但尚未证明可以消除
XYZ–Region mismatch；不得把该设计写成已验证机制。

不加入 Transformer、self-attention、kNN/EdgeConv、概率 PnP、新 loss 或新 pose
representation。backbone 与 geometry head 冻结，只训练 `pnp_net`。

## 固定训练与比较协议

- official checkpoint fresh start；SHA-256
  `bafa869d4e6c00410517ecb1add59f234ed1642e47fabcf3aa6e0e8a1b498a8c`；
- LM-PBR 全训练集、LM-O BOP test、GT bbox、seed `42`、40 epochs；
- Ranger `lr=8e-4`、weight decay `0.01`、warmup 200；
- automatic best checkpoint 关闭，正式结论只使用固定 Epoch 40；
- 主比较为 official 与 EXP005/B；报告 BOP19 AR、ADD(-S) object-macro 和逐物体；
- gate：BOP `+0.005`、ADD(-S) `+0.01`、至少 `5/8` 物体不退化。

## 已验证工程事实

本地 CPU 与 RTX 4060 CUDA FP32 preflight PASS：

```text
official shared tensors:      375
legacy PnP tensors filtered:   17
new pose-head tensors:         49
trainable parameters:      868746
full-model forward:           PASS（CPU/CUDA）
optimizer step:               PASS
strict checkpoint roundtrip:  PASS（CPU exact）
fvcore supported FLOPs:       178333952 / 64x64 sample
```

FLOPs 数值不包含 fvcore 未建模的 elementwise/GELU/nan-sanitize 操作，只作为
结构预算审计；参数量与可计数 FLOPs 均低于 1.5M / 250M 上限。

此前审查发现的 checkpoint bug 已在正式运行前阻断：旧实现把新 head 自身的
rotation/translation output keys 当作 legacy keys 删除。当前输出键与 official/CPM
完全错开，并由 strict roundtrip 测试锁定。visible mask 也已移动到任何局部混合
之前，零支持区输入变化不会再影响输出。

## 待运行

lab1 必须使用干净、已推送的 detached release。按用户 2026-08-18 的明确授权，
EXP012 在服务器 access、容器准备与 CUDA gate 通过后直接启动 formal，不要求先跑
smoke 或 audit48：

```text
access -> create（仅在容器不存在时）-> gate -> formal
```

这是运行授权和计划，不是完成事实。当前没有 lab1 access、create、gate、formal run ID、
checkpoint 或指标证据。
