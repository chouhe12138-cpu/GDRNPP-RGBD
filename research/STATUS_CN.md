# 当前研究状态

最后核对：2026-09-20。

## EXP025

状态：`SERVER_GATE_PASS_AT_32768 / FORMAL_READY / FORMAL_NOT_STARTED`。

当前模型使用四级 Image-SA 写回、统一 T3=512 分类、T2/T1 marginal NLL、预测 soft-T3
条件化的 Residual V2、可见 mask BCE 和固定 `consistent_v3` hierarchy。optimizer resume、
GradScaler 跳步时 scheduler gate 与可配置 `SOLVER.AMP.INIT_SCALE` 已接入生产训练路径。

正式比较包含两个明确 arm：

| arm | server | initialization | backbone |
|---|---|---|---|
| `official_frozen` | lab0 / GPU 0 | 原 GDRNPP LM-O checkpoint | frozen |
| `imagenet_full` | lab1 / GPU 1 | ConvNeXt ImageNet checkpoint | trainable，LR×0.1 |

两臂共享 LM-O PBR40、GT-box、seed42、真实 batch48、AdamW、40 epoch、E5–E40 固定评价点
和相同 EXP025 head。初始化和训练范围同时变化，不能作单因素冻结消融。

2026-09-20 两臂各在服务器跑完 EGL 真实 batch48 gate：65536 在**两臂同一步、同一张量**
（`cad_attention_head.mask_predictor.weight`）溢出，32768 两臂均 PASS（0 skipped step、
四级 Image-SA 90/90 参数更新、checkpoint 往返一致）。共同 scale = **32768**，已写入
`common.py` 的 `SOLVER.AMP.INIT_SCALE`，`FORMAL_READY=True`。

同期做了一次 formal 前的协议修订：`train_imagenet_full.py` 的 `BACKBONE_LR_MULT` 由 `.1`
改为 `1.`（主干与头同为 AdamW 3e-4）。依据是官方 LM-O PBR 配方对本干不加乘子
（`GDRN_double_mask.py:912-914`，base lr 8e-4），而原来的 3e-5 继承自 LM13 的 GDR-Net 协议、
无实验证据支持；否则"全量训练"臂的主干几乎不更新，组合对比的信息量会下降。该改动只影响
lab1 臂。

剩余两步：**lab1 需要在最终配置（主干 3e-4）下重跑一次 gate**（gate 报告记录 optimizer lr），
然后两臂启动 formal。第二轮不需要重建镜像。

LM13 后续臂状态：`PREPARED / LOCAL_CPU_AND_CUDA_SMOKE_PASS / SERVER_NOT_ENABLED /
FORMAL_NOT_READY`。配置为 ImageNet ConvNeXt 全量训练，沿用历史 GDR-Net LM 协议：real LM +
DeepIM renders、Ranger 1e-4、backbone LR×0.1、effective batch24、160 epoch、1000-step
warmup、72% 后 cosine 到0、每20 epoch checkpoint/eval。LM-O 完成前不把 `exp025_lm13`
加入 launcher。

历史 EXP000–024 已退出 HEAD 的执行面。所有原始状态、结论和证据缺口继续由
[实验索引](EXPERIMENT_INDEX.md) 链接的 RECORD 保存。
