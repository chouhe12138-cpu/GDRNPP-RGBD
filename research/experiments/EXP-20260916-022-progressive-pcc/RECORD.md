# EXP022 渐进式层级 CAD 对应与多尺度 PCC

- `experiment_id`: `EXP-20260916-022-progressive-pcc`
- 状态：`STAGE1_IMPLEMENTED / LOCAL_CPP_SMOKE_PASS / SERVER_EGL_PENDING / FORMAL_NOT_STARTED`
- 日期：2026-09-16
- 主训练 `run_id`：未分配；source commit：待本地提交后填写；seed：42

## 问题、变量与范围

在冻结官方 ConvNeXt、RGB/PBR40/GT-box 不变条件下，以 8⁴ CAD 路由、多尺度 PCC、局部球形有界残差和显式 RANSAC-PnP，测试是否较 EXP021 的 64→64 层级降低切向/重投影 correspondence error 并改善 matched pose。主变量是完整 EXP022 PCC 结构。`independent_v1.npz` 是层级来源消融，只做实现 smoke；本阶段不进入全量 backbone 训练。

主臂 `train_reused.py`，正式训练 40 epoch、batch 48、16 workers、AdamW `3e-4`、4% warmup 后 cosine、AMP FP16，E5/E10/E15/E20/E25/E30/E35/E40 为预定正式评估点。EXP021 comparator 选择待其结果完整后确定，不据现有 E15 指标事后指定。

## 评价与 gate（预注册）

- Integrity：层级 8⁴/对象顺序/叶子映射、官方 340 个 backbone 张量加载、冻结/梯度隔离、四级概率归一化/路径合法、残差半径、可见 mask、GT 几何和 fixed support 评价链。失败则不能进入机制判断。
- Mechanism：四级 GT-parent/top1/top2 路由、错误累积、叶子误差、对称等价下的 correspondence mean/median、normal/tangent、重投影，以及同一固定支持集的 RANSAC solve/BOP/ADD；比较 EXP021 选定臂和官方 A。文档要求切向与重投影改善方向一致、PnP 能稳定消费、Beam K2 有恢复效应；未给硬性数值阈值，不能事后用单次细小差异宣称稳定机制成立。
- Advisory：每项 loss/梯度尺度、gate collapse、batch48 显存/吞吐、native-own-mask 结果；资源不可接受时不启动正式训练，先记录测量和处理。native support 不替代主 fixed-support 结论。
- Reproduction：与 EXP021 相同官方 checkpoint、数据/GT box、RANSAC seed/threshold/iterations 和固定 support。EXP021 B/C 源码、checkpoint、epoch 与指标需在正式比较时明确记录。

## Observed（本地工程检查，不是正式科学结果）

- `reused_v1.npz` 已生成至 `.local/dataset_cache/exp022/`，不进入 Git。主层级每对象四级节点数为 8/64/512/4096；主层级 4096 个原叶子索引均唯一，叶子 anchor 保持 EXP021 数值。独立采样初版 `independent_v1.npz` 存在 level4 零半径，未通过 loader；文件保留于 ignored cache、不用于实验。修复后生成 `independent_v2.npz`，单点 cell 半径下限 `1e-5 m`。
- EXP022 单元/配置测试：9 passed；完整 `pytest -q research`：197 passed。主臂与独立消融臂官方 backbone CPU preflight 均 PASS，加载 340 个张量，仅 PCC 的 3,684,168 参数可训练，三项 loss finite，反传/优化步和推理形状通过。该 CPU synthetic 不是实际数据性能。
- 本机 RTX 4060/CUDA+CPP 真实 PBR batch 4，两步 AMP smoke PASS，无跳步；loss route `2.1083→1.7396`、residual `0.4414→0.4470`、mask `0.7184→0.4296`；固定 batch 的两步 optimizer 时间 `1762.2/168.1 ms`，峰值 allocated `0.8862 GB`。首步含初始化/预热，不能将两步中位数视为稳定训练吞吐。
- 同机 batch 48 两步 AMP smoke PASS，无跳步；loss route `2.1264→1.8956`、residual `0.4757→0.4927`、mask `0.7648→0.3478`；固定 batch 的两步时间 `1600.3/669.5 ms`，峰值 allocated `3.6321 GB`。这不包含每步 DataLoader 和 renderer，也不是服务器 EGL 总耗时。
- 独立的本机 batch48 同一真实 batch 10 步 AMP 检查 PASS，无跳步；第 2–10 步固定 batch 优化时间中位数 `660.1 ms`（范围 `650.9–667.5 ms`），峰值 allocated `3.6342 GB`。loss route `2.1191→1.5583`、residual `0.4635→0.4249`、mask `0.7253→0.1029`。重复同一 batch 只作优化/性能检查，不代表全数据收敛，也不含每步加载与渲染。
- 独立消融 `independent_v2.npz` 的本机 CUDA+CPP batch4 两步 AMP smoke PASS，无跳步；第二步固定 batch 优化耗时 `195.2 ms`，峰值 allocated `0.9977 GB`。不进入正式 40 epoch。
- LM-O 单目标 fixed-support evaluator 接线 smoke COMPLETE：官方 A 和随机初始化 PCC 均生成固定支持集 correspondence、原生分辨率四级路由、PnP、own-mask supplemental 与计时字段；随机 PCC 数值不进入科学结论。最终接线输出在 ignored `output/experiments/exp022-evaluator-random-wiring-smoke-3/`；第一次索引错误的失败产物和第二次旧口径输出均保留，不参与正式结果。
- 服务器 CUDA/EGL 真 batch、稳定 batch48 profile、正式训练、完整 matched PnP/BOP：未运行或未生成。所有 E5–E40 指标：未生成。正式 checkpoint/运行目录：未生成。

## Decision

当前只完成第一阶段实现与本地 CPU 工程验证；正式训练启动需真实 batch CUDA/EGL smoke 和 batch48 时间/显存检查。未作机制通过或失败结论。
