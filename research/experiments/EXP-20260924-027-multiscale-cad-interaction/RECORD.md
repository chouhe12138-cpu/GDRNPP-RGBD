# EXP027 — 多尺度 CAD Interaction 与 Region Query

## 身份与状态

- experiment ID：`EXP-20260924-027-multiscale-cad-interaction`。历史 Git 曾把 EXP027 用作
  EXP025 E15 checkpoint 诊断编号；该诊断现归属 EXP025 `diagnostics/`，本实验使用新的完整 ID。
- 状态：`LOCAL_VALIDATION_COMPLETE / SERVER_RELEASE_BLOCKED / FORMAL_NOT_STARTED`。
  本地实现和验证基于 `510ff38` 之后的工作树；本记录随实现一起提交后，以该提交为本地源码
  快照。未连接服务器、未运行 batch48/EGL gate，也没有正式姿态精度结果。
- 研究问题 A：在同一 T3 dense output contract 下，ConvNeXt 多尺度 lateral 特征与逐尺度
  image-query/CAD-context 交互的影响是什么？研究问题 B：在同一多尺度视觉特征上，固定身份的
  CAD region token 作 query 并沿父子树传播，相对 A 的 image-query 方向有何影响？A/B 是
  两种完整架构，B 相对 A 除 query 方向外还替换 dense classifier，不能解释为单算子消融。

## 冻结协议与实现

- 两臂共同使用 LM-O PBR40/GT box、ImageNet ConvNeXt Full、seed42、batch48、AdamW
  backbone/head 3e-4、wd .01、40 epoch、4% warmup + cosine、E5/E10/E15/E20/E25/E30/E35/E40
  固定评价、RANSAC-PnP、predicted-route residual、mask 和 EXP026 adaptive λ1 三层
  8/64/512 hierarchy，SHA256 为 `7ac75daa756ed7ae59463614737ccc46cd746173452cd1943a4a024d0817c631`。
  正式 AMP 初始 scale 候选为 **4096**，仍须两台服务器分别真实 batch48/EGL gate。
- A 配置：`configs/gdrn/lmo_pbr/research/exp027_multiscale_cad/train_a_multiscale_fpn.py`；
  ConvNeXt C1/C2/C3/C4 实测为 128@64/256@32/512@16/1024@8；P4→P1 延续现有
  512/256/128/64 decoder，每级 Image-SA 后由 image query 读取 T0/T1/T2/T3 CAD bank。
  前三级 CAD delta 复用现有 stage projection 写回下一级特征。三个 1×1 lateral 只 ADD，
  scalar alpha 初值 .01；融合的逐像素乘加在 FP32 完成后转回特征 dtype。
- B 配置：`configs/gdrn/lmo_pbr/research/exp027_multiscale_cad/train_b_cad_region_query.py`；
  与 A 共享 FPN 和 Image-SA，CAD T0/T1/T2/T3 query 逐级读取图像特征；已条件化的父 query
  通过 1→8→64→512 传播到子 query。像素/query 各经 256→256 投影，以 `/sqrt(256)`
  相似度生成 `[B,512,64,64]` logits；residual 的 CAD context 保持 geometry-only T3 bank。
  未引入自由 object query、Hungarian、hard route、beam 或 teacher forcing。
- 两臂都沿用 T3 dense NLL、FP32 logsumexp T2/T1 marginal、symmetry branch、
  predicted-route residual target、decode/PnP。正式配置的 `SERVER_RELEASE_ALLOWED` 与
  `FORMAL_READY` 均为 False；待 release 授权、真实 batch48 gate 后另行修改并建立 release。

## 本地 Observed（2026-09-24）

- `pytorch22` CPU preflight：两臂加载 ImageNet backbone 340 tensors 精确一致；hierarchy
  SHA/类别顺序匹配；forward/loss/backward、输出形状、optimizer 参数覆盖、strict state_dict
  往返通过。A 粗层 CA、B Q0→Q3 parent 路径、lateral、classifier/query projection、
  residual、mask 有非零梯度。相关回归测试 70 passed；`bash -n`、compile、`git diff --check` PASS。
- 完整参数统计见 [parameter report](evidence/parameter_report.json)：

| arm | total/trainable | backbone | head | FPN lateral | cross/query | classifier | residual+mask | Δ vs EXP026 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EXP026 adaptive | 98,352,068 | 87,564,416 | 10,787,652 | 0 | 2,110,464 | 131,584 | 99,652 | 0 |
| A | 98,524,551 | 87,564,416 | 10,960,135 | 172,483 | 2,110,464 | 131,584 | 99,652 | +172,483 (+0.175%) |
| B | 98,721,927 | 87,564,416 | 11,157,511 | 172,483 | 2,439,424 | 0 | 99,652 | +369,859 (+0.376%) |

- 共同保存的真实 LM-O PBR/CPP batch4（本地
  `.local/exp025/fixed_batch4_a01.pt`）、seed42、AMP scale4096，三臂各 100 步固定批次。
  `run_id` 分别为 `baseline-fixed-b4-a02`、`A-fixed-b4-a04`、`B-fixed-b4-a02`；完整紧凑
  步进及 t1/t2/t3/residual/mask loss、representability、XYZ 指标在对应
  [baseline](evidence/baseline_fixed_b4_s42.json)、[A](evidence/a_fixed_b4_s42.json)、
  [B](evidence/b_fixed_b4_s42.json)。三臂 batch、backbone 初始化、seed、scale 和步数匹配；
  head 参数化不同，不声称 head 初始值逐张量相同。完整本地诊断产物（含 checkpoint）位于
  `.local/exp027/`，不进入 Git。

| arm | route loss 0→100 | representability 0→100 | residual valid 0→100 | anchor/full XYZ@100 (mm) | gain@100 (mm) |
|---|---:|---:|---:|---:|---:|
| EXP026 adaptive | 12.636→0.638 | 1.58%→96.51% | 45→3676 | 6.164/5.635 | +0.529 |
| A | 12.583→0.749 | 1.06%→95.58% | 26→3635 | 6.359/5.841 | +0.518 |
| B | 12.553→2.037 | 0.48%→70.39% | 21→2664 | 8.969/8.578 | +0.390 |

- B route loss 从 step80 的 1.652 回升至
  step100 的 2.037，固定批次表现弱于 A/EXP026，应作为后续 gate 风险，而非被最终 PASS
  总标签遮蔽。固定批次只证明局部可学习性，不代表泛化或正式姿态精度。
- 最终结构的 batch4、8-step AMP smoke 均在 scale4096 PASS，0 skipped/non-finite，
  gradient/update coverage 与模型/optimizer/scaler checkpoint 往返 PASS，见
  [A smoke](evidence/a_smoke_b4_s42.json)、[B smoke](evidence/b_smoke_b4_s42.json)。
  早期 A 在 scale16384 step5、B 在 scale8192 step1 的 lateral alpha 梯度非有限；
  FP32 lateral 融合前 A 的固定100步在 scale4096 step81、scale2048 step98 失败。
  [失败摘要](evidence/)保留原样；这些 run 不进入通过证据。

| arm | forward/backward/optimizer/full median (ms) | peak allocated/reserved (GB) |
|---|---:|---:|
| EXP026 adaptive | 53.85 / 252.77 / 40.89 / 347.03 | 3.179 / 3.295 |
| A | 49.74 / 243.51 / 41.19 / 334.76 | 2.914 / 2.993 |
| B | 49.20 / 245.88 / 42.15 / 336.41 | 2.857 / 2.917 |

以上时间和显存为同一本地 RTX 4060 保存批次的工程对照；不含 DataLoader/EGL 与服务器吞吐。

## Decision / 下一步

本地结构接线和 batch4 gate 完成。A/B 的真正机制与姿态表现须由服务器 batch48/EGL gate、
完整 E5–E40 固定点评价，以及逐物体、correspondence 指标判断。EXP026 formal 尚在运行，
不得在其正式训练期间替换服务器 release/镜像。服务器 release 与 formal 仍受配置阻断；
待用户另行授权及 EXP026 运行结束后，按 `research/exp027/SERVER_PREP_CN.md` 准备 release。
