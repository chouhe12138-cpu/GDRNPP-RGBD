# EXP027 — 多尺度 CAD Interaction 与 Region Query

## 2026-09-25 lab2 A gate 与 launcher 修复

- 用户提供的 lab2 `report.json` 显示 A 的真实 batch48/EGL gate `PASS`：AMP 初始 scale 4096、
  8 步、0 skipped step、checkpoint roundtrip `PASS`；峰值 reserved 20.23 GB。首次 gate
  因资源包遗漏 VOC `diningtable_trainval.txt` 失败，用户补传后得到上述通过结果。
- formal 启动在证据匹配前失败：launcher 的 `container_config_value` 对整数 `4096` 输出空串，
  导致 `gate_evidence.py` 报 `scale: invalid float value: ''`。正式训练未启动，未产生正式指标。
  本地修复配置数值读取；新 commit/release 必须重新运行同 commit 的 batch48 gate，
  lab2 容器须切换挂载到新 release。

## 2026-09-25 lab2 A 独立 release 准备

- 用户决定在 `lab2` / 物理 GPU 2 运行 A，lab0/lab1 的 EXP026 formal 不动；B 暂不放行。
  用户提供的 lab2 截图显示资源压缩包已解压并输出 `EXP027A_RESOURCES_EXTRACTED`；
  这是服务器命令成功的转述，不代表 Agent 已连接服务器核验。
- 新增 `train_a_lab2.py`，继承冻结的 A 配置，只改 output 与
  `SERVER_RELEASE_ALLOWED`/`FORMAL_READY`；共用 A 的 ImageNet Full、adaptive λ1、
  AMP4096、batch48、40 epoch、E5–E40 和 predicted-route residual。launcher 将
  EXP027-A 映射到 lab2/GPU 2，EXP027-B 仍映射 lab1 且配置继续阻断。
- 本地 `pytorch22` 的正式 run contract、EXP027 hierarchy preflight 与配置差异检查通过；
  相关 pytest `2 passed`，`bash -n docker/l40/experiment.sh` 通过。正式 source commit
  以本次提交为准；lab2 release、容器、gate、formal 尚未启动。
- 资源归档把 hierarchy 放在 `cache/gdrnpp_datasets/exp026/`，而冻结配置需要其下
  `RUN-20260921-ga-hfps-s20260919-a01/`；已给用户单文件移动命令，完成情况待回报。
  操作步骤见 [lab2 A](../../exp027/LAB2_A_CN.md)。只有 lab2 真实 batch48/EGL gate
  PASS 后才启动 formal；本地 batch4 不能替代。

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

## 2026-09-24 结构收口复核（审查基准 `d6234eb`）

### 变更与协议边界

- B 的三次 image-conditioned parent-query 传播加入可训练 scalar gate，初值均为 `0.01`。
  原 B 固定批次在 step80→100 的 route loss 回升且 representability 下降；小初值旨在降低
  早期动态父 query 对子 query 的扰动。原生 AMP 融合通过本地 smoke 与 100 步检查，未扩大
  FP32 运算范围。checkpoint sentinel、strict roundtrip、梯度和参数更新均覆盖该参数。
- EXP027 A/B 的 backbone channels `(128,256,512,1024)`、feature resolutions
  `(64,32,16,8)`、pyramid channels `(64,128,256,512)` 现由 resolved config 显式给出；
  多尺度 stage、transition、lateral 从该规格构建。现有四级 tensor 形状和 `[B,512,64,64]`
  输出保持相同。wrapper 读取 head capability 路由单/四尺度，architecture 与
  `out_indices` 仍 fail-closed。EXP025/026 legacy 配置未加这些字段。
- A/B 并非仅改变 attention 方向：B 还以 pixel/query similarity 替换 dense classifier，
  final residual/mask 使用最终 image token，而 A 使用 image-query/CAD interaction 后的
  token。两臂 residual 的 CAD context 都保持 geometry-only T3 bank。
- 保持两臂 40 epoch、E5–E40 固定评价；E15 仅观察，不设独立 screening 或停训判据。
  数据、hierarchy、predicted-route residual、T2/T1 marginal、RANSAC-PnP、优化器、
  LR、batch、seed 与原协议一致。formal/release 开关继续关闭。

### Observed：resolved config 与本地验证

- EXP027-A 相对 EXP026 adaptive 的 resolved-config 差异仅有实验/arm/output 身份、
  `TRAIN_PROTOCOL.NAME`、backbone `out_indices`、head architecture、新增三组多尺度规格、
  AMP 初值 `16384→4096`、release/formal flags；白名单 diff 和关键科学字段等值测试通过。
  A/B resolved config 仍仅在 arm、output 和 architecture 三项不同。
- `pytorch22` CPU preflight A/B PASS：hierarchy SHA 精确匹配、ImageNet backbone 340 tensors
  精确加载、forward/loss/backward finite、optimizer 覆盖完整；B 的三枚 gate 有非零有限
  gradient。参数统计见更新后的 [parameter report](evidence/parameter_report.json)：

| arm | total=trainable | head | Δ vs EXP026 | Δ vs 收口前 |
|---|---:|---:|---:|---:|
| EXP026 adaptive | 98,352,068 | 10,787,652 | 0 | 0 |
| A | 98,524,551 | 10,960,135 | +172,483 (+0.175%) | 0 |
| B | 98,721,930 | 11,157,514 | +369,862 (+0.376%) | +3 |

- 同一 `.local/exp025/fixed_batch4_a01.pt`，seed42、batch4、AMP4096、100 步，
  新 run 为 `A-config-fixed-b4-a01` 与 `B-gate-fixed-b4-a01`；紧凑原始报告分别见
  [A fixed](evidence/a_config_fixed_b4_s42.json)、[B fixed](evidence/b_gate_fixed_b4_s42.json)。
  旧报告仍保留，均为单次固定批次诊断，不构成正式姿态精度或跨 seed 证据。

| arm | route loss step80→100 | representability @100 | full XYZ @100 (mm) | median full step (ms) |
|---|---:|---:|---:|---:|
| A 原版 | 1.086→0.749 | 95.58% | 5.841 | 334.76 |
| A 收口后 | 1.253→0.811 | 95.29% | 5.903 | 328.82 |
| B 原版 | 1.652→2.037 | 70.39% | 8.578 | 336.41 |
| B gate 后 | 1.713→1.607 | 77.48% | 7.982 | 361.68 |

- B 的 step100 route loss 比原版低 `0.430`（`2.037−1.607`），representability 高
  `7.09` 个百分点，full XYZ 低 `0.596 mm`；末段 route/representability 均继续改善，
  不再出现原版后期回升。三个 gate 终值为 `0.01028/0.01640/0.01912`。A 末点 route
  高 `0.062`、representability 低 `0.29` 个百分点，记录为本地运行差异。B 本次 step
  中位数比原版高约 `25.26 ms`；本地计时不含 DataLoader/EGL，不推断服务器吞吐。
- A/B batch4、8-step AMP4096 smoke 均 PASS：0 skipped、0 non-finite、相关参数梯度及
  更新、model/optimizer/scaler strict checkpoint roundtrip PASS，见
  [A smoke](evidence/a_config_smoke_b4_s42.json)、[B smoke](evidence/b_gate_smoke_b4_s42.json)。
  EXP025/026/027 相关 pytest `52 passed`，compileall 和 `git diff --check` PASS。
  原版失败记录及原始对照不覆盖。

### Interpretation / Decision

此次单批次证据支持 B gate 缓解后期 route 恶化；仍不能推断正式泛化、跨 run 稳定性或
更高 batch 的数值稳定性。建议在 EXP026 正式运行结束、完成服务器只读检查且用户授权后，
A/B 各自执行真实 batch48/EGL 八步 gate。只有对应 arm 的 0 skipped、0 non-finite、
checkpoint roundtrip PASS 才进入其 40 epoch formal；当前 release/formal 仍阻断。
