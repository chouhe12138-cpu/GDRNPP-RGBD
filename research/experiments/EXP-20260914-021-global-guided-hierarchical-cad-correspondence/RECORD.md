# EXP021 — 全局引导的层级 CAD 对应预测

## Status

- lifecycle：`CLOSED`
- execution：`FORMAL_RESULTS_PARTIAL`
- B/C direct-pose E5–E40：完整
- run exit code、服务器 checkpoint 内容、matched PnP/K sweep：未生成或未确认
- 用户于 2026-09-19 确认训练和固定评价点结束；不在该设计上继续。

## Research question

在 RGB、官方 backbone/decoder/mask/Patch-PnP 冻结时，以固定 CAD 64×64 hierarchy、
GT-parent 子路由与球形受限 residual 替换独立 Region+XYZ；再加入 8×8 全局图像—CAD 交互，
是否能改善 correspondence 并由 matched RANSAC-PnP 转化为 pose 收益？

## Protocol

- experiment ID：`EXP-20260914-021-global-guided-hierarchical-cad-correspondence`。
- A：official GDRNPP checkpoint，仅作 GT-box matched evaluator/profile comparator。
- B：hierarchical route + bounded residual。
- C：B + 两层 global Transformer、coarse bias、零初始化 residual injection。
- B/C：LM-O PBR40、batch48、16 workers、40 epoch、Ranger 8e-4、wd .01、warmup200、
  seed42；FP16 AMP。
- 正式 source：`effc99b93eb70a9197805a427ce0156365a2bbae`；镜像
  `gdrnpp-research:torch220-cu121-sm89-c0be1ade7ea9`。
- B run：`RUN-20260914-125103-formal-s42-a01`；C run：
  `RUN-20260914-125349-formal-s42-a01`。
- hierarchy：coarse `[8,64,3]`、fine `[8,64,64,3]`；八物体对称分支数
  `[1,1,1,1,1,2,2,1]`。
- 三项正式 loss 权重由服务器 EGL 标定固定为 coarse/fine/XYZ `0.25/1/16`。

主评价预定 official A fixed support、K=1/2/4/8 matched RANSAC-PnP，并报告 correspondence、
normal/tangent、reprojection 与 pose 指标。实际只完成仓库常规 direct-pose 评价，因此 formal
结果不能解释为 matched correspondence 机制结论。

## Engineering and integrity summary

- B/C official checkpoint 只缺各自 CAD head keys，optimizer 仅包含 CAD head，冻结参数在 smoke
  中逐张量不变；可训练参数约 0.23M/2.92M（正式 source 时的结构）。
- 本机 CPP 只作接线和性能诊断；服务器 EGL 对 loss 权重、AMP 数值完整性、batch48 显存和
  Image/CAD 训练路径进行了 gate。固定 batch/profile 不作为姿态结果。
- 训练热路径曾因逐实例/逐 parent 构图和 DataLoader 并发出现性能问题；向量化、16 workers、
  AMP 与 feature-only path 修复后启动正式 run。工程优化保持 hierarchy/loss/科学变量不变。
- 后续本地加速提交 `f01dff5` 晚于正式 source `effc99b`；不能把其性能或代码状态归入本次
  formal run。
- 两份日志都到 epoch40、iteration `255919/255920[100.0%]`，记录八个固定 checkpoint 和
  `FINAL_EVAL_REUSED periodic_epoch=40`，未见 Traceback/OOM/RuntimeError；但没有 exit code，
  也未读取服务器 checkpoint 文件。

## Observed direct-pose results

| Epoch | arm | BOP AR | ADD(-S)0.1d | AR_reS | AR_teS |
|---:|---|---:|---:|---:|---:|
| 5 | B | 0.683502 | 0.525952 | 0.521799 | 0.799539 |
| 5 | C | 0.685772 | 0.498270 | 0.543483 | 0.790542 |
| 10 | B | 0.685010 | 0.526644 | 0.521107 | 0.793080 |
| 10 | C | 0.693179 | 0.560554 | 0.538639 | 0.803460 |
| 15 | B | 0.676270 | 0.489965 | 0.525952 | 0.788697 |
| 15 | C | 0.683116 | 0.491349 | 0.531488 | 0.787313 |
| 20 | B | 0.676593 | 0.511419 | 0.514879 | 0.786621 |
| 20 | C | 0.693368 | 0.547405 | 0.538639 | 0.800692 |
| 25 | B | 0.683903 | 0.506574 | 0.526413 | 0.792849 |
| 25 | C | 0.687476 | 0.532180 | 0.547866 | 0.793310 |
| 30 | B | 0.680676 | 0.500346 | 0.526182 | 0.785698 |
| 30 | C | 0.689677 | 0.523183 | 0.542099 | 0.800231 |
| 35 | B | 0.674044 | 0.476817 | 0.526413 | 0.777624 |
| 35 | C | 0.680687 | 0.485121 | 0.551326 | 0.788466 |
| 40 | B | 0.688999 | 0.525952 | 0.537024 | 0.795386 |
| 40 | C | 0.693675 | 0.534256 | 0.546021 | 0.806690 |

全部 16 个 BOP AR 点与同 epoch score JSON 一致；ADD(-S)0.1d 来自
`EVAL_SUMMARY.add_s_0.1d`，不以 BOP toolkit 的 AD 口径替代。

### C−B differences

单位为百分点；只作描述，不重选模型。

| Epoch | Δ BOP AR | Δ ADD(-S) | Δ AR_reS | Δ AR_teS |
|---:|---:|---:|---:|---:|
| 5 | +0.227 | −2.768 | +2.168 | −0.900 |
| 10 | +0.817 | +3.391 | +1.753 | +1.038 |
| 15 | +0.685 | +0.138 | +0.554 | −0.138 |
| 20 | +1.678 | +3.599 | +2.376 | +1.407 |
| 25 | +0.357 | +2.561 | +2.145 | +0.046 |
| 30 | +0.900 | +2.284 | +1.592 | +1.453 |
| 35 | +0.664 | +0.830 | +2.491 | +1.084 |
| 40 | +0.468 | +0.830 | +0.900 | +1.130 |

C 高于 B 的点数：BOP 8/8、ADD 7/8、reS 8/8、teS 7/8。B 的 BOP/reS/teS 最佳点为
E40，ADD 最佳为 E10；C 的 BOP/teS 最佳为 E40，ADD 为 E10，reS 为 E35。这些最佳点是
事后描述，不改变固定点评价。

### E40 per-object ADD(-S)0.1d

| object | B | C | C−B |
|---|---:|---:|---:|
| ape | 0.485714 | 0.491429 | +0.005714 |
| can | 0.733668 | 0.804020 | +0.070352 |
| cat | 0.415205 | 0.461988 | +0.046783 |
| driller | 0.795000 | 0.835000 | +0.040000 |
| duck | 0.333333 | 0.250000 | −0.083333 |
| eggbox | 0.427778 | 0.377778 | −0.050000 |
| glue | 0.707143 | 0.750000 | +0.042857 |
| holepuncher | 0.315000 | 0.310000 | −0.005000 |

所有 epoch 的逐物体 ADD 原始值保存在相应 `eval_summary.jsonl`。

## Interpretation / decision

Observed 层面，C 在多数固定点高于 B，但 E5 有反向指标，逐物体方向也不一致。由于专用
fixed-support matched PnP/K sweep 没有运行，本实验不裁决“全局交互改善 correspondence 并被
PnP 消费”的机制命题，也不把 direct-pose 结果包装成 matched 结论。

用户确认实验在 E40 收口，并指出现有设计不再继续。EXP022 及后续工作另行设计；本实验不补做
退出码、checkpoint 内容验证或 matched PnP，也不根据八个点追加 seed/选模。

## Evidence map

- B：[canonical run evidence](evidence/RUN-20260914-125103-formal-s42-a01/)
- C：[canonical run evidence](evidence/RUN-20260914-125349-formal-s42-a01/)
- 服务器 EGL 工程 gate：[six compact reports](evidence/8a736c8-lab0-egl/)

B/C run 目录各保留 E5–E40 八份 score JSON 和一份 `eval_summary.jsonl`。完整日志、checkpoint
和大型输出外置于 `E:\6D姿态估计\EXP021\B` 与 `C`。

## Known gaps / boundaries

- 未生成：run exit code、服务器 checkpoint 文件/内容核验、A/B/C matched K sweep、完整
  fixed-support matched PnP、机制 gate 和 batch-1 resource gate。
- 单 seed direct-pose 结果不证明跨 run 稳定性或 correspondence 机制。
- 外部 `compare/` 报告只覆盖早期 E5/E10/E15，不作为 canonical evidence。
