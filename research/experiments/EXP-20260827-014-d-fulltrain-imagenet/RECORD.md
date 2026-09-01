# EXP014-D — ImageNet Full End-to-End Training

## 协议与当前状态

- 状态：`PAUSED / FORMAL_A01_INVALIDATED / NO_ACTIVE_RETRAIN`
- 配置：`configs/gdrn/lmo_pbr/research/exp013/d_fulltrain/train.py`
- 服务器：lab1；seed `42`
- formal source commit：旧记录未留存，不能从 smoke commit 推定
- 目标：ImageNet 初始化并解冻 95,159,884 个参数，执行 40 epoch full training

本地 full-model preflight 和真实数据 smoke 曾通过：三类参数均有梯度/更新，smoke
完成 2,048 iterations，loss 有限并保存 E1 checkpoint。服务器 smoke
`RUN-20260827-151207-smoke-s42-a01` 的训练同样完成，但旧 checkpoint-isolation
规则错误套用了“冻结非 PnP 参数”语义；这是 gate 不适配，不是训练失败。

## 历史预注册规则

- 固定决策点 E40，不按 E5–E35 选模。
- 主对照官方 checkpoint BOP `0.690399`，次对照 EXP013A E40 `0.683956`。
- 若 D E40 < `0.686`，先核查初始化、renderer 和全量解冻是否生效。
- 历史方案要求首轮 seed 42 后再决定重复；当前 pause 覆盖旧启动授权，若恢复必须
  重新注册资源、renderer 和 seed 规则。

## formal a01 作废

formal `RUN-20260827-160112-formal-s42-a01` 实际继承了
`MODEL.POSE_NET.XYZ_RENDERER="cpp"`。配置把 `XYZ_RENDERER="egl"` 写在无效顶层，
旧 preflight 也只检查了这个死键，因此“EGL 已验证”的结论作废。

a01 在 E5 评估与 checkpoint 保存后，因共享 L40 被其他进程占用而 CUDA OOM。
E5 BOP AR `0.5517` 只作为中断证据，不进入 gate。随后 EGL 覆盖已移入正确的
`MODEL.POSE_NET` 层级并保留测试，但没有获得从头重训授权。

因此 EXP014-D 没有有效 E40、ADD(-S) 或逐物体正式结果；a01 不能用于结构结论。
外置事故证据位于 `E:\6D姿态估计\EXP-014\`。
