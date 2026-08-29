# EXP013E 运行记录

- 当前状态:`AUTHORIZED / LOCAL_GATE_PASS / FORMAL_NOT_STARTED`。
- 目标服务器:lab0。协议与 EXP013A 一致(冻结 backbone/geometry、只训 pnp 头、
  40ep、batch 48、Ranger 8e-4、seed 42、固定 epoch_040),头为官方 `ConvPnPNet`
  随机初始化,渲染器状态:**关闭**(无 CPP/EGL,冻结几何监督关闭,
  `TRAIN_SUPERVISION=False`)。

## 2026-08-29 smoke v1 失败记录(保留证据)

- `RUN-20260829-063652-smoke-s42-a01`:v1 机制(pnp 剥离派生权重
  `model_final_wo_optim_wo_pnp.pth` 由 gate 生成)在服务器 smoke 时报
  `Checkpoint ... not found!`——gate 与 smoke 之间的容器文件状态不可靠。
- 结论:放弃派生文件方案,E 机制改为 wrapper 键名错位保护(见 DESIGN v2)。
  该 run 目录保留,不计入任何结果。

## 2026-08-29 本地门禁 v2(PASS)

- 机制:`OfficialConvPnPNetRandomInit` wrapper(官方模块嵌套于 `head.*`),
  `MODEL.WEIGHTS` = 原始官方 ckpt(容器内必然存在);官方 warm start 时
  17 个官方 pnp 键被 legacy 过滤器丢弃,随机初始化不可能被覆盖;
  单测覆盖"官方键无法改写随机初始化"这一核心保证。
- 配置合并检查:train/smoke/audit48 三态均为
  `OfficialConvPnPNetRandomInit + gelu + MASK=none + 原始官方 ckpt`,
  几何监督关闭,渲染器 = 无(引擎不会构造 CPP/EGL)。
- pytest:`research/exp013/tests` 全部通过(含 E 专属 5 项 + smoke 隔离扩展)。
- preflight `--variant E`:CPU(含严格 checkpoint 往返)与 CUDA 均通过;
  375 共享张量迁移、17 个官方 pnp 键过滤、64→17 个新头张量;
  trainable params `9,029,513`;head FLOPs `138,094,848`;
  batch1 延迟 CPU `2.56ms` / CUDA `0.40ms` 级;峰值显存(batch1)`~0.55GB`。
- 真实数据 smoke:`lmo_pbr_stage3_local_train`,batch 4,2 workers,2048/2048
  iterations,输出目录
  `output/experiments/EXP-20260829-015-e-official-head-random/LOCAL-20260829-e-wrapper-no-render-smoke`;
  全程日志 0 次 renderer 构造;checkpoint `model_epoch_001.pth` 已保存。

## 服务器流程

见 `research/RUNBOOK_CN.md` EXP013E 小节(gate 走通用 EXP013 通配,直接使用
原始官方 ckpt)。formal 启动前照例检查 lab0 空闲显存。
