# EXP013E 运行记录

- 当前状态:`AUTHORIZED / LOCAL_GATE_PASS / FORMAL_NOT_STARTED`。
- 目标服务器:lab0。协议与 EXP013A 一致(冻结 backbone/geometry、只训 pnp 头、
  40ep、batch 48、Ranger 8e-4、seed 42、固定 epoch_040),头为官方 `ConvPnPNet`
  随机初始化,渲染器状态:**关闭**(无 CPP/EGL,冻结几何监督关闭,
  `TRAIN_SUPERVISION=False`)。

## 2026-08-29 本地门禁(PASS)

- 权重剥离:`research/exp013/e_prep.py` 从 SHA 校验的官方 ckpt 生成
  `pretrained_models/lmo_pbr/model_final_wo_optim_wo_pnp.pth`;保留 375 个
  非 pnp 张量(逐值一致)、剥离 17 个 pnp 张量;
  SHA-256 `de07b832c2b28260f7c16915790d8db71fea0e921abd10594b3e67e7b844281c`。
- 配置合并检查:train/smoke/audit48 三态均为 `ConvPnPNet + gelu + MASK=none +
  剥离权重`,几何监督关闭,渲染器 = 无(引擎不会构造 CPP/EGL)。
- pytest:`research/exp013/tests` 共 27 项全部通过(含 6 项 E 专属 + smoke 隔离
  扩展)。
- preflight `--variant E`:CPU(含严格 checkpoint 往返)与 CUDA 均通过;
  375 共享张量迁移、17 个 pnp 剥离键、随机初始化未被覆盖;
  trainable params `9,029,513`;head FLOPs `138,094,848`;
  batch1 延迟 CPU `8.13ms` / CUDA `0.52ms`;峰值显存(batch1)`578,798,080` 字节。
- 真实数据 smoke:`lmo_pbr_stage3_local_train`,batch 4,2 workers,2048/2048
  iterations,输出目录
  `output/experiments/EXP-20260829-015-e-official-head-random/LOCAL-20260829-e-no-render-smoke`;
  全程日志 0 次 renderer 出现(未创建 CPP/EGL renderer);
  final total loss `0.3981`(running mean `1.698`),checkpoint
  `model_epoch_001.pth` 已保存。

## 服务器流程

见 `research/RUNBOOK_CN.md` EXP013E 小节(gate 会先运行 e_prep 生成剥离权重再跑
preflight)。formal 启动前照例检查 lab0 空闲显存。
