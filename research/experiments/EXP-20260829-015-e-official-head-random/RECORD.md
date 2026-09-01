# EXP013E 运行记录

- 最终状态：`COMPLETE / DIAGNOSTIC_PARTIAL_SUPPORT_M2_M3`。
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

## Formal E40 完成与诊断判读

- formal run：`RUN-20260829-080742-formal-s42-a01`，lab0，seed 42，source commit
  `d231bf634b94807e01948cc7126f3f0e6d54582e`。
- environment image：`sha256:f3055cb660032bbb4c1b7cfd9b1840a6c98359d0562a3a4f0601f7238f7291ee`；
  `MANAGED_RUN_FINISH status=PASS`（2026-08-31T12:04:43Z）。

| Epoch | BOP AR | ADD(-S) target-micro | AR_reS | AR_teS |
|---:|---:|---:|---:|---:|
| 5 | 0.655557 | 0.440830 | 0.467128 | 0.768166 |
| 10 | 0.674012 | 0.514879 | 0.502422 | 0.797232 |
| 15 | 0.672540 | 0.482353 | 0.488120 | 0.789850 |
| 20 | 0.667696 | 0.460208 | 0.488120 | 0.783391 |
| 25 | 0.666990 | 0.490657 | 0.495040 | 0.787313 |
| 30 | 0.672561 | 0.480277 | 0.522491 | 0.789158 |
| 35 | 0.679548 | 0.474740 | 0.523183 | 0.786621 |
| 40 | **0.688581** | **0.510727** | **0.535409** | **0.801153** |

- 固定 E40 的 ADD(-S) macro-object 为 `0.5129398717`。reS `0.535409` 落在
  预注册的 0.52–0.54 区间，诊断结论为「部分支撑：M2 与 M3 并重」；距离
  0.54 强支撑线 `0.004591`。E 没有 PASS/FAIL gate，不把该诊断读数改写成方法通过。
- E40 checkpoint：`519078868` bytes，epoch 40 / iteration `255919`，392 个模型
  张量，含 optimizer 与 scheduler；SHA-256
  `0df045f4238b39611d04cc550fc91067dd41547c809200b5ac23d32be649aa2e`。
- console SHA-256：`e0d0f107241282526d72063d0faaf728c6c4e1dd4a73c2074011c110cdb77d33`；
  E40 score SHA-256：`d164348318a8ed8c4d889f756ec0c77c94707db475c1818ff9be44586417cde3`。
- 外置证据位于 `E:\6D姿态估计\EXP-013\实验E\`，不进入 Git。

## 服务器流程（历史）

见 `research/RUNBOOK_CN.md` EXP013E 小节。formal 已完成，历史 run 只读，不得
重新执行 `launch` 或覆盖原输出。
