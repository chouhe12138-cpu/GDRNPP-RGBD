# EXP-20260809-009 — CPM-Head

状态：`AUTHORIZED — LOCAL PIPELINE PASS, SERVER RELEASE PENDING`

## 研究问题

在 backbone 和 geometry head 不变的前提下，在任何可学习压缩之前显式编码
可见支持上的 2D–3D 低阶联合统计，是否能让 direct pose head 更稳定地利用
XYZ 与 ROI 2D 的配对关系，并改善固定 Epoch 40 的姿态性能？

这是可证伪假设，不是已确认结论。CPM 只测试 Region-conditioned low-order
joint moments 是否足以改善几何消费；即使 CPM 失败，也不能推出 2D–3D
correspondence 本身不重要。

## 冻结协议

- 训练集 `lmo_pbr_train`，评估集 `lmo_bop_test`，GT bbox；
- 新 managed run 固定 seed `42`，40 epoch，正式比较只使用固定 Epoch 40；
- official checkpoint 初始化，冻结 backbone 与 geometry head，只训练 CPM；
- 复用 official `allo_rot6d`、`centroid_z`、pose conversion 与 loss；
- optimizer、schedule、augmentation 与 mandatory B 对照匹配；
- 不根据 LM-O test 中间 checkpoint 选择模型；
- 自动 best-checkpoint 保存关闭；中间评估只作为诊断曲线；
- 主指标为 BOP19 AR、ADD(-S)@0.1d macro-object 和 per-object；
- 机制验收复用 XYZ、ROI、Region、coverage-only、CXU-null 诊断。

## 本地执行证据

2026-08-09 已完成非正式 local chain：8192-sample moment audit、模型构建、
official 权重迁移、forward/backward、2048 micro-batch 短训、checkpoint 保存与
严格重载、完整 LM-O evaluator、BOP/ADD(-S) 索引及 8-target × 19-condition
diagnostic smoke（含 XYZ α、coverage-only、CXU-null）。
所有工程 gate 通过。local Epoch 1 的 BOP/ADD(-S) 均为 0，只证明链路可执行，
不作为方法性能证据，也不用于修改结构或超参数。

详细哈希和验收结果见 `research/cpm_head/LOCAL_CHAIN_20260809.json`。

## 服务器启动条件

1. C2 已完成 Epoch 40、最小归档和旧容器清理，该条件已满足；
2. mandatory B matched-training-protocol control 与 CPM 均使用 seed `42`、同一
   source commit 和同一 environment image，可分别在 lab0/lab1 启动；
3. formal 必须使用干净、已推送的 detached Git release，并绑定已验证的稳定
   environment image ID；两者分别记录，不要求 build-source commit 相等；
4. 新建 `lab1_chx` 后仍须依次通过 gate、smoke、batch-48 audit，才能启动 formal。

本地 checkpoint 只作为工程验收，不能当作正式初始结果或性能结果。

## 结果与结论

`PENDING`。
