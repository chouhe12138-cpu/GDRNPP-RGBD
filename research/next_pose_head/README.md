# EXP012 层级密集 Correspondence Pose Head

本目录保存 EXP012 的工程 preflight 与测试。科学协议和事实状态以
`research/experiments/EXP-20260817-012-hierarchical-correspondence-head/` 为准。

本地命令必须先激活 `pytorch22`：

```bash
PYTHONPATH=. pytest -q research/next_pose_head/tests
PYTHONPATH=. python -m research.next_pose_head.preflight --device cpu
PYTHONPATH=. python -m research.next_pose_head.preflight \
  --device cuda --skip-round-trip
```

CPU 路径执行 exact checkpoint roundtrip；CUDA 路径执行完整 GDRN FP32 forward、
pose-head backward 和单步 optimizer。两者都只使用合成输入，不训练数据集。

服务器在代码提交、push、detached release 和运行状态只读检查完成后，由用户按阶段
运行。先执行 access；只有确认项目自有的 `lab1_chx` 容器不存在、或已按运行手册明确
处置旧容器后，才执行 create，不能把下列命令当作无条件连续脚本：

```bash
docker/l40/managed_experiment.sh lab1 EXP012 access
docker/l40/managed_experiment.sh lab1 EXP012 create
docker/l40/managed_experiment.sh lab1 EXP012 gate
docker/l40/managed_experiment.sh lab1 EXP012 launch
```

按用户 2026-08-18 的明确授权，EXP012 跳过 smoke/audit48，在 gate PASS 后直接
formal；这项例外不改变其他实验的门槛。不得跳过 gate，不得覆盖已有 run 目录。
当前尚未实际执行上述服务器命令，服务器状态必须在执行时重新核验。
