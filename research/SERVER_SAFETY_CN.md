# 服务器安全与只读检查

Agent 不主动 SSH。每次由用户在目标服务器执行：

```bash
(
set -Eeuo pipefail

machine=lab0  # lab1 时改为 lab1
gpu=0         # lab1 时改为 1

test "$(id -un)" = "${machine}"
/usr/bin/docker info >/dev/null
nvidia-smi -i "${gpu}"
/usr/bin/docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'
)
```

- lab0 只使用物理 GPU 0，lab1 只使用物理 GPU 1。
- 只操作带 `gdrnpp.project=GDRNPP-RGBD` 和对应 machine label 的项目容器。
- 不使用 `sudo docker`、不执行 prune，不修改其他容器、进程、账户、GPU 分配或数据。
- repo、BOP/VOC dataset 和 weights 只读挂载；只有 output、cache、项目 home 可写。
- `GDRN_DATASET_CACHE_DIR` 固定为 `/home/gdrn/.cache/gdrnpp_datasets`；EXP025 hierarchy
  位于其 `exp025/consistent_v3.npz`，并由 preflight 校验 SHA256。
- `GDRN_CONVNEXT_BASE_WEIGHTS` 固定指向容器内 ImageNet ConvNeXt 权重。
- `experiment.sh` 只接受 `TRAIN_PROTOCOL.NAME=exp025_lmo`；未知或历史协议直接拒绝。
- `gate/run/eval` 在创建输出前检查 clean tree、ownership、mount、CUDA、环境、native、
  数据、权重、hierarchy、arm 与机器映射。脚本不提供 stop/remove/prune。

容器替换属于破坏性服务器操作，必须按 [RUNBOOK](RUNBOOK_CN.md) 的精确目标检查块执行。
