# 服务器安全与只读检查

Agent 不主动 SSH。以下同一代码块可原样在 lab0、lab1 或 lab2 执行；机器和 GPU 由登录账户自动
解析，不需要修改变量：

```bash
(
set -Eeuo pipefail

machine="$(id -un)"
case "${machine}" in
    lab0) gpu=0 ;;
    lab1) gpu=1 ;;
    lab2) gpu=2 ;;
    *) printf 'FAIL: unsupported account: %s\n' "${machine}" >&2; exit 1 ;;
esac

/usr/bin/docker info >/dev/null
nvidia-smi -i "${gpu}"
/usr/bin/docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}\t{{.Labels}}'
)
```

- lab0 只使用物理 GPU 0，lab1 只使用物理 GPU 1，lab2 只使用物理 GPU 2。
- 只操作同时带 `gdrnpp.project=GDRNPP-RGBD` 和当前 machine label 的项目容器。
- 不使用 `sudo docker`、不执行 prune，不修改其他容器、进程、账户、GPU 分配、镜像、权限
  或数据。
- repo、BOP/VOC dataset 和 weights 只读挂载；只有 output、cache、项目 home 可写。
- `GDRN_DATASET_CACHE_DIR` 固定为 `/home/gdrn/.cache/gdrnpp_datasets`；EXP025 hierarchy
  位于其 `exp025/consistent_v3.npz`，并由 preflight 校验 SHA256。
- `GDRN_CONVNEXT_BASE_WEIGHTS` 固定指向容器内 ImageNet ConvNeXt 权重。
- `gate/run/eval` 在创建输出前检查 clean tree、ownership、mount、CUDA、环境、native、
  数据、权重、hierarchy、arm 与机器映射。
- 命令必须自动解析 bundle、release、镜像、容器和 run ID；零个或多个机械候选时停止并列出
  候选，不要求用户记忆或手填标识。
- 容器替换是破坏性操作。先执行 RUNBOOK 的只读预检；用户确认后再执行独立替换块，替换块
  会重新核对精确目标。脚本不提供 stop/remove/prune。

详细流程见 [RUNBOOK](RUNBOOK_CN.md)。
