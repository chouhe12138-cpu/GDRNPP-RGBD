# L40 Docker 环境

`Dockerfile`、requirements、vendor 和 native 构建脚本用于稳定离线镜像。vendor
校验和只保护外部构建输入，不参与实验运行身份管理。

服务器实验统一使用 `experiment.sh`。它固定 lab/GPU 映射、检查项目容器标签、
使用只读代码/数据/权重挂载并拒绝覆盖输出。具体命令见
`research/RUNBOOK_CN.md`，安全边界见 `research/SERVER_SAFETY_CN.md`。

fresh Git release 执行 `experiment.sh ... create IMAGE_REF` 时会检查 image revision
兼容性，并自动提取 Git ignored native artifacts；不需要手工复制 `.so`。

普通 Python/config 变化复用已有镜像。只有 Dockerfile、requirements、vendor、
C++/CUDA 或 ABI 变化时才运行 `build_image.sh`/`build_native.sh` 重建。
