# L40 Docker 环境

服务器实验统一使用 `experiment.sh`。当前 launcher 只接受 EXP025 的 `exp025_lmo` profile，
并强制 `official_frozen → lab0`、`imagenet_full → lab1`。

repo、BOP/VOC 和 weights 只读挂载；output、cache 和项目 home 外置可写。launcher 注入
`GDRN_DATASET_CACHE_DIR`、`GDRN_CONVNEXT_BASE_WEIGHTS` 和 `BOP_RENDERER_PATH`，运行前执行
image/source、native、CUDA、资源、hierarchy SHA 与 CPU preflight gate。

普通 Python/config 变化复用稳定镜像。只有 Dockerfile、requirements、vendor、C++/CUDA
或 ABI 变化才重建镜像。完整 release、batch48 gate 和 formal 命令见
`research/RUNBOOK_CN.md`。
