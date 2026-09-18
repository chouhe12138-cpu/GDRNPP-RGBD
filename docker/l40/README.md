# L40 Docker 环境

`Dockerfile`、requirements、vendor 和 native 构建脚本用于稳定离线镜像。vendor
校验和只保护外部构建输入，不参与实验运行身份管理。

服务器实验统一使用 `experiment.sh`。它固定 lab/GPU 映射、检查项目容器标签、
使用只读代码/数据/权重挂载并拒绝覆盖输出。具体命令见
`research/RUNBOOK_CN.md`，安全边界见 `research/SERVER_SAFETY_CN.md`。

`create` 的 mount 集合覆盖 repo、`BOP_DATASETS`、`VOCdevkit`、`lm_imgn`（EXP023 的
DeepIM 渲染图）和 `weights`，并注入 `GDRN_DATASET_CACHE_DIR`、
`GDRN_CONVNEXT_BASE_WEIGHTS` 与 `BOP_RENDERER_PATH`。`lm_imgn` 在 host 上不存在时
会被建成空目录，旧 LMO 容器仍可创建，但 LM13 的资源门会因空数据拒绝启动。

`run`/`eval` 按配置里的 `TRAIN_PROTOCOL.NAME` 选择 profile（`lm13` / `lm13_pbr` /
`legacy_lmo`）并检查对应数据与权重，LM13 两档还会在容器内运行
`research.exp022.server_preflight`。

fresh Git release 执行 `experiment.sh ... create IMAGE_REF` 时会检查 image revision
兼容性，并自动提取 Git ignored native artifacts；不需要手工复制 `.so`。

普通 Python/config 变化复用已有镜像。只有 Dockerfile、requirements、vendor、
C++/CUDA 或 ABI 变化时才运行 `build_image.sh`/`build_native.sh` 重建。
