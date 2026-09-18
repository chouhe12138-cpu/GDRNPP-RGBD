# 服务器安全与只读检查

本页不保存动态 GPU 或容器快照。每次操作都在服务器重新执行检查。

## 操作前

```bash
id -un
/usr/bin/docker info >/dev/null
nvidia-smi -i 0   # lab0
nvidia-smi -i 1   # lab1
/usr/bin/docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'
```

- `lab0` 只使用物理 GPU 0，`lab1` 只使用物理 GPU 1。
- 发现未知同名容器或路径权限异常时停止，不猜测资源归属。GPU 可与其他任务共享；
  launcher 默认要求至少 `12000 MiB` 空闲显存，可按任务用
  `GDRN_MIN_FREE_GPU_MB` 显式覆盖，其他 compute process 本身不构成拒绝条件。
- 不运行 `sudo docker`、`docker system prune`，不停止、删除或重命名其他容器。

## 项目隔离

- 容器名为 `gdrnpp_chx_lab0` 或 `gdrnpp_chx_lab1`，且必须带
  `gdrnpp.project=GDRNPP-RGBD` 和对应 machine label。
- repo、BOP/VOC dataset、`datasets/lm_imgn` 和 weights 只读挂载；output、cache、项目
  home 外置可写。`lm_imgn` 是 EXP023 主协议的数据源，对应 host
  `${root}/datasets/lm_imgn`，缺失时 `/workspace/gdrnpp/datasets/lm_imgn` 必须是空目录，
  否则会静默改变训练数据。
  dataset cache 固定由 `GDRN_DATASET_CACHE_DIR` 指向项目 cache，通用 runtime cache
  由 `XDG_CACHE_HOME` 指向同一外置 cache mount。
- `GDRN_CONVNEXT_BASE_WEIGHTS` 由 launcher 固定为
  `/workspace/gdrnpp/pretrained_models/convnext/convnext_base_1k_224_ema.pth`；不要用
  host 路径覆盖它，也不要在配置里写 `/data/labs/...`。
- 容器只暴露分配给该账户的单张 GPU，容器内显示为逻辑 GPU 0。
- 不修改宿主机 Python、CUDA、驱动、账户、权限或共享数据。

安全入口：`docker/l40/experiment.sh`。脚本拒绝输出覆盖和非项目容器，也不会执行
stop/remove/prune。

`create` 先执行 image/source compatibility gate，再从兼容镜像自动 hydration
Git ignored native artifacts；release 中不手工复制 `.so`，源码 mount 始终只读。

`run/eval` 在创建 run 目录前执行统一 runtime gate：复核 ownership 和全部 bind
mount、output/cache 可写、单卡 CUDA、环境与 native verifier，在容器内用
`mmcv.Config.fromfile` 加载目标配置，再按配置里的 `TRAIN_PROTOCOL.NAME` 选择资源门
（`lm13` / `lm13_pbr` / `legacy_lmo`，见 `RUNBOOK_CN.md`）。LM13 两档另外在容器里跑
`research.exp022.server_preflight`。任一项失败都不会启动训练/评估，也不会创建输出目录。
