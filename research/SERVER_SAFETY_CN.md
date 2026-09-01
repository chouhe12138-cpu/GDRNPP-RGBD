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
- 发现活动计算进程、未知同名容器或路径权限异常时停止，不猜测资源归属。
- 不运行 `sudo docker`、`docker system prune`，不停止、删除或重命名其他容器。

## 项目隔离

- 容器名为 `gdrnpp_chx_lab0` 或 `gdrnpp_chx_lab1`，且必须带
  `gdrnpp.project=GDRNPP-RGBD` 和对应 machine label。
- repo、BOP/VOC dataset 和 weights 只读挂载；output、cache、项目 home 可写。
- 容器只暴露分配给该账户的单张 GPU，容器内显示为逻辑 GPU 0。
- 不修改宿主机 Python、CUDA、驱动、账户、权限或共享数据。

安全入口：`docker/l40/experiment.sh`。脚本拒绝输出覆盖和非项目容器，也不会执行
stop/remove/prune。
