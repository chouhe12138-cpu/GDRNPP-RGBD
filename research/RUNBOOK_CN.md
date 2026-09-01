# 实验运行手册

## 本地检查

```bash
source /home/wsluser/miniconda3/etc/profile.d/conda.sh
conda activate pytorch22
PYTHONPATH="$PWD" pytest -q research/next_pose_head/tests research/exp013/tests \
  research/diagnostics/pose_structure/tests
python -m research.next_pose_head.preflight --device cpu
python -m research.exp013.preflight --variant A --device cpu
```

其他 EXP013 变体把 `A` 改为 `B/C/E/F`。D 仍暂停，不运行 formal。

## 服务器

先读 `research/SERVER_SAFETY_CN.md`，确认代码 checkout、数据、权重、镜像和 GPU。

```bash
docker/l40/experiment.sh lab0 check
docker/l40/experiment.sh lab0 create gdrnpp-l40:stable
docker/l40/experiment.sh lab0 run EXP-... configs/.../smoke.py smoke
docker/l40/experiment.sh lab0 run EXP-... configs/.../train.py formal
docker/l40/experiment.sh lab0 status
docker/l40/experiment.sh lab0 logs EXP-.../RUN-...
```

独立评估使用：

```bash
docker/l40/experiment.sh lab0 eval EXP-... configs/.../eval.py \
  /data/labs/lab0/docker_data/chx/outputs/experiments/EXP-.../RUN-.../model_epoch_040.pth
```

`run`/`eval` 自动建立唯一输出目录，后台执行并写 `console.log`、`exit_code`。启动后
不要修改服务器 checkout 或镜像。完成后把关键指标、checkpoint 文件名/epoch、
run ID、源码 commit 和结论写入对应 RECORD；不要提交 checkpoint 或完整日志。
