# 实验运行手册

## 本地检查

```bash
source /home/wsluser/miniconda3/etc/profile.d/conda.sh
conda activate pytorch22
PYTHONPATH="$PWD" pytest -q research/tests research/next_pose_head/tests \
  research/exp013/tests research/diagnostics/pose_structure/tests
python -m research.next_pose_head.preflight --device cpu
python -m research.exp013.preflight --variant A --device cpu
```

其他 EXP013 变体把 `A` 改为 `B/C/E/F`。D 仍暂停，不运行 formal。

长期 PnP-only matched control 使用：

```bash
docker/l40/experiment.sh lab0 run EXP-20260731-005-pnp-only-control \
  configs/gdrn/lmo_pbr/research/controls/pnp_only/smoke.py smoke
```

正式训练时把配置换为 `train.py`。该入口按当前代码体系维护，用于未来实验的统一
matched comparison，不替代历史 EXP005 的精确复现。

## 历史实验精确复现

已退出 HEAD 的历史实验不要把旧 config 单独复制回当前 core。先从对应
`research/experiments/*/RECORD.md` 读取 source commit，再用独立 worktree 恢复
当时代码和配置，例如：

```bash
git worktree add ../gdrnpp-exp009 652d7fd9d38f8ea5cea0c5a98cc9477b66623180
cd ../gdrnpp-exp009
```

复现完成后删除 worktree，不把历史执行框架重新并入当前 main。

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

fresh bundle release checkout 后，`create IMAGE_REF` 会先核对镜像 revision 与当前
native/环境输入是否兼容，再自动从镜像补齐 Git ignored native artifacts；无需手工
复制 `.so`。标准流程保持为 bundle/checkout → `create` → `run`/`eval`。

独立评估使用：

```bash
docker/l40/experiment.sh lab0 eval EXP-... configs/.../eval.py \
  /data/labs/lab0/docker_data/chx/outputs/experiments/EXP-.../RUN-.../model_epoch_040.pth
```

`run`/`eval` 自动建立唯一输出目录，后台执行并写 `console.log`、`exit_code`。启动后
不要修改服务器 checkout 或镜像。完成后把关键指标、checkpoint 文件名/epoch、
run ID、源码 commit 和结论写入对应 RECORD；不要提交 checkpoint 或完整日志。
