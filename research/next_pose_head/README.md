# EXP012 Hierarchical Correspondence Head

该模块保留 dense metric XYZ 与 absolute ROI2D 的局部配对，在全局压缩前完成
多尺度编码。正式结果和结论边界见 EXP012 `RECORD.md`。

```bash
python -m research.next_pose_head.preflight --device cpu
pytest -q research/next_pose_head/tests
```

服务器运行统一使用 `docker/l40/experiment.sh` 和 EXP012 的 train/smoke/eval
配置；不要复用已经删除的 Stage 3C managed workflow。
