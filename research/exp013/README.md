# EXP013 Pose Head 分枝

EXP013 比较从 EXP012 correspondence 表示出发的 A/B/C，以及官方随机头 E 和
GLM-Pose-L F；D 是暂停的全量训练分枝。状态与正式结果见实验索引和各 RECORD。

本地接口检查：

```bash
python -m research.exp013.preflight --variant A --device cpu
pytest -q research/exp013/tests research/next_pose_head/tests
```

结构诊断统一使用 `tools/run_pose_structure_diagnostics.py`。它读取明确 config 与
checkpoint，只生成 `results.json` 和 `SUMMARY.md`，不替代正式 BOP 评估。
