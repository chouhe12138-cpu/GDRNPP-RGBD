# Pose Head 结构诊断

当前工具用于 EXP012/013 的低成本机制检查，包括 R/t oracle、分枝消融、XYZ
利用、R/t 梯度关系、接口适配和空间敏感性。诊断不会执行 optimizer step，也不把
子集指标当作 BOP 正式结果。

```bash
python tools/run_pose_structure_diagnostics.py \
  --config-file configs/gdrn/lmo_pbr/research/exp013/a_xyz_residual/train.py \
  --checkpoint /ABS/PATH/model_epoch_040.pth \
  --output-dir output/diagnostics/exp013a_e40 \
  --diagnostics d1,d2,d3,d4,d5,d6 \
  --max-batches 4 --batch-size 2 --num-workers 0 --device cuda:0
```

输出只包含 `results.json` 和 `SUMMARY.md`。完整正式结论写入对应实验 RECORD。
