# EXP013 几何信息利用实验

EXP013 只检验一条主线：XYZ correspondence 是否有独立、稳定的路径到达最终姿态输出。A 添加 Region-free 几何残差；B 仅在 A 上添加 masked 3×3 局部几何注意力；C 固定继承 B，并使用 rotation/translation 专用聚合与独立 latent。

## 本地 gate

```bash
python -m research.exp013.preflight --variant A --weights pretrained_models/lmo_pbr/model_final_wo_optim.pth --device cpu
python -m research.exp013.preflight --variant B --weights pretrained_models/lmo_pbr/model_final_wo_optim.pth --device cpu
python -m research.exp013.preflight --variant C --weights pretrained_models/lmo_pbr/model_final_wo_optim.pth --device cpu
pytest -q research/exp013/tests research/next_pose_head/tests
```

CUDA gate 使用同一命令并改为 `--device cuda --skip-round-trip`。真实数据 smoke 使用各变体的 `smoke.py`，batch size 4、2 workers、1 epoch，不运行正式测试集。

## E40 Three-Path 诊断

```bash
python -m research.exp013.diagnostics --variant A --weights <A_E40.pth> --output-dir <A_DIAGNOSTIC_DIR> --mode full --device cuda
python -m research.exp013.diagnostics --variant B --weights <B_E40.pth> --output-dir <B_DIAGNOSTIC_DIR> --mode full --device cuda
```

固定运行 alpha `0/0.25/0.5/0.75/1` 与三条 Region 路径：`pred` 对应 fixed_pred_region，`gt` 对应 synced_region，`zero` 对应 region0。full 模式要求 LM-O 的 1,445 个 GT-bbox targets，并验证 checkpoint 与模型状态不变。

C 的本地工程 gate 不构成服务器授权。只有 A/B 的 E40 正式精度门槛都通过，且 B 按预注册规则优于 A，才修改 C 的 `EXPERIMENT.json` 为 `AUTHORIZED` 并提交新的 metadata commit。
