# EXP017 Support-aware Rotation Residual

EXP017 完整继承 EXP013A，只在其 Region-free 8×8 geometry grid 上增加 position-aware、
valid-support masked pooling 的 rotation-only raw-output residual adapter。translation 路径
保持 A 不变，adapter 不读取 Region、main-stream feature 或 depth stats。

本地门禁：

```bash
python -m research.exp017.preflight --device cpu
pytest -q research/exp017/tests
python -m research.exp017.real_smoke --device cuda:0 --batch-size 2 --num-workers 0
```

EXP017 checkpoint 的共享 geometry autograd 诊断（只读、无 optimizer step）使用单行
命令，输出子集 re/te、adapter OFF 和 detach 干预以及共享 encoder 梯度分解：

```bash
python -m research.exp017.shared_geometry_coupling_diagnostic --config-file configs/gdrn/lmo_pbr/research/exp017/support_aware_rotation_residual/train.py --checkpoint /ABS/PATH/model_epoch_010.pth --output-dir output/diagnostics/exp017_e10_shared_geometry --max-batches 4 --batch-size 2 --num-workers 0 --device cuda:0 --seed 42
```

子集 re/te 不是正式 BOP/ADD；detach 只改变训练计算图，因此同一 checkpoint 上 normal
与 detached 的 forward 必须完全相同。

服务器只允许在用户确认 commit/release/GPU 后使用统一 launcher 和 EXP017 `smoke.py`；
当前不得启动 40 epoch formal。科学协议和本地结果见对应 `RECORD.md`。
