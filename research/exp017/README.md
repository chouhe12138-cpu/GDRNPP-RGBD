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

服务器只允许在用户确认 commit/release/GPU 后使用统一 launcher 和 EXP017 `smoke.py`；
当前不得启动 40 epoch formal。科学协议和本地结果见对应 `RECORD.md`。
