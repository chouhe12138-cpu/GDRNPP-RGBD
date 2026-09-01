# EXP008 — Stage 3C2 Joint Adaptation

## 协议

- 状态：`COMPLETE / C2_SCREEN_FAIL`
- 从官方 checkpoint 独立初始化；冻结 backbone/geometry head
- 同时训练 Patch-PnP 与 quality/coverage 模块；seed `20260731`；40 epoch
- LM-O GT-box 每 5 epoch 评估；不使用 test 中间点选论文模型

## 正式轨迹

| Epoch | BOP AR (%) | ADD macro-object (%) | ADD micro-target (%) |
|---:|---:|---:|---:|
| 5 | 69.1306 | 51.5069 | 51.4187 |
| 10 | 69.1052 | 51.6128 | 51.4187 |
| 15 | 68.8088 | 50.4537 | 50.2422 |
| 20 | 68.9209 | 49.2216 | 49.0657 |
| 25 | 69.2738 | 未取得 | 49.6194 |
| 30 | 69.1532 | 未取得 | 50.3806 |
| 35 | **69.3520** | 未生成 | 未生成 |
| 40 | 69.3006 | 未生成 | 未生成 |

官方 BOP 基线 `69.0415%`，预注册门槛为至少 `+0.50 pp`。最高观测 E35 只提高
`+0.3105 pp`，固定 E40 提高 `+0.2591 pp`，均未过门槛。历史 `model_best.pth`
在 E25 保存，不能代表最高 BOP checkpoint；正式归档使用 E40。

macro-object 是八类等权，micro-target 是 1,445 targets 等权，不能互换。E35/E40
的 ADD 文件没有生成/留存，因此保持缺失，不用 BOP `ad` 冒充。原本主要因果比较
是 C2−EXP005；现有结果足以判定未过相对官方筛选，但不声称已经精确分离模块的
额外贡献。E40 checkpoint 与日志外置于 `E:\6D姿态估计\C2\`。
