# EXP-20260805-008 — C2 联合适应

状态：`COMPLETE — C2_SCREEN_FAIL`。

C2 从官方 LM-O checkpoint 独立初始化，冻结 backbone 和 geometry head，同时
训练 Patch-PnP 与质量/覆盖模块。实际协议为 40 epoch、seed `20260731`、
LM-O GT-box 每 5 epoch 评估；训练于 2026-08-10 自然完成。

## 最终周期评估

| Epoch | BOP AR (%) | ADD(-S) macro-object (%) | ADD(-S) micro-target (%) |
|---:|---:|---:|---:|
| 5 | 69.1306 | 51.5069 | 51.4187 |
| 10 | 69.1052 | 51.6128 | 51.4187 |
| 15 | 68.8088 | 50.4537 | 50.2422 |
| 20 | 68.9209 | 49.2216 | 49.0657 |
| 25 | 69.2738 | 未取得 | 49.6194 |
| 30 | 69.1532 | 未取得 | 50.3806 |
| 35 | **69.3520** | 未生成 | 未生成 |
| 40 | 69.3006 | 未生成 | 未生成 |

官方 BOP 基线为 `69.0415%`，预注册筛选门槛为至少提高 `+0.50 pp`。观测到的
最高值是 Epoch 35 的 `69.3520%`，提高约 `+0.3105 pp`；固定 Epoch 40 为
`69.3006%`，提高约 `+0.2591 pp`。两者均未过门槛，因此 C2 最终判定为
`C2_SCREEN_FAIL`。

日志只在 Epoch 25 保存了新的 `model_best.pth`，尽管 Epoch 35 的 BOP 更高；
因此历史 `model_best.pth` 不能代表最终最高 BOP checkpoint。按用户确认的失败
实验最小归档策略，只保留固定 Epoch 40 checkpoint，不使用 LM-O test 中间周期
选择论文模型。

## 指标边界

- `macro-object` 是八个物体 recall 的平均值；`micro-target` 是 1,445 个 target
  上的成功比例，二者不能互换。
- Epoch 35/40 evaluator 成功产生 BOP19 结果，但历史自动 ADD(-S) 查找返回
  `null`，下载资料中也没有对应 ADD(-S) 宏平均或逐物体文件。因此这里只记录
  “未生成/未核验”，不把 BOP `ad` 指标误写成 ADD(-S)@0.1d。
- C2 的主要因果比较原本是 `C2-B`。若 B 最终取消，只能报告 C2 相对官方基线
  未过筛选，不能声称已经测得质量/覆盖模块相对 PnP-only 适应的额外贡献。

## 最小归档

- 冻结代码：`24a1d7aa7159cfc8113f4cfe8c11bbd2ce3ba290`。
- 历史镜像：`sha256:04a68c9b5b54de9bd386e120d52dd9bf812c5d20ec9b24bf06e7829d4c983495`。
- Epoch 40 checkpoint：`E:\6D姿态估计\C2\model_epoch_040.pth`，SHA-256
  `44d69b5cf97829d0a982bba6c6644b4fe2f41a3c88908fecb6ebb0deb905b270`。
- checkpoint 已在本地 CPU 严格读取：`epoch=40`、`iteration=255919`、401 个模型
  tensor，并包含 optimizer/scheduler 状态。
- 完整日志与 Epoch 5–40 BOP JSON 保存在 `E:\6D姿态估计\C2`；机器可读结果和
  来源哈希见 `RESULTS.json`。
- 服务器另在 `weights/archives/EXP-20260805-008-C2-epoch40` 留存小型身份、
  resolved config、日志和 BOP 证据；旧 C1/C2 容器与 raw outputs 已于
  2026-08-11 按用户授权删除，数据集、官方权重、baseline 和镜像保留。
