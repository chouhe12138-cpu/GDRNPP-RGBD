# CPM moment 数值审计（2026-08-09）

## 范围与边界

本审计在 `lmo_pbr_train` 上固定抽取 8192 个样本，只运行 official checkpoint
的 backbone、geometry head 和 CPM moment 编码，不创建 optimizer、不更新参数，
也不保存逐实例 moment。它只决定训练前的固定数值缩放和低支持 Region 的工程
处理，不构成 CPM 性能或研究假设成立的证据。

- official checkpoint SHA256：
  `bafa869d4e6c00410517ecb1add59f234ed1642e47fabcf3aa6e0e8a1b498a8c`
- seed：`20260809`
- 数值类型：CUDA FP32（`n_eff` 单独以 FP64 计算）
- 原始机器产物：`output/cpm_head/audit_8192/`
- 状态：`COMPLETE / PASS`

## 审计结论

全部 524288 个 Region descriptor 有效，moment group 的 finite ratio 均为 1。
低 coverage 且 `n_eff < 2` 的 Region 为 36 个，占有效 Region 的
`6.866455078125e-05`；这些 Region 的 moment 未出现异常爆炸。因此第一版不增加
阈值裁剪、hard top-k、learned quality 或其他低支持处理，继续保留原始 soft
Region weighting，并以 coverage 向共享 MLP 提供相对质量信息。

五组 moment 的 raw P95 absolute value 如下：

| group | P95 absolute value |
|---|---:|
| `mu_X` | 0.053791501000523545 |
| `mu_U` | 0.8383049368858337 |
| `C_XX` | 0.00045910440967418244 |
| `C_UU` | 0.03372693955898269 |
| `C_XU` | 0.0008616717066615817 |

最大/最小比为 `1825.9570573080807`，超过预先固定的触发阈值 10。因此训练前
冻结确定性缩放：各组除以上表中对应常数；coverage 不缩放。该公式无可学习
参数，不根据 LM-O test 性能调整，也不作为论文创新点。

后续所有 CPM 主实验和正式消融必须复用同一组常数。若改变输入定义、数据域、
Region 数量或 moment 公式，应视作协议变化并重新做训练前审计，不能静默继承。
