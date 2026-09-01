# EXP013F 实验记录

## 状态

- 本地门禁:v2(见 LOCAL_GATE.json);服务器:未启动(lab1)。
- 分支 `exp013f-glm-pose-l`,bundle `GDRNPP-RGBD-<sha>.bundle` 位于
  `E:\6D姿态估计\EXP-013\实验F\`。

## 2026-08-29 F v1 失败记录(保留证据)

- v1 实现把 `INPUT.HEAD_DEPTH` 深度统计只加进离线数据类
  `GDRN_DatasetFromList`(data_loader.py);而 lmo_pbr 配置
  `XYZ_ONLINE=True`,`build_gdrn_train_loader` 走的是
  `GDRN_Online_DatasetFromList`(data_loader_online.py),该类没有
  HEAD_DEPTH 支持 → 训练 smoke 崩溃 `KeyError: 'roi_depth_stats'`。
  分支当时回退,WIP 保存在 stash,实验记录未建。
- v1 另有两处潜在缺陷(本版一并修复):
  1. stats 在**整图深度**上计算(与函数文档"ROI 裁剪深度"不符;测试时
     会变成场景级统计且同图所有实例相同);
  2. `[None, …]` 形状在 batch 后变 `[B,1,4]`,会被头的形状断言拒绝。

## 2026-08-29 v2 修复与门禁

- `compute_roi_depth_stats` 移入独立模块
  `core/gdrn_modeling/datasets/roi_depth_stats.py`(避免
  data_loader ↔ data_loader_online 循环导入);函数入口把 3D(HxWx1)
  输入归一化为 HxW(cv2 resize/warpAffine 可能丢单通道维度)。
- 三个调用点统一:在线训练类 `read_data_train`、离线训练路径、
  离线测试收集循环,统计一律在 **ROI 裁剪深度**(与 roi_depth 同几何,
  INTER_NEAREST,bbox_center/scale/input_res)上计算;形状 [4]。
- 在线类 `read_data_test` 同步补齐 HEAD_DEPTH(防同类路径分叉陷阱);
  删除在线类中 `if self.norm_depth:` 死代码块(属性不存在、
  normalSpeed 未导入,扩展条件后首次暴露)。
- 深度增强/整图 resize 块条件扩为 `with_depth or head_depth`
  (head_depth 时深度与图像同几何);`HEAD_DEPTH`+`BP_DEPTH` 互斥。
- 门禁结果(详见 LOCAL_GATE.json):pytest 28 项全过;preflight F
  CPU(严格往返)与 CUDA 均 PASS;真实数据 smoke
  `LOCAL-20260829-f-online-depth-smoke` 1 epoch 2048 iters batch 4
  workers 2,loss 有限(final 0.678,均值 1.903 递减),渲染器 0 次构造,
  `model_epoch_001.pth` 已保存。

## 2026-08-29 独立子代理审计(v2 之后)

- 结论 **PASS,0 P1**;12 项清单(M1 继承、M2 结构、M3 注入点、输出契约、
  参数预算、渲染器保证、协议常数、预注册 gate、深度管线 a–f、键安全、
  测试运行、家族独立性)全部 PASS;头参数实测 929,175。
- P2-1(已修):`batch_data_train_online`(监督开启的在线路径)补齐
  `roi_depth_stats` 堆叠——未来若组合 XYZ_ONLINE+监督开+HEAD_DEPTH,
  训练统计不再被静默零填充(当前 F 协议下该路径不可达,属防御性修复)。
- P2-2(不修,记录在案):A 基类遗留死方法 `_encode_main`/`_encode_geometry`
  引用已删除模块;它们在 A/B/C 归档代码中即已不可达,改共享基类文件
  超出 F 范围且无行为收益。
- P2-3(已修):F EXPERIMENT.json 基线里 A 的 ar_teS 误写 0.788247,
  更正为档案权威值 0.797693(A RECORD.md:17);四条预注册 gate 数字
  未动(0.8028/0.5107/0.4930/0.6838 为用户批准的契约)。
- 审计后复跑:F 单测 7 项 + 全套 28 项、preflight F CPU,全部 PASS。

## 2026-08-30 formal 训练进度(E5–E15 评估完成,训练至 E20)

formal run `RUN-20260829-103858-formal-s42-a01`(lab1,seed 42,commit e924b96)。

| Epoch | BOP AR | ADD(-S) macro | AR_reS | AR_teS |
|---:|---:|---:|---:|---:|
| 5 | 0.575645 | 0.329412 | 0.285582 | 0.704037 |
| 10 | 0.638674 | 0.452595 | 0.408766 | 0.765398 |
| 15 | 0.648992 | 0.458824 | 0.426990 | 0.771857 |
| 20 | 0.649107 | 0.458824 | 0.437601 | 0.765167 |
| 25 | 0.656858 | 0.467128 | 0.437601 | 0.788466 |
| 30 | 0.669123 | 0.496194 | 0.496886 | 0.781084 |
| 35 | 0.672293 | 0.501730 | 0.486044 | 0.784775 |
| 40（正式 gate） | **0.684129** | **0.504498** | **0.515802** | **0.799308** |

- 全程单调上升,无任何 epoch 回落;E35 与 A 同期差距 `0.007`,逐段收窄。
- 四条 gate 的 E35 位置:BOP `0.6723`(差 0.0115)、ADD macro `0.5017`(差 0.009)、
  reS `0.4860`(差 0.007)、teS `0.7848`(差 0.018);teS 仍是最难项,
  M3 的 teS 增益至今未显现。

## 2026-08-31 训练进度更新

- 训练至 E38(95%,lr 退火至 6e-5),E40 评估待落地。

## 2026-08-31 E40 最终结果与 gate 判决

- `MANAGED_RUN_FINISH status=PASS`(2026-08-31T14:09Z),40/40 epoch 完成。
- 固定 E40:BOP AR `0.6841291810841984`、ADD(-S) macro `0.5044982698961937`、
  reS `0.5158016147635525`、teS `0.7993079584775087`。
- 四条 gate 判决:
  - reS ≥ 0.4930:`0.5158` **PASS**;
  - BOP ≥ 0.6838:`0.6841` **PASS**(+0.0003);
  - teS ≥ 0.8028:`0.7993` **FAIL**(差 0.0035);
  - ADD ≥ 0.5107:`0.5045` **FAIL**(差 0.0062)。
  - 四条全过才 PASS → **`SCREEN_FAIL`**;teS/ADD 差距均在单评估噪声带(±0.01)
    内,属边缘失败。
- 正面事实:reS `0.5158` 为家族第二高旋转(C `0.5250` 之后,A/B `0.4980`),
  注意力池化的旋转读出能力得到验证;BOP AR 高于 A(`0.6840`)与 B(`0.6837`);
  teS 高于 A(`0.7977`)、低于 B(`0.8012`)——M3 深度统计未表现出超越 B 的
  平移优势,也不足以独立跨过 0.8028 门槛。
- 逐物体 E40 ADD(-S):ape 0.486、can 0.804、cat 0.462、driller 0.775、
  duck 0.122、eggbox 0.378、glue 0.750、holepuncher 0.275。
