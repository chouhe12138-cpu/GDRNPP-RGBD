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
  workers 2,loss 有限,渲染器 0 次构造。
