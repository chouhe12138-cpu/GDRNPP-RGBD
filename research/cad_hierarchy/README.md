# CAD hierarchy 公共接口

运行时 artifact contract 位于 `core.gdrn_modeling.cad.hierarchy`；NumPy 几何与
诊断位于本目录。公共模块不导入 EXP021/022、模型头或数据集注册代码。

```python
from core.gdrn_modeling.cad.hierarchy import load_cad_hierarchy
from research.cad_hierarchy.geometry import traverse_hierarchy, oracle_decode
from research.cad_hierarchy.diagnostics import hierarchy_sanity, surface_representation

h = load_cad_hierarchy(".local/dataset_cache/exp022/consistent_v3.npz",
                       expected_object_ids=(1, 5, 6, 8, 9, 10, 11, 12), dataset_key="lmo")
levels = h.numpy_levels()  # CPU float64；原始 Torch 数据通过 h.level(3) 访问
report = hierarchy_sanity(levels, h.object_ids.tolist())
# 单物体 points: [P,3]，物体坐标系，单位米；输出是每层的全局节点 ID。
# paths = traverse_hierarchy(points, [levels[d]["anchors"][0] for d in (1, 2, 3)], 8)
# leaf = paths[:, -1]
# xyz = oracle_decode(points, levels[3]["anchors"][0, leaf], levels[3]["radii"][0, leaf])
```

层级从 1 开始，支持完整固定分支树；T0 不存入 artifact。`h.depth`、
`h.branch_factor`、`h.level_counts` 从数组推导，若 NPZ 已声明这些 metadata 则交叉核对。
loader 保留 dtype、已有 metadata 与可选 `source_leaf_indices`，不改写文件，也不要求
旧文件新增 schema。新 schema/builder 的设计与新版 artifact 生成留给后续实验。

`dataset_key` 在显式指定时必须一致；缺失 metadata 默认拒绝，历史适配器可显式使用
`allow_missing_dataset=True`。该选项不允许实际存储的数据集名称不匹配。
Generic loader 只检查结构与数值，不限制 mode/version 或最多两条 symmetry。
旧 PCC loader 仍要求 reused/1 或 independent/2、四层 8⁴、source_leaf_indices、S≤2；
新 consistent/3 不能用于旧 PCC 训练。

`geometry` 提供 sample_surface、farthest_point_sampling、nearest_anchor、
normalize_vectors、traverse_hierarchy、parent_of/children_of，以及 residual encode、
clip、decode、oracle_decode。采样函数按面积取点后追加全部 mesh 顶点，返回 float32；
FPS/nearest 的原始算术、随机顺序与 SciPy 实现保留。普通 decode 不隐式裁剪；
oracle_decode 才执行单位球投影。旧网络的 tanh、Torch 路由及梯度路径未迁移。

`diagnostics` 提供 hierarchy_shape_report、parent_child_coverage、hierarchy_sanity、
surface_representation 与 residual_stats。sanity 检查所有相邻层的父球覆盖，
失败关系列在 failed_relations。surface_representation 可选择任意实际深度，
单独报告采样覆盖与未命中节点，不把未命中当作空节点证明，也不产生机制 gate。
历史 global_coverage 字段指全局最近锚点的球，不是所有球的并集。
父球覆盖、有限采样覆盖都不能单独证明整个连续表面的覆盖。

历史 oracle CLI、fixed PnP 和 DatasetContext 接线仍在 EXP022；它们不是公共接口。
T3+residual 是后续方法候选：surface oracle 支持其几何上限，但未证明网络可学习性。
本轮不分配 EXP025，不实现 attention、模型或训练 preflight。

测试：激活 Conda `pytorch22` 后运行 `python -m pytest -q research/cad_hierarchy/tests`。
合成测试无需数据集；三个真实 artifact 回归缺文件时 skip。历史 Torch 路由对照与
采样 reference 测试使用已安装的项目依赖；生产公共模块没有这些历史依赖。
