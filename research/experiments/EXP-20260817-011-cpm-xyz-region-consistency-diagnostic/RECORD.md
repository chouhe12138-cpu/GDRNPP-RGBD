# EXP011 — CPM XYZ–Region Consistency Diagnostic

- 状态：`COMPLETE / MISMATCH_IMPORTANT`
- run：`RUN-20260817-023144-full-s20260817-a02`
- checkpoint：EXP009 E40，固定权重
- 范围：1,445 targets × 10 conditions，QC PASS

Pred Region 下 GT-XYZ effect：BOP `-0.289476`、ADD(-S)
`-0.279648`；GT Region 下分别为 `-0.138113`、`-0.102498`。
interaction 为 BOP `+0.151363`、ADD `+0.177150`，ADD interaction 8/8
物体为正。结论支持 XYZ–Region 不一致是污染因素，但不表示 GT Region 本身是
可部署改进，也未证明它是唯一根因。
