# Conclusion — Frozen Patch-PnP Underutilizes Improved XYZ

Decision: `PATCH_PNP_UNDERUTILIZATION`

Progressively replacing predicted XYZ with GT XYZ on a fixed visible support
raises RANSAC ADD(-S) from 53.841% to 99.377%, but lowers the official direct
Patch-PnP result from 50.242% to 49.550%. RANSAC improves on all eight objects;
Patch-PnP is non-negative on five.

The experiment isolates utilization because mask, region logits, 2D
coordinates, support, bbox, weights, and all other network outputs remain
fixed. Therefore the main causal conclusion is not that direct pose regression
is inherently unsuitable. It is that this frozen Patch-PnP mapping has learned
a representation-specific coupling and does not respond appropriately when
its XYZ input becomes geometrically better.

RANSAC is not the proposed final method. It is a measuring instrument showing
that useful pose information exists in the corrected correspondences. The
next model experiment should retain direct `R,t` and add one lightweight,
region-balanced quality/coverage mechanism before Patch-PnP. It must be
trained; frozen official weights cannot evaluate a newly introduced
aggregation rule.
