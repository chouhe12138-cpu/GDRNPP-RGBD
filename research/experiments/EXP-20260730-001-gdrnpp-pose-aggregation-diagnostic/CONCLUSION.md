# Conclusion

Status: `FAIL`

The experiment does not support the strong claim that GDRNPP Patch-PnP
systematically fails to use pose information that a simple explicit solver can
recover.  Unfiltered RANSAC-EPnP produces a small overall gain, but the effect
is not stable across objects and fails under extreme occlusion.

The useful result is more specific:

1. Dense GDRNPP correspondences do retain recoverable geometric information:
   RANSAC-EPnP raises BOP AR from 69.021% to 69.594% and ADD(-S) recall from
   50.242% to 53.080%.
2. The gain is a trade-off rather than a general replacement: VSD/MSSD improve,
   MSPD declines, and only 5/8 objects are non-negative.
3. Existing mask and region confidence is not a valid reliability estimate.
   Selecting its top half performs worse than using all correspondences.
4. Very low visibility creates catastrophic but numerically valid PnP
   translations.  Any future geometric aggregation must model degeneracy and
   pose plausibility rather than merely select high-confidence points.

Under the frozen graduation-priority protocol, Stage 1 stops here and does not
authorize a new head.  If this route is selected later, the defensible research
problem is “learned correspondence reliability plus degeneracy-aware geometric
aggregation,” not “replace Patch-PnP with ordinary RANSAC.”  The archived RGB-D
route remains available for a separate Stage 2 decision.
