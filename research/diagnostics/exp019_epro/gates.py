import numpy as np


def _rankdata_average(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and values[order[end]] == values[order[index]]:
            end += 1
        ranks[order[index:end]] = 0.5 * (index + end - 1) + 1.0
        index = end
    return ranks


def spearman(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("same-shape 1D inputs required")
    rx, ry = _rankdata_average(x), _rankdata_average(y)
    if rx.std() == 0 or ry.std() == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def evaluate_gates(cfg, summary):
    keys = [f"{alpha:.2f}" for alpha in cfg.alphas]
    epro_add = [float(summary["epro"][key]["add"]) for key in keys]
    epro_bop = [float(summary["epro"][key]["bop"]) for key in keys]
    ransac_add = [float(summary["ransac"][key]["add"]) for key in keys]
    ransac_bop = [float(summary["ransac"][key]["bop"]) for key in keys]
    delta_epro_add = epro_add[-1] - epro_add[0]
    delta_epro_bop = epro_bop[-1] - epro_bop[0]
    delta_ransac_add = ransac_add[-1] - ransac_add[0]
    delta_ransac_bop = ransac_bop[-1] - ransac_bop[0]
    recovery_add = delta_epro_add / delta_ransac_add if delta_ransac_add > 1e-12 else 0.0
    recovery_bop = delta_epro_bop / delta_ransac_bop if delta_ransac_bop > 1e-12 else 0.0
    rho_add = spearman(cfg.alphas, epro_add)
    rho_bop = spearman(cfg.alphas, epro_bop)
    gate_a = epro_add[-1] >= cfg.gate_gt_xyz_add and epro_bop[-1] >= cfg.gate_gt_xyz_bop
    gate_b = (
        rho_add >= cfg.gate_spearman
        and rho_bop >= cfg.gate_spearman
        and recovery_add >= cfg.gate_recovery_ratio
        and recovery_bop >= cfg.gate_recovery_ratio
    )
    return {
        "gate_A_gt_xyz_wiring": bool(gate_a),
        "gate_B_geometry_responsiveness": bool(gate_b),
        "gt_xyz": {"add": epro_add[-1], "bop": epro_bop[-1]},
        "spearman": {"add": rho_add, "bop": rho_bop},
        "delta_epro": {"add": delta_epro_add, "bop": delta_epro_bop},
        "delta_ransac": {"add": delta_ransac_add, "bop": delta_ransac_bop},
        "recovery_ratio": {"add": recovery_add, "bop": recovery_bop},
        "decision": (
            "PASS_GEOMETRY_UTILIZATION_DIAGNOSTIC"
            if gate_a and gate_b
            else "STOP_DO_NOT_TRAIN"
        ),
    }

