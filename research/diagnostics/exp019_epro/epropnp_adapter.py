"""Isolated adapter for the public EPro-PnP-v2 6DoF solver.

GDRNPP and EPro-PnP both expose a top-level ``lib`` package, so EPro runs in
a persistent spawned process.  The supplied upstream snapshot also pins a
Pyro release that rejects PyTorch 2.x; a small inference-only compatibility
shim supplies the three distribution APIs used by EPro without editing or
vendoring the third-party source.
"""

import math
import multiprocessing as mp
import os
import sys
import types
from pathlib import Path

import numpy as np


def _matrix_to_quaternion_wxyz(R):
    m = np.asarray(R, dtype=np.float64)
    q = np.empty(4, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        q[:] = [
            0.25 * scale,
            (m[2, 1] - m[1, 2]) / scale,
            (m[0, 2] - m[2, 0]) / scale,
            (m[1, 0] - m[0, 1]) / scale,
        ]
    else:
        axis = int(np.argmax(np.diag(m)))
        if axis == 0:
            scale = np.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            q[:] = [(m[2, 1] - m[1, 2]) / scale, 0.25 * scale,
                    (m[0, 1] + m[1, 0]) / scale, (m[0, 2] + m[2, 0]) / scale]
        elif axis == 1:
            scale = np.sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            q[:] = [(m[0, 2] - m[2, 0]) / scale, (m[0, 1] + m[1, 0]) / scale,
                    0.25 * scale, (m[1, 2] + m[2, 1]) / scale]
        else:
            scale = np.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            q[:] = [(m[1, 0] - m[0, 1]) / scale, (m[0, 2] + m[2, 0]) / scale,
                    (m[1, 2] + m[2, 1]) / scale, 0.25 * scale]
    return q / np.linalg.norm(q)


def _quaternion_wxyz_to_matrix(q):
    w, x, y, z = np.asarray(q, dtype=np.float64).reshape(4) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _install_pyro_compat(torch):
    """Install only the Pyro distribution surface used by EPro-PnP-v2."""

    class MultivariateStudentT(torch.distributions.Distribution):
        arg_constraints = {}
        has_rsample = True

        def __init__(self, df, loc, scale_tril, validate_args=None):
            self.df = torch.as_tensor(df, dtype=loc.dtype, device=loc.device)
            self.loc = loc
            self.scale_tril = scale_tril
            super().__init__(loc.shape[:-1], loc.shape[-1:], validate_args=validate_args)

        def rsample(self, sample_shape=torch.Size()):
            shape = self._extended_shape(sample_shape)
            normal = torch.randn(shape, dtype=self.loc.dtype, device=self.loc.device)
            scaled = torch.matmul(self.scale_tril, normal.unsqueeze(-1)).squeeze(-1)
            gamma = torch.distributions.Gamma(self.df / 2, self.df / 2)
            factor = gamma.rsample(sample_shape + self.batch_shape).sqrt().unsqueeze(-1)
            return self.loc + scaled / factor

        def log_prob(self, value):
            delta = value - self.loc
            solved = torch.linalg.solve_triangular(
                self.scale_tril, delta.unsqueeze(-1), upper=False
            ).squeeze(-1)
            mahal = solved.square().sum(-1)
            dims = self.event_shape[0]
            half_log_det = self.scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1)
            normalizer = (
                torch.lgamma((self.df + dims) / 2)
                - torch.lgamma(self.df / 2)
                - 0.5 * dims * torch.log(self.df * self.df.new_tensor(math.pi))
                - half_log_det
            )
            return normalizer - 0.5 * (self.df + dims) * torch.log1p(mahal / self.df)

    pyro_module = types.ModuleType("pyro")
    distributions = types.ModuleType("pyro.distributions")
    util = types.ModuleType("pyro.distributions.util")
    distributions.MultivariateStudentT = MultivariateStudentT
    distributions.TorchDistribution = torch.distributions.Distribution
    distributions.constraints = torch.distributions.constraints
    util.broadcast_shape = torch.broadcast_shapes
    pyro_module.distributions = distributions
    sys.modules["pyro"] = pyro_module
    sys.modules["pyro.distributions"] = distributions
    sys.modules["pyro.distributions.util"] = util


def _worker_main(conn, root, cfg_dict, device):
    try:
        root = str(Path(root).resolve())
        sys.path.insert(0, root)
        import cv2
        import torch

        # Spawn re-imports the parent entry module.  That module may build the
        # GDRN adapter and preload this repository's unrelated top-level lib.
        # Clearing it is safe here because this process is dedicated to EPro.
        for module_name in list(sys.modules):
            if module_name == "lib" or module_name.startswith("lib."):
                del sys.modules[module_name]
        # The supplied EPro snapshot calls the removed torch.solve API once.
        # Preserve its old (solution, LU) indexing contract inside this worker.
        torch.solve = lambda b, A: (torch.linalg.solve(A, b), None)
        _install_pyro_compat(torch)
        from lib.ops.pnp.camera import PerspectiveCamera
        from lib.ops.pnp.cost_fun import AdaptiveHuberPnPCost
        from lib.ops.pnp.epropnp import EProPnP6DoF
        from lib.ops.pnp.levenberg_marquardt import LMSolver

        solver = EProPnP6DoF(
            mc_samples=cfg_dict["mc_samples"],
            num_iter=cfg_dict["num_iter"],
            solver=LMSolver(dof=6, num_iter=cfg_dict["lm_num_iter"]),
        ).to(device)
        solver.eval()
        conn.send({"kind": "ready"})
    except Exception as exc:
        conn.send({"kind": "fatal", "error": f"{type(exc).__name__}: {exc}"})
        conn.close()
        return

    while True:
        try:
            message = conn.recv()
        except EOFError:
            break
        if message["kind"] == "close":
            break
        if message["kind"] != "solve":
            conn.send({"kind": "error", "error": f"unknown message {message['kind']}"})
            continue
        try:
            x3d_np = np.asarray(message["x3d"], dtype=np.float64)
            x2d_np = np.asarray(message["x2d"], dtype=np.float64)
            K_np = np.asarray(message["K"], dtype=np.float64)
            n = int(x3d_np.shape[0])
            ok, rvec, tvec = cv2.solvePnP(
                x3d_np,
                x2d_np,
                K_np,
                np.zeros((4, 1), dtype=np.float64),
                flags=cv2.SOLVEPNP_EPNP,
            )
            if not ok:
                conn.send({"kind": "result", "success": False,
                           "message": "cv2.solvePnP(EPNP) failed", "num_points": n})
                continue
            R0, _ = cv2.Rodrigues(rvec)
            pose0 = np.concatenate([tvec.reshape(3), _matrix_to_quaternion_wxyz(R0)])
            x3d = torch.as_tensor(x3d_np, dtype=torch.float32, device=device)[None]
            x2d = torch.as_tensor(x2d_np, dtype=torch.float32, device=device)[None]
            K = torch.as_tensor(K_np, dtype=torch.float32, device=device)[None]
            pose_init = torch.as_tensor(pose0, dtype=torch.float32, device=device)[None]
            w2d = torch.full(
                (1, n, 2), cfg_dict["uniform_weight"], dtype=torch.float32, device=device
            )
            camera = PerspectiveCamera(cam_mats=K, z_min=cfg_dict["z_min"])
            cost_fun = AdaptiveHuberPnPCost(relative_delta=cfg_dict["huber_delta"])
            cost_fun.set_param(x2d, w2d)
            with torch.no_grad():
                pose_opt = solver(
                    x3d,
                    x2d,
                    w2d,
                    camera,
                    cost_fun,
                    pose_init=pose_init,
                    fast_mode=cfg_dict["fast_mode"],
                )[0][0].detach().float().cpu().numpy()
            if pose_opt.shape != (7,) or not np.isfinite(pose_opt).all():
                raise RuntimeError(f"invalid pose_opt: {pose_opt}")
            conn.send(
                {
                    "kind": "result",
                    "success": True,
                    "R": _quaternion_wxyz_to_matrix(pose_opt[3:]),
                    "t": pose_opt[:3].astype(np.float64),
                    "message": "",
                    "num_points": n,
                }
            )
        except Exception as exc:
            conn.send(
                {
                    "kind": "result",
                    "success": False,
                    "message": f"{type(exc).__name__}: {exc}",
                    "num_points": int(np.asarray(message.get("x3d", [])).shape[0]),
                }
            )
    conn.close()


class EProPnPWorker:
    """Persistent EPro-PnP worker isolated from GDRNPP's ``lib`` package."""

    def __init__(self, cfg, device="cuda:0", root=None):
        from .types import PoseResult

        self._PoseResult = PoseResult
        if cfg.weight_mode != "uniform":
            raise ValueError("EXP019 requires uniform w2d")
        root = root or os.environ.get("EPROPNP_ROOT") or os.environ.get("EPROPMP_ROOT")
        if not root:
            raise RuntimeError(
                "Set EPROPNP_ROOT to EPro-PnP-v2-main/EPro-PnP-6DoF_v2"
            )
        root = Path(root).resolve()
        required = root / "lib" / "ops" / "pnp" / "epropnp.py"
        if not required.is_file():
            raise RuntimeError(f"Invalid EPROPNP_ROOT; missing {required}")
        cfg_dict = {
            "mc_samples": int(cfg.epro_mc_samples),
            "num_iter": int(cfg.epro_num_iter),
            "lm_num_iter": int(cfg.epro_lm_num_iter),
            "uniform_weight": float(cfg.uniform_weight),
            "z_min": float(cfg.epro_z_min_m),
            "huber_delta": float(cfg.epro_relative_huber_delta),
            "fast_mode": bool(cfg.epro_fast_mode),
        }
        context = mp.get_context("spawn")
        self._parent, child = context.Pipe()
        self._proc = context.Process(
            target=_worker_main, args=(child, str(root), cfg_dict, device), daemon=True
        )
        self._proc.start()
        hello = self._parent.recv()
        if hello["kind"] != "ready":
            self.close()
            raise RuntimeError(f"EPro worker failed to start: {hello.get('error', hello)}")

    def solve(self, corr):
        from .types import failed_pose

        if corr.n < 6:
            return failed_pose("epropnp_v2", "fewer_than_six_points", corr.n)
        self._parent.send(
            {"kind": "solve", "x3d": corr.x3d_m, "x2d": corr.x2d_px, "K": corr.K}
        )
        out = self._parent.recv()
        if out["kind"] != "result" or not out.get("success", False):
            return failed_pose(
                "epropnp_v2", out.get("message", out.get("error", str(out))), corr.n
            )
        return self._PoseResult(
            R=np.asarray(out["R"], dtype=np.float64).reshape(3, 3),
            t=np.asarray(out["t"], dtype=np.float64).reshape(3),
            success=True,
            solver="epropnp_v2",
            num_points=int(out["num_points"]),
        )

    def close(self):
        parent = getattr(self, "_parent", None)
        process = getattr(self, "_proc", None)
        if parent is not None and process is not None:
            try:
                if process.is_alive():
                    parent.send({"kind": "close"})
            except (BrokenPipeError, EOFError):
                pass
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
