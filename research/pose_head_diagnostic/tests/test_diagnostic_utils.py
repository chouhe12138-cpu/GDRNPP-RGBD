from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from core.gdrn_modeling.models.heads.conv_pnp_net import ConvPnPNet
from research.pose_aggregation.metrics import pose_metrics
from research.pose_head_diagnostic.analyze_existing import analyze
from research.pose_head_diagnostic.diagnostic_utils import (
    CONDITIONS,
    apply_intervention,
    matched_spatial_masks,
    response_metrics,
    tensor_state_sha256,
)
from research.pose_head_diagnostic.run_statistical_diagnostic import POSE_METRICS
from research.pose_head_diagnostic.revalidate_full import assess_full_quality_control
from research.pose_head_diagnostic.statistical_utils import (
    aggregate_scalar_records,
    summarize_values,
)


class DiagnosticUtilsTest(unittest.TestCase):
    def setUp(self):
        self.xyz = np.arange(36, dtype=np.float32).reshape(3, 4, 3)
        self.gt = self.xyz + 100
        self.roi = np.arange(24, dtype=np.float32).reshape(3, 4, 2)
        self.region = np.arange(48, dtype=np.float32).reshape(3, 4, 4)
        self.support = np.zeros((3, 4), dtype=bool)
        self.support[1:, 1:3] = True

    def test_all_interventions_preserve_shapes_and_outside_support(self):
        for condition in CONDITIONS:
            xyz, roi, region = apply_intervention(
                self.xyz, self.gt, self.roi, self.region, self.support, condition, 7
            )
            self.assertEqual(xyz.shape, self.xyz.shape)
            self.assertEqual(roi.shape, self.roi.shape)
            self.assertEqual(region.shape, self.region.shape)
            np.testing.assert_array_equal(xyz[~self.support], self.xyz[~self.support])
            np.testing.assert_array_equal(roi[~self.support], self.roi[~self.support])
            np.testing.assert_array_equal(region[~self.support], self.region[~self.support])
        xyz, _, _ = apply_intervention(
            self.xyz, self.gt, self.roi, self.region, self.support, "gt_x", 7
        )
        np.testing.assert_array_equal(xyz[..., 0][self.support], self.gt[..., 0][self.support])
        np.testing.assert_array_equal(xyz[..., 1:][self.support], self.xyz[..., 1:][self.support])

    def test_matched_spatial_masks(self):
        xyz = np.zeros((9, 9, 3), dtype=np.float32)
        gt = np.arange(9 * 9 * 3, dtype=np.float32).reshape(9, 9, 3)
        support = np.zeros((9, 9), dtype=bool)
        support[1:8, 1:8] = True
        masks = matched_spatial_masks(
            xyz, gt, support, seed=11, boundary_width=1
        )
        self.assertEqual(
            int(masks["gt_xyz_boundary"].sum()),
            int(masks["gt_xyz_interior_matched"].sum()),
        )
        self.assertGreater(int(masks["gt_xyz_boundary"].sum()), 0)
        self.assertEqual(
            int(masks["gt_xyz_high_error"].sum()),
            int(masks["gt_xyz_random_matched"].sum()),
        )
        for mask in masks.values():
            self.assertTrue(np.all(mask <= support))

    def test_permutations_are_deterministic_and_preserve_value_sets(self):
        first = apply_intervention(
            self.xyz, self.gt, self.roi, self.region, self.support, "permute_xyz", 19
        )[0]
        second = apply_intervention(
            self.xyz, self.gt, self.roi, self.region, self.support, "permute_xyz", 19
        )[0]
        np.testing.assert_array_equal(first, second)
        original_values = self.xyz[self.support].reshape(-1)
        changed_values = first[self.support].reshape(-1)
        np.testing.assert_array_equal(np.sort(original_values), np.sort(changed_values))

    def test_statistical_summary(self):
        summary = summarize_values(
            [1.0, 2.0, 3.0, float("nan")], seed=3, bootstrap_samples=50
        )
        self.assertEqual(summary["count"], 4)
        self.assertEqual(summary["finite_count"], 3)
        self.assertEqual(summary["median"], 2.0)
        rows = aggregate_scalar_records(
            [
                {"condition": "a", "metric_a": 1.0},
                {"condition": "a", "metric_a": 3.0},
            ],
            ("condition",),
            ("metric_a",),
            seed=3,
            bootstrap_samples=20,
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["mean"], 2.0)

    def test_pose_metric_schema_matches_aggregated_record(self):
        direct_metrics = pose_metrics(
            np.eye(3),
            np.zeros(3),
            np.eye(3),
            np.zeros(3),
            np.zeros((4, 3)),
            1.0,
            1,
        )
        derived_metrics = {
            "rotation_delta_deg",
            "translation_delta_mm",
            "rotation_error_improvement_deg",
            "translation_error_improvement_mm",
            "add_s_improvement",
        }
        self.assertEqual(set(POSE_METRICS), set(direct_metrics) | derived_metrics)

    def test_full_quality_control_calibration(self):
        qc = {
            "processed_targets": 1445,
            "condition_counts": {condition: 1445 for condition in CONDITIONS},
            "empty_support_targets": 7,
            "nonfinite_scalar_count": 98,
            "max_baseline_raw_rotation_abs_error": 4.5e-5,
            "max_baseline_raw_translation_abs_error": 1e-5,
            "max_baseline_rotation_abs_error": 2.5e-4,
            "max_baseline_translation_abs_error": 2e-5,
            "baseline_add_s_0.1d_recall": 730 / 1445,
            "state_unchanged": True,
        }
        calibrated = assess_full_quality_control(qc, 0.6903990772779699)
        self.assertTrue(calibrated["passed"])
        self.assertEqual(calibrated["expected_nonfinite_scalar_count"], 98)
        self.assertEqual(calibrated["unexpected_nonfinite_scalar_count"], 0)
        amp_like = {
            **qc,
            "max_baseline_raw_rotation_abs_error": 9.765625e-4,
        }
        self.assertFalse(
            assess_full_quality_control(amp_like, 0.6903990772779699)["passed"]
        )

    def test_response_metrics(self):
        ref = torch.tensor([1.0, 2.0, 3.0])
        metrics = response_metrics(ref, ref.clone())
        self.assertEqual(metrics["relative_l2"], 0.0)
        self.assertAlmostEqual(metrics["cosine_distance"], 0.0)
        self.assertEqual(metrics["finite_count"], 3)

    def test_synthetic_head_hooks_and_state_unchanged(self):
        model = ConvPnPNet(
            nIn=69, num_regions=64, featdim=128, num_stride2_layers=3,
            final_spatial_size=(8, 8), norm="GN"
        ).eval()
        seen = {}
        call_index = {"value": 0}

        def activation_hook(_module, _inputs, output):
            stage = (2, 5, 8)[call_index["value"]]
            seen[stage] = output.detach()
            call_index["value"] += 1

        handles = [
            model.features[2].register_forward_hook(activation_hook)
        ]
        before = tensor_state_sha256(model.state_dict())
        with torch.no_grad():
            rot, trans = model(
                torch.rand(1, 5, 64, 64),
                region=torch.rand(1, 64, 64, 64),
                extents=torch.ones(1, 3),
            )
        for handle in handles:
            handle.remove()
        self.assertEqual(tuple(seen[2].shape), (1, 128, 32, 32))
        self.assertEqual(tuple(seen[5].shape), (1, 128, 16, 16))
        self.assertEqual(tuple(seen[8].shape), (1, 128, 8, 8))
        self.assertEqual(tuple(rot.shape), (1, 6))
        self.assertEqual(tuple(trans.shape), (1, 3))
        self.assertEqual(before, tensor_state_sha256(model.state_dict()))


class AnalyzerFixtureTest(unittest.TestCase):
    def test_analyzer_fixture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            headers = [
                "method", "scene_id", "im_id", "instance_id", "obj_id", "obj_name",
                "visibility", "rotation_error_deg", "translation_error_mm", "add_s_0.1d"
            ]
            with (root / "per_instance.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=headers)
                writer.writeheader()
                for index in range(1445):
                    for alpha_index, token in enumerate(("a000", "a025", "a050", "a075", "a100")):
                        writer.writerow({
                            "method": f"patch_{token}", "scene_id": 1, "im_id": index,
                            "instance_id": 0, "obj_id": 1, "obj_name": "ape",
                            "visibility": 0.8, "rotation_error_deg": 5 - alpha_index,
                            "translation_error_mm": 10 - alpha_index, "add_s_0.1d": 1,
                        })
            dense_headers = ["alpha_token"]
            with (root / "dense_interpolation.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=dense_headers)
                writer.writeheader()
                writer.writerows({"alpha_token": token} for token in ("a000", "a100"))
            aggregate_headers = [
                "method", "obj_id", "obj_name", "visibility_bin", "instances",
                "add_s_0.1d_recall", "mean_rotation_error_deg", "mean_translation_error_mm"
            ]
            for name in ("per_object.csv", "visibility_bins.csv"):
                with (root / name).open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=aggregate_headers)
                    writer.writeheader()
                    for method, value in (("patch_a000", 0.5), ("patch_a100", 0.6)):
                        writer.writerow({
                            "method": method, "obj_id": 1, "obj_name": "ape",
                            "visibility_bin": "ge_0.5", "instances": 1445,
                            "add_s_0.1d_recall": value, "mean_rotation_error_deg": 5,
                            "mean_translation_error_mm": 10,
                        })
            output = root / "out"
            summary = analyze(root, output)
            self.assertEqual(summary["instances"], 1445)
            self.assertEqual(summary["endpoint"]["rotation_improved_count"], 1445)
            self.assertFalse((output / "existing_failure_cases.csv").exists())


if __name__ == "__main__":
    unittest.main()
