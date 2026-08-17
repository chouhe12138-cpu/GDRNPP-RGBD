from __future__ import annotations

import csv
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from core.gdrn_modeling.models.heads.conv_pnp_net import ConvPnPNet
from core.gdrn_modeling.models.heads.cpm_pnp_net import CorrespondenceAwareMomentPnPNet
from research.pose_aggregation.metrics import pose_metrics
from research.pose_head_diagnostic.analyze_existing import analyze
from research.pose_head_diagnostic.diagnostic_utils import (
    CONDITIONS,
    apply_intervention,
    matched_spatial_masks,
    response_metrics,
    tensor_state_sha256,
)
from research.pose_head_diagnostic.run_statistical_diagnostic import (
    POSE_METRICS,
    build_gt_region_posterior,
    classify_factorial_result,
    factorial_summary,
    region_consistency_statistics,
)
from research.pose_head_diagnostic.run_information_flow import (
    CPM_EXTRA_CONDITIONS,
    CPM_XYZ_REGION_2X2_CONDITIONS,
    CPM_XYZ_REGION_ALPHA_CONDITIONS,
    apply_cpm_diagnostic_intervention,
    apply_cpm_moment_intervention,
    apply_cpm_xyz_region_intervention,
    cpm_xyz_region_condition,
    conditions_for_model,
)
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

    def test_cpm_specific_conditions_are_explicit_and_non_learned(self):
        cpm = CorrespondenceAwareMomentPnPNet()
        cpm_conditions = conditions_for_model(SimpleNamespace(pnp_net=cpm))
        self.assertEqual(cpm_conditions[: len(CONDITIONS)], CONDITIONS)
        self.assertEqual(cpm_conditions[len(CONDITIONS) :], CPM_EXTRA_CONDITIONS)
        self.assertEqual(
            conditions_for_model(SimpleNamespace(pnp_net=object())), CONDITIONS
        )

        xyz, roi, region, moment_condition = apply_cpm_diagnostic_intervention(
            self.xyz,
            self.gt,
            self.roi,
            self.region,
            self.support,
            "gt_xyz_alpha_025",
            7,
        )
        np.testing.assert_allclose(
            xyz[self.support],
            0.75 * self.xyz[self.support] + 0.25 * self.gt[self.support],
        )
        np.testing.assert_array_equal(xyz[~self.support], self.xyz[~self.support])
        np.testing.assert_array_equal(roi, self.roi)
        np.testing.assert_array_equal(region, self.region)
        self.assertIsNone(moment_condition)

        raw = torch.arange(2 * 64 * 21, dtype=torch.float32).reshape(2, 64, 21)
        scaled = cpm._apply_moment_scales(raw)
        coverage_raw, coverage_scaled = apply_cpm_moment_intervention(
            cpm, raw, scaled, "coverage_only"
        )
        torch.testing.assert_close(coverage_raw[..., 0], raw[..., 0])
        self.assertEqual(int(torch.count_nonzero(coverage_raw[..., 1:])), 0)
        torch.testing.assert_close(
            coverage_scaled, cpm._apply_moment_scales(coverage_raw)
        )

        no_cross_raw, _ = apply_cpm_moment_intervention(
            cpm, raw, scaled, "cxu_null"
        )
        torch.testing.assert_close(no_cross_raw[..., :15], raw[..., :15])
        self.assertEqual(int(torch.count_nonzero(no_cross_raw[..., 15:21])), 0)

    def test_cpm_xyz_region_factorial_changes_only_requested_sources(self):
        cpm = CorrespondenceAwareMomentPnPNet()
        model = SimpleNamespace(pnp_net=cpm)
        self.assertEqual(
            conditions_for_model(model, "cpm_xyz_region_2x2"),
            CPM_XYZ_REGION_2X2_CONDITIONS,
        )
        self.assertEqual(
            conditions_for_model(model, "cpm_xyz_region_alpha_sweep"),
            CPM_XYZ_REGION_ALPHA_CONDITIONS,
        )
        self.assertEqual(cpm_xyz_region_condition("gt_xyz_gt_region"), (1.0, "gt"))
        self.assertEqual(
            cpm_xyz_region_condition("xyz_alpha_050_pred_region"), (0.5, "pred")
        )

        pred_region = np.full((3, 4, 4), 0.25, dtype=np.float32)
        gt_region = np.zeros_like(pred_region)
        gt_region[..., 2] = 1.0
        for condition in CPM_XYZ_REGION_2X2_CONDITIONS:
            xyz, roi, region = apply_cpm_xyz_region_intervention(
                self.xyz,
                self.gt,
                self.roi,
                pred_region,
                gt_region,
                self.support,
                condition,
            )
            alpha, region_source = cpm_xyz_region_condition(condition)
            expected_xyz = (
                (1.0 - alpha) * self.xyz[self.support]
                + alpha * self.gt[self.support]
            )
            np.testing.assert_allclose(xyz[self.support], expected_xyz)
            np.testing.assert_array_equal(xyz[~self.support], self.xyz[~self.support])
            np.testing.assert_array_equal(roi, self.roi)
            np.testing.assert_array_equal(
                region[self.support],
                (gt_region if region_source == "gt" else pred_region)[self.support],
            )
            np.testing.assert_array_equal(
                region[~self.support], pred_region[~self.support]
            )

    def test_gt_region_reuses_training_definition_and_reports_agreement(self):
        gt_xyz_m = np.zeros((2, 3, 3), dtype=np.float32)
        support = np.zeros((2, 3), dtype=bool)
        support[0, :2] = True
        gt_xyz_m[0, 0] = [0.1, 0.0, 0.0]
        gt_xyz_m[0, 1] = [0.0, 0.2, 0.0]
        fps = np.asarray(
            [[0.1, 0.0, 0.0], [0.0, 0.2, 0.0], [1.0, 1.0, 1.0]],
            dtype=np.float32,
        )
        posterior, labels = build_gt_region_posterior(
            gt_xyz_m, fps, support, num_regions=3
        )
        np.testing.assert_array_equal(labels[support], [1, 2])
        np.testing.assert_array_equal(posterior[support], [[1, 0, 0], [0, 1, 0]])
        self.assertEqual(int(np.count_nonzero(posterior[~support])), 0)
        stats = region_consistency_statistics(posterior, labels, support)
        self.assertEqual(stats["support_points"], 2)
        self.assertEqual(stats["argmax_matches"], 2)
        self.assertEqual(stats["gt_probability_sum"], 2.0)
        self.assertEqual(stats["posterior_sum_max_error"], 0.0)

    def test_factorial_summary_and_preregistered_decision(self):
        names = {
            "pred_xyz_pred_region": 0.50,
            "gt_xyz_pred_region": 0.40,
            "pred_xyz_gt_region": 0.52,
            "gt_xyz_gt_region": 0.51,
        }
        add_summary = {
            name: {
                "macro_object": value,
                "micro_target": value,
                "per_object": {f"obj{i}": value for i in range(8)},
            }
            for name, value in names.items()
        }
        bop_scores = {
            "pred_xyz_pred_region": 0.60,
            "gt_xyz_pred_region": 0.50,
            "pred_xyz_gt_region": 0.61,
            "gt_xyz_gt_region": 0.60,
        }
        summary = factorial_summary(add_summary, bop_scores)
        self.assertAlmostEqual(
            summary["add_s_0.1d_macro_object"]["interaction"], 0.09
        )
        self.assertEqual(summary["positive_per_object_add_s_interactions"], 8)
        self.assertEqual(classify_factorial_result(summary), "MISMATCH_IMPORTANT")

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
