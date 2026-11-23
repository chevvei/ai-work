import contextlib
import csv
import importlib.util
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "inference-handbook" / "assets" / "scenario_lab.py"
SPEC = importlib.util.spec_from_file_location("scenario_lab", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
lab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lab)


class ScenarioLabTests(unittest.TestCase):
    def test_projection_numeric_and_filtering(self):
        self.assertEqual(lab.camera_point([20, -1, 0]), [1, 0, 20])
        self.assertEqual(lab.project(lab.camera_point([20, -1, 0])), [680, 360])
        self.assertIsNone(lab.project([0, 0, -1]))
        self.assertIsNone(lab.project([0, 0, 0]))
        self.assertIsNone(lab.project([16, 0, 20]))
        result = lab.fusion()
        self.assertEqual(result["baseline"]["valid_count"], 11)
        self.assertEqual(result["baseline"]["roi_count"], 11)
        self.assertEqual(result["baseline"]["roi_median_optical_depth_m"], 20)

    def test_time_delay_and_same_yaw_compensation(self):
        delayed = lab.fusion(delay_ms=100)
        self.assertEqual(delayed["pixel_offset_from_baseline"], [-20, 0])
        self.assertEqual(delayed["misaligned"]["target_center_pixel"], [660, 360])
        self.assertEqual(delayed["time_compensated_same_yaw"], delayed["baseline"])
        yaw = lab.fusion(yaw_deg=2, delay_ms=100)
        self.assertEqual(yaw["time_compensated_same_yaw"], lab.fusion(yaw_deg=2)["misaligned"])
        self.assertGreater(yaw["time_compensated_same_yaw"]["target_center_pixel"][0], 680)

    def test_voxel_floor_boundary_and_aggregation(self):
        result = lab.voxel()
        self.assertEqual(result["grid_shape_xyz"], [40, 20, 8])
        self.assertEqual(result["accepted_count"], 6)
        self.assertEqual(result["rejected_count"], 4)
        cells = {tuple(v["index_xyz"]): v for v in result["voxels"]}
        negative = cells[(19, 9, 3)]
        self.assertEqual(negative["count"], 2)
        self.assertEqual(negative["cell_center_m"], [-0.25, -0.25, -0.25])
        for value in negative["point_centroid_m"]:
            self.assertAlmostEqual(value, -0.15)
        self.assertIn((0, 0, 0), cells)
        self.assertIn((39, 19, 7), cells)
        self.assertEqual(len(result["bev_occupied_xy"]), 4)
        self.assertEqual(result["dense_grid_memory_example"]["bytes"], 1638400)
        for size in (0, -1, float("nan"), float("inf")):
            with self.subTest(size=size), self.assertRaises(ValueError):
                lab.voxelize([], size)

    def test_action_clipping_metadata(self):
        result = lab.action()
        self.assertEqual(result["clipped_normalized"], [-1, 0.5, 1])
        for actual, expected in zip(result["decoded_increment_m"], [-0.1, 0.05, 0.05]):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(result["metadata"]["shape"], [3])
        self.assertEqual(result["metadata"]["units"], "m")
        self.assertEqual(result["metadata"]["frame"], "toy_robot_base")
        for invalid in ([0, 0], [0, 0, float("nan")]):
            with self.assertRaises(ValueError):
                lab.decode_action(invalid)

    def test_action_latency_boundaries(self):
        for latency, remaining in ((0, 4), (1, 3), (100, 3), (300, 1), (301, 0), (400, 0), (900, 0)):
            with self.subTest(latency=latency):
                result = lab.action(latency)
                self.assertEqual(result["remaining_valid_actions"], remaining)
                self.assertEqual(result["source_state_age_at_arrival_ms"], latency)
        self.assertEqual(lab.action(300)["remaining_chunk_indices"], [3])

    def test_rollout_error_growth(self):
        result = lab.rollout(20)
        self.assertEqual(len(result["oracle_states"]), 21)
        self.assertTrue(all(len(state) == 2 for state in result["predicted_states"]))
        self.assertAlmostEqual(result["oracle_states"][-1][0], 2.4)
        self.assertAlmostEqual(result["oracle_states"][-1][1], 1.4)
        self.assertAlmostEqual(result["teacher_forced_max_abs_error_x_v"][0], 0.001)
        self.assertAlmostEqual(result["teacher_forced_max_abs_error_x_v"][1], 0.02)
        for error in result["rollout_final_abs_error_x_v"]:
            self.assertAlmostEqual(error, 0.4)
        self.assertEqual(result["rollout_max_abs_error_x_v"], result["rollout_final_abs_error_x_v"])
        errors = result["rollout_abs_errors_x_v"]
        self.assertTrue(all(a[0] < b[0] for a, b in zip(errors, errors[1:])))

    def test_argument_validation_and_paths(self):
        invalid = [[], ["fusion", "--yaw-deg", "nan"], ["fusion", "--delay-ms", "-1"],
                   ["voxel", "--size", "0"], ["voxel", "--size", "-0.5"],
                   ["voxel", "--size", "inf"], ["action", "--latency-ms", "-1"],
                   ["rollout", "--steps", "0"], ["rollout", "--steps", "1.5"],
                   ["kitti"], ["kitti", "--points", "relative.bin", "--calib", "/tmp/c.txt"],
                   ["kitti", "--points", "/tmp/p.bin", "--calib", "relative.txt"],
                   ["kitti", "--points", "/tmp/p.bin", "--calib", "/tmp/c.txt", "--width", "0"]]
        invalid.extend([[name, "--work-dir", "relative"] for name in ("fusion", "voxel", "action", "rollout")])
        for argv in invalid:
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                lab.build_parser().parse_args(argv)
        for name in ("fusion", "voxel", "action", "rollout"):
            self.assertEqual(lab.build_parser().parse_args([name]).work_dir, Path("/tmp/inference-case"))

    def test_json_outputs_all_synthetic_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("fusion", "voxel", "action", "rollout"):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    lab.main([name, "--work-dir", directory])
                result = json.loads(output.getvalue())
                self.assertEqual(result, json.loads((Path(directory) / (name + ".json")).read_text()))
                self.assertIn("synthetic", result["notice"])


class KittiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.points = self.directory / "tiny.bin"
        self.calib = self.directory / "calib.txt"
        # Nonidentity rectification and translation plus a nonzero P2 last column.
        self.calib.write_text(
            "P2: 800 0 640 40 0 800 360 0 0 0 1 0\n"
            "R0_rect: 0 -1 0 1 0 0 0 0 1\n"
            "Tr_velo_to_cam: 0 -1 0 1 0 0 -1 0 1 0 0 0\n", encoding="utf-8")
        records = [(20, 1, 1, 0.5), (-5, 0, 0, 0.2), (20, 0, 20, 0.2),
                   (0, 0, 0, 0.1), (float("nan"), 0, 0, 0.1), (20, 1, 16, 0.1)]
        self.points.write_bytes(b"".join(struct.pack("<ffff", *record) for record in records))

    def test_full_matrix_projection_and_csv_cli(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = lab.main(["kitti", "--points", str(self.points), "--calib", str(self.calib),
                               "--width", "1280", "--height", "720", "--work-dir", str(self.directory)])
        self.assertEqual(result["total_points"], 6)
        self.assertEqual(result["visible_count"], 1)
        self.assertEqual(result["rejected_counts"], {"nonfinite": 1, "nonpositive_depth": 2, "outside_fov": 2})
        point = result["visible_points"][0]
        self.assertEqual(point["u_px"], 682)
        self.assertEqual(point["v_px"], 360)
        self.assertEqual(point["rectified_optical_depth_m"], 20)
        self.assertEqual(json.loads(output.getvalue()), result)
        self.assertEqual(json.loads((self.directory / "kitti.json").read_text()), result)
        with (self.directory / "kitti_visible.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 1)
        self.assertEqual(float(rows[0]["u_px"]), 682)

    def test_invalid_bytes_and_calibration(self):
        self.points.write_bytes(b"bad")
        with self.assertRaisesRegex(ValueError, "16"):
            lab.kitti(self.points, self.calib)
        self.calib.write_text("P2: 1 2 3\n")
        with self.assertRaisesRegex(ValueError, "P2"):
            lab.read_kitti_calibration(self.calib)
        self.calib.write_text("R0_rect: 1 0 0 0 1 0 0 0 1\n")
        with self.assertRaisesRegex(ValueError, "缺少"):
            lab.read_kitti_calibration(self.calib)
        self.calib.write_text("P2: nan 0 0 0 0 0 0 0 0 0 0 0\n")
        with self.assertRaisesRegex(ValueError, "P2"):
            lab.read_kitti_calibration(self.calib)

    def test_empty_points_and_absolute_paths(self):
        self.points.write_bytes(b"")
        self.assertEqual(lab.kitti(self.points, self.calib)["visible_count"], 0)
        with self.assertRaisesRegex(ValueError, "绝对路径"):
            lab.kitti("tiny.bin", self.calib)
        args = lab.build_parser().parse_args(["kitti", "--points", str(self.points), "--calib", str(self.calib)])
        self.assertEqual((args.width, args.height), (1242, 375))


if __name__ == "__main__":
    unittest.main()
