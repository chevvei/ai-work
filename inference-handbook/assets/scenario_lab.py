#!/usr/bin/env python3
"""Standard-library synthetic scenario demonstrations (Python 3.8+)."""
import argparse
import csv
import json
import struct
import math
from pathlib import Path
from statistics import median


NOTICE = "全部数据为 synthetic 合成演示，不是真实传感器数据或模型预测；不发送实车命令。"
ROTATION = [[0, -1, 0], [0, 0, -1], [1, 0, 0]]
INTRINSICS = [[800, 0, 640], [0, 800, 360], [0, 0, 1]]
ROI = [620, 300, 720, 420]
RANGES = [(-10, 10), (-5, 5), (-2, 2)]
LOW = [-0.1, -0.1, -0.05]
HIGH = [0.1, 0.1, 0.05]


def finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise argparse.ArgumentTypeError("必须为有限数")
    return number


def positive_float(value):
    number = finite_float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("必须大于 0")
    return number


def nonnegative_float(value):
    number = finite_float(value)
    if number < 0:
        raise argparse.ArgumentTypeError("不能为负数")
    return number


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("必须为正整数")
    return number


def absolute_dir(value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("--work-dir 必须为绝对路径")
    return path


def camera_point(point, yaw_deg=0.0, camera_x_shift=0.0):
    x, y, z = [-point[1] + camera_x_shift, -point[2], point[0]]
    angle = math.radians(yaw_deg)
    c, s = math.cos(angle), math.sin(angle)
    return [c * x + s * z, y, -s * x + c * z]


def project(point):
    x, y, z = point
    if z <= 0:
        return None
    u, v = 800 * x / z + 640, 800 * y / z + 360
    if not (0 <= u < 1280 and 0 <= v < 720):
        return None
    return [u, v]


def projection_report(points, yaw_deg=0.0, delay_ms=0.0):
    projected = []
    for index, point in enumerate(points):
        # Only the first nine points form the moving target; background is static.
        shift = -5 * delay_ms / 1000 if index < 9 else 0.0
        camera = camera_point(point, yaw_deg, shift)
        pixel = project(camera)
        if pixel is not None:
            projected.append({"pixel": pixel, "optical_depth_m": camera[2]})
    inside = [p for p in projected
              if ROI[0] <= p["pixel"][0] < ROI[2]
              and ROI[1] <= p["pixel"][1] < ROI[3]]
    center = project(camera_point(points[4], yaw_deg, -5 * delay_ms / 1000))
    return {"target_center_pixel": center, "valid_count": len(projected),
            "roi_count": len(inside),
            "roi_median_optical_depth_m": median(p["optical_depth_m"] for p in inside) if inside else None,
            "valid_projections": projected}


def fusion(yaw_deg=0.0, delay_ms=0.0):
    points = [[20, y, z] for y in (-1.2, -1.0, -0.8) for z in (-0.2, 0, 0.2)]
    points += [[60, 0, 0], [80, -2, 1], [40, 40, 0], [-5, 0, 0]]
    baseline = projection_report(points)
    shifted = projection_report(points, yaw_deg, delay_ms)
    compensated = projection_report(points, yaw_deg, 0)
    center = shifted["target_center_pixel"]
    base_center = baseline["target_center_pixel"]
    return {"command": "fusion", "notice": NOTICE,
            "frames": {"lidar": "x前 y左 z上", "camera": "x右 y下 z前"},
            "rotation_lidar_to_camera": ROTATION, "translation_m": [0, 0, 0],
            "K": INTRINSICS, "image_size_wh": [1280, 720], "roi_xyxy_half_open": ROI,
            "synthetic_lidar_points_m": points,
            "yaw_deg": yaw_deg, "delay_ms": delay_ms,
            "target_camera_x_speed_m_s": 5,
            "baseline": baseline, "misaligned": shifted,
            "pixel_offset_from_baseline": [center[i] - base_center[i] for i in range(2)] if center else None,
            "time_compensated_same_yaw": compensated,
            "note": "点云落后当前图像 delay_ms，目标沿 camera x 以 +5m/s 运动，老点相对当前位置偏移 -v*dt；再绕 camera y 轴施加 yaw 扰动。正 yaw 通常使投影右移且改变光轴深度。100ms 在 20m、yaw=0 时产生 -20px 横向偏移。时间补偿精确撤销已知合成运动，保留同一 yaw 标定误差；yaw=0 时回到基准。ROI 光轴深度中位数不是欧氏距离，不是3D检测或可靠融合。"}


def voxelize(points, size):
    if not math.isfinite(size) or size <= 0:
        raise ValueError("size 必须为有限正数")
    cells = {}
    rejected = 0
    for point in points:
        if not all(low <= value < high for value, (low, high) in zip(point, RANGES)):
            rejected += 1
            continue
        index = tuple(math.floor((value - RANGES[axis][0]) / size)
                      for axis, value in enumerate(point))
        cells.setdefault(index, []).append(point)
    voxels = [{"index_xyz": list(index), "count": len(group),
               "point_centroid_m": [sum(p[a] for p in group) / len(group) for a in range(3)],
               "cell_center_m": [RANGES[a][0] + (index[a] + 0.5) * size for a in range(3)]}
              for index, group in sorted(cells.items())]
    return voxels, rejected


def voxel(size=0.5):
    points = [[-10, -5, -2], [-0.1, -0.1, -0.1], [-0.2, -0.2, -0.2],
              [0, 0, 0], [0.2, 0.2, 0.2], [9.9, 4.9, 1.9],
              [10, 0, 0], [0, 5, 0], [0, 0, 2], [-10.1, 0, 0]]
    voxels, rejected = voxelize(points, size)
    shape = [math.ceil((high - low) / size) for low, high in RANGES]
    memory = math.prod(shape) * 64 * 4
    return {"command": "voxel", "notice": NOTICE, "size_m": size,
            "range_xyz_half_open_m": RANGES, "synthetic_points_m": points,
            "grid_shape_xyz": shape, "accepted_count": len(points) - rejected,
            "rejected_count": rejected, "voxels": voxels,
            "bev_occupied_xy": [list(index) for index in sorted({tuple(v["index_xyz"][:2]) for v in voxels})],
            "dense_grid_memory_example": {"channels": 64, "dtype": "FP32", "bytes_per_value": 4,
                                          "bytes": memory, "MiB": memory / (1024 ** 2)},
            "note": "索引为 floor((point-range_min)/size)，范围为半开区间，max boundary 被排除。point_centroid 是点均值，cell_center 是规则网格中心；尺寸不整除范围时末格中心可能越界，点仍严格按范围过滤。BEV 仅表示 XY 占用；内存是 dense XYZ*C64*4 的算例，非实测，非完整 PointPillars。"}


def decode_action(normalized):
    if len(normalized) != 3 or not all(math.isfinite(value) for value in normalized):
        raise ValueError("动作必须为有限的三维向量")
    clipped = [max(-1.0, min(1.0, value)) for value in normalized]
    decoded = [low + (value + 1) * 0.5 * (high - low)
               for value, low, high in zip(clipped, LOW, HIGH)]
    return clipped, decoded


def action(latency_ms=300.0):
    if not math.isfinite(latency_ms) or latency_ms < 0:
        raise ValueError("latency_ms 必须为有限非负数")
    normalized = [-1, 0.5, 1.3]
    clipped, decoded = decode_action(normalized)
    chunk = [normalized, [0, 0, 0], [0.2, -0.2, 0], [0.4, 0, -0.4]]
    elapsed = min(4, math.ceil(latency_ms / 100))
    return {"command": "action", "notice": NOTICE,
            "metadata": {"shape": [3], "units": "m", "frame": "toy_robot_base",
                         "components": ["delta_x", "delta_y", "delta_z"],
                         "representation": "translation_increment", "normalized_range": [-1, 1]},
            "p1_low_m": LOW, "p99_high_m": HIGH,
            "normalized_toy": normalized, "clipped_normalized": clipped, "decoded_increment_m": decoded,
            "chunk_normalized": chunk, "chunk_shape": [4, 3], "chunk_step_ms": 100,
            "horizon_steps": 4, "latency_ms": latency_ms,
            "expired_action_count": elapsed, "remaining_valid_actions": 4 - elapsed,
            "remaining_chunk_indices": list(range(elapsed, 4)),
            "source_state_age_at_arrival_ms": latency_ms,
            "note": "toy 假设归一化范围 [-1,1]，先 clip（1.3→1），再用 p1/p99 线性反归一化为米增量；不是所有 VLA 使用相同表示。chunk 对应采样时刻 t=0/100/200/300ms；到达时早于当前时刻的动作过期，恰好同刻仍有效；300ms 后仅剩索引3，状态已旧300ms。过期动作不挤到后续时刻。本演示不执行控制。"}


def oracle_step(state, dt=0.1, acceleration=0.2):
    x, v = state
    return [x + v * dt + 0.5 * acceleration * dt ** 2, v + acceleration * dt]


def predictor_step(state, dt=0.1):
    x, v = state
    return [x + v * dt, v]


def rollout(steps=20):
    if not isinstance(steps, int) or steps <= 0:
        raise ValueError("steps 必须为正整数")
    truth, predicted = [[0.0, 1.0]], [[0.0, 1.0]]
    teacher_predictions, teacher_errors, errors = [], [], []
    for _ in range(steps):
        teacher = predictor_step(truth[-1])
        truth.append(oracle_step(truth[-1]))
        predicted.append(predictor_step(predicted[-1]))
        teacher_predictions.append(teacher)
        teacher_errors.append([abs(teacher[i] - truth[-1][i]) for i in range(2)])
        errors.append([abs(predicted[-1][i] - truth[-1][i]) for i in range(2)])
    return {"command": "rollout", "notice": NOTICE, "steps": steps, "dt_s": 0.1,
            "acceleration_m_s2": 0.2, "state_components": ["x_m", "v_m_s"],
            "oracle_states": truth, "predicted_states": predicted,
            "teacher_forced_next_states": teacher_predictions,
            "teacher_forced_abs_errors_x_v": teacher_errors,
            "rollout_abs_errors_x_v": errors,
            "teacher_forced_max_abs_error_x_v": [max(e[i] for e in teacher_errors) for i in range(2)],
            "rollout_max_abs_error_x_v": [max(e[i] for e in errors) for i in range(2)],
            "rollout_final_abs_error_x_v": errors[-1],
            "note": "oracle 使用恒加速度积分；有偏 predictor 忽略加速度，闭环每步速度保持初值。teacher forcing 每步从 oracle 状态预测一步，闭环则回馈自身预测。状态列表包含初始状态共 steps+1 行；误差分别报告位置和速度，避免混合单位。这是确定性算术演示，非学习世界模型。"}


def read_kitti_calibration(path):
    required = {"P2": (3, 4), "R0_rect": (3, 3), "Tr_velo_to_cam": (3, 4)}
    matrices = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition(":")
        key = key.strip()
        if separator and key in required:
            rows, columns = required[key]
            values = [float(item) for item in value.split()]
            if len(values) != rows * columns or not all(math.isfinite(x) for x in values):
                raise ValueError("无效标定矩阵: " + key)
            matrices[key] = [values[i * columns:(i + 1) * columns] for i in range(rows)]
    missing = set(required) - set(matrices)
    if missing:
        raise ValueError("缺少标定字段: " + ", ".join(sorted(missing)))
    return matrices


def matrix_vector(matrix, vector):
    return [sum(a * b for a, b in zip(row, vector)) for row in matrix]


def kitti(points_path, calib_path, width=1242, height=375):
    if not Path(points_path).is_absolute() or not Path(calib_path).is_absolute():
        raise ValueError("points 和 calib 必须为绝对路径")
    if width <= 0 or height <= 0:
        raise ValueError("图像尺寸必须为正数")
    calibration = read_kitti_calibration(calib_path)
    data = Path(points_path).read_bytes()
    if len(data) % 16:
        raise ValueError("点云字节数必须为16的倍数（每点4个 little-endian float32）")
    visible = []
    rejected = {"nonfinite": 0, "nonpositive_depth": 0, "outside_fov": 0}
    for index, point in enumerate(struct.iter_unpack("<ffff", data)):
        if not all(math.isfinite(value) for value in point):
            rejected["nonfinite"] += 1
            continue
        camera = matrix_vector(calibration["Tr_velo_to_cam"], list(point[:3]) + [1.0])
        rectified = matrix_vector(calibration["R0_rect"], camera)
        homogeneous = matrix_vector(calibration["P2"], rectified + [1.0])
        if not all(math.isfinite(value) for value in rectified + homogeneous):
            rejected["nonfinite"] += 1
            continue
        if rectified[2] <= 0 or homogeneous[2] <= 0:
            rejected["nonpositive_depth"] += 1
            continue
        u, v = homogeneous[0] / homogeneous[2], homogeneous[1] / homogeneous[2]
        if not (0 <= u < width and 0 <= v < height):
            rejected["outside_fov"] += 1
            continue
        visible.append({"point_index": index, "lidar_x_m": point[0], "lidar_y_m": point[1],
                        "lidar_z_m": point[2], "reflectance": point[3], "u_px": u, "v_px": v,
                        "rectified_optical_depth_m": rectified[2]})
    return {"command": "kitti",
            "notice": "用户提供的本地 KITTI object 格式文件；来源和真实性未经验证（测试使用 synthetic 微型文件）。这是几何投影，不是模型预测，不发送实车命令。",
            "points_path": str(points_path), "calib_path": str(calib_path),
            "image_size_wh": [width, height], "calibration": calibration,
            "total_points": len(data) // 16, "visible_count": len(visible),
            "rejected_counts": rejected, "visible_points": visible,
            "visible_points_csv": "kitti_visible.csv",
            "note": "按 P2 @ R0_rect @ Tr_velo_to_cam 对齐齐次坐标计算并除以投影第三分量；要求 rectified optical depth 和投影分母都为正，FOV 为 [0,width)×[0,height)。只读本地文件，不下载数据、不读取图像、不识别目标；建议使用小样本，文件及结果在内存中处理。"}


def build_parser():
    parser = argparse.ArgumentParser(description=NOTICE)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("fusion", "voxel", "action", "rollout", "kitti"):
        sub = commands.add_parser(name)
        sub.add_argument("--work-dir", type=absolute_dir, default=Path("/tmp/inference-case"))
        if name == "fusion":
            sub.add_argument("--yaw-deg", type=finite_float, default=0.0)
            sub.add_argument("--delay-ms", type=nonnegative_float, default=0.0)
        elif name == "voxel":
            sub.add_argument("--size", type=positive_float, default=0.5)
        elif name == "action":
            sub.add_argument("--latency-ms", type=nonnegative_float, default=300.0)
        elif name == "rollout":
            sub.add_argument("--steps", type=positive_int, default=20)
        else:
            sub.add_argument("--points", type=absolute_dir, required=True)
            sub.add_argument("--calib", type=absolute_dir, required=True)
            sub.add_argument("--width", type=positive_int, default=1242)
            sub.add_argument("--height", type=positive_int, default=375)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "fusion":
        result = fusion(args.yaw_deg, args.delay_ms)
    elif args.command == "voxel":
        result = voxel(args.size)
    elif args.command == "action":
        result = action(args.latency_ms)
    elif args.command == "rollout":
        result = rollout(args.steps)
    else:
        result = kitti(args.points, args.calib, args.width, args.height)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    if args.command == "kitti":
        fields = ["point_index", "lidar_x_m", "lidar_y_m", "lidar_z_m", "reflectance",
                  "u_px", "v_px", "rectified_optical_depth_m"]
        with (args.work_dir / "kitti_visible.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(result["visible_points"])
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    (args.work_dir / (args.command + ".json")).write_text(text + "\n", encoding="utf-8")
    print(text)
    return result


if __name__ == "__main__":
    main()
