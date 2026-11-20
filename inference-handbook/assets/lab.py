#!/usr/bin/env python3
"""ResNet18 synthetic deployment smoke test; not a task-quality evaluation."""
import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import time
from urllib.parse import urlparse

SEED = 2026
NOTICE = "固定随机张量链路冒烟；不是实际图像任务，不计算任务 accuracy。"


def absolute_dir(value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("--work-dir 必须为绝对路径")
    return path


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("必须为正整数")
    return number


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(data, ensure_ascii=False, indent=2))


def inspect_environment():
    # Only standard-library discovery: do not import GPU runtimes or dump env vars.
    data = {"python": sys.version.split()[0], "executable": sys.executable,
            "platform": platform.platform(), "packages": {}}
    for name in ("numpy", "torch", "torchvision", "onnx", "onnxruntime", "tensorrt"):
        try:
            data["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            found = importlib.util.find_spec(name) is not None
            data["packages"][name] = "模块可发现，缺少同名发行包版本元数据（未导入验证）" if found else "未安装（当前解释器）"
    binary = shutil.which("nvidia-smi")
    if binary:
        try:
            result = subprocess.run(
                [binary, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10, check=False)
            data["gpu"] = result.stdout.strip() if result.returncode == 0 else "nvidia-smi 查询失败"
        except (OSError, subprocess.TimeoutExpired):
            data["gpu"] = "nvidia-smi 不可用或超时"
    else:
        data["gpu"] = "未找到 nvidia-smi；不代表 torch CUDA 一定可用或不可用"
    data["note"] = "nvidia-smi 的驱动信息不等于已安装 CUDA Toolkit、TensorRT 或兼容的 torch。"
    print(json.dumps(data, ensure_ascii=False, indent=2))


def load_model(args):
    import torch
    from torchvision.models import ResNet18_Weights, resnet18

    weights = ResNet18_Weights.DEFAULT
    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / Path(urlparse(weights.url).path).name
    if not checkpoint.is_file() and not args.allow_download:
        raise RuntimeError("DEFAULT 公开权重未缓存。首次加载会联网下载；确认许可和网络后显式加 --allow-download。")
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("当前 torch 无可用 CUDA；请检查安装或改用 --device cpu。")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    model = resnet18(weights=weights).eval().to(device)
    # Generate on CPU so all backends receive exactly the saved FP32 input.
    tensor = torch.randn(args.batch, 3, 224, 224, generator=torch.Generator().manual_seed(SEED))
    return torch, model, tensor, device, weights.name


def comparison(reference, candidate, atol=1e-4, rtol=1e-3):
    import numpy as np

    if reference.shape != candidate.shape:
        raise ValueError("输出 shape 不一致")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("输出存在 NaN/Inf，不接受数值验证")
    delta = np.abs(reference - candidate)
    return {"allclose": bool(np.allclose(candidate, reference, atol=atol, rtol=rtol)),
            "max_abs": float(delta.max()), "mean_abs": float(delta.mean()),
            "top1_agreement": float(np.mean(reference.argmax(axis=1) == candidate.argmax(axis=1))),
            "atol": atol, "rtol": rtol,
            "note": "top1 agreement 是两个后端的类别一致率，不是对真实标签的 accuracy。"}


def timing_statistics(samples):
    ordered = sorted(samples)
    if not ordered:
        raise ValueError("计时样本不能为空")
    # Linear interpolation, equivalent to numpy percentile's default method.
    position = (len(ordered) - 1) * 0.95
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    p95 = ordered[low] + (ordered[high] - ordered[low]) * (position - low)
    return {"median_ms": statistics.median(ordered), "p95_ms": p95, "samples": len(ordered),
            "p95_method": "linear interpolation at (n-1)*0.95"}


def ort_session(path):
    import onnxruntime as ort

    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def run(args):
    import numpy as np

    work = args.work_dir
    work.mkdir(parents=True, exist_ok=True)
    if args.command == "compare":
        import onnxruntime as ort

        x = np.load(work / "input.npy", allow_pickle=False)
        reference = np.load(work / "baseline.npy", allow_pickle=False)
        metadata = json.loads((work / "baseline.json").read_text(encoding="utf-8"))
        if hashlib.sha256(x.tobytes()).hexdigest() != metadata["input_sha256"]:
            raise ValueError("input.npy 与 baseline.json 不匹配，请重新运行 baseline")
        session = ort_session(work / "resnet18.onnx")
        output = session.run(["logits"], {"input": x})[0]
        np.save(work / "ort-output.npy", output)
        report = comparison(reference, output, args.atol, args.rtol)
        report.update({"notice": NOTICE, "provider": session.get_providers(),
                       "onnxruntime": ort.__version__, "input_shape": list(x.shape)})
        write_json(work / "compare.json", report)
        if not report["allclose"]:
            raise RuntimeError("数值容差未通过；不要直接放宽容差，先检查模型、输入与版本。")
        return

    torch, model, x, device, weights_name = load_model(args)
    metadata = {"notice": NOTICE, "seed": SEED, "model": "resnet18", "weights": weights_name,
                "torch": torch.__version__, "torchvision": importlib.metadata.version("torchvision"),
                "device": device, "shape": list(x.shape), "dtype": "float32",
                "input_sha256": hashlib.sha256(x.numpy().tobytes()).hexdigest()}
    if device == "cuda":
        metadata["gpu"] = torch.cuda.get_device_name()
    if args.command == "baseline":
        with torch.inference_mode():
            output = model(x.to(device)).cpu().numpy()
        np.save(work / "input.npy", x.numpy())
        np.save(work / "baseline.npy", output)
        metadata["output_shape"] = list(output.shape)
        write_json(work / "baseline.json", metadata)
    elif args.command == "export":
        import onnx

        # Export on CPU; dynamic batch only, fixed spatial dimensions.
        with torch.no_grad():
            torch.onnx.export(model.cpu(), x, str(work / "resnet18.onnx"),
                              dynamo=False, opset_version=17,
                              input_names=["input"], output_names=["logits"],
                              dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}})
        graph = onnx.load(str(work / "resnet18.onnx"))
        onnx.checker.check_model(graph)
        metadata.update({"export_device": "cpu", "opset": 17, "onnx_checker": "passed",
                         "input_name": "input", "output_name": "logits", "dynamic_axis": "batch"})
        write_json(work / "export.json", metadata)
    elif args.command == "benchmark":
        if args.backend == "ort":
            session = ort_session(work / "resnet18.onnx")
            array = x.numpy()
            execute = lambda: session.run(["logits"], {"input": array})
            synchronize = lambda: None
            scope = "ORT CPU session.run，含主机 API 和输出分配；不是 GPU 性能"
            metadata["device"] = "cpu"
            metadata["provider"] = session.get_providers()
        else:
            tensor = x.to(device)
            execute = lambda: model(tensor)
            synchronize = (lambda: torch.cuda.synchronize()) if device == "cuda" else (lambda: None)
            scope = "预驻留张量的 forward 墙钟时间，含调用/输出分配与同步开销；不含 H2D/D2H、图像预后处理"
        samples = []
        with torch.inference_mode():
            for _ in range(args.warmup):
                execute()
            synchronize()
            for _ in range(args.runs):
                synchronize()
                start = time.perf_counter()
                execute()
                synchronize()
                samples.append((time.perf_counter() - start) * 1000)
        report = dict(metadata, **timing_statistics(samples), backend=args.backend,
                      warmup=args.warmup, timing_scope=scope, batch=args.batch)
        with (work / "benchmark.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["iteration", "latency_ms"])
            writer.writerows(enumerate(samples, 1))
        write_json(work / "benchmark.json", report)


def main():
    parser = argparse.ArgumentParser(description=NOTICE)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("inspect", help="仅使用标准库检查解释器、包版本与 nvidia-smi；不写文件")
    for command in ("baseline", "export", "compare", "benchmark"):
        child = sub.add_parser(command, help={"baseline": "保存固定输入与 torch 输出",
                              "export": "导出并检查动态 batch ONNX", "compare": "ORT CPU 数值比较",
                              "benchmark": "预热后统计时延"}[command])
        child.add_argument("--work-dir", type=absolute_dir,
                           default=Path("/tmp/inference-handbook-lab"),
                           help="绝对输出目录，默认 /tmp/inference-handbook-lab；同名产物会覆盖")
        if command != "compare":
            child.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
            child.add_argument("--batch", type=positive_int, default=1)
            child.add_argument("--allow-download", action="store_true", help="允许首次加载下载公开 DEFAULT 权重")
        if command == "compare":
            child.add_argument("--atol", type=float, default=1e-4)
            child.add_argument("--rtol", type=float, default=1e-3)
        if command == "benchmark":
            child.add_argument("--backend", choices=("torch", "ort"), default="torch")
            child.add_argument("--warmup", type=positive_int, default=20)
            child.add_argument("--runs", type=positive_int, default=100)
    args = parser.parse_args()
    if args.command == "compare" and (not 0 <= args.atol < float("inf") or not 0 <= args.rtol < float("inf")):
        parser.error("容差必须为有限非负数")
    try:
        if args.command == "inspect":
            inspect_environment()
        else:
            run(args)
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"失败：{error}\n请检查当前解释器、依赖与上一步产物；不会自动安装依赖。\n")


if __name__ == "__main__":
    main()
