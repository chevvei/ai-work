"""PyTorch -> ONNX -> TensorRT 本机 GPU 基准。

用一个 PointPillars 风格的 BEV 主干（伪图像 -> 多尺度卷积 -> 检测头）走完整部署链路：
  1. PyTorch FP32 基线（CUDA Event 计时，不用 time.time，避免异步假象）
  2. 导出 ONNX，并用 onnxruntime 对齐精度
  3. 构建 TensorRT FP32 / FP16 / INT8 引擎，测延迟并与 PyTorch 输出对比
结果写入 logs/bench_<时间>.json 与 .md，作为优化前后对比的证据。

运行：~/venvs/gpu/bin/python gpu-lab/trt_bench.py [--iters 200] [--skip-int8]
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent


class BevBackbone(nn.Module):
    """简化版 PointPillars 主干：输入 (N,64,H,W) 伪图像，输出 cls/box 两个头。"""

    def __init__(self, in_ch=64, num_anchors=2, num_classes=3):
        super().__init__()

        def block(cin, cout, stride, n):
            layers = [nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
            for _ in range(n):
                layers += [nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
            return nn.Sequential(*layers)

        self.b1, self.b2, self.b3 = block(in_ch, 64, 2, 3), block(64, 128, 2, 5), block(128, 256, 2, 5)
        self.up1 = nn.Sequential(nn.ConvTranspose2d(64, 128, 1, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(128, 128, 2, 2, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.up3 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, 4, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.cls = nn.Conv2d(384, num_anchors * num_classes, 1)
        self.box = nn.Conv2d(384, num_anchors * 7, 1)

    def forward(self, x):
        x1 = self.b1(x)
        x2 = self.b2(x1)
        x3 = self.b3(x2)
        feat = torch.cat([self.up1(x1), self.up2(x2), self.up3(x3)], dim=1)
        return self.cls(feat), self.box(feat)


def cuda_time(fn, warmup, iters):
    """套路：GPU 是异步的，必须用 CUDA Event 在流上打点，并先预热（cudnn 选算法、显存池分配）。"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return {"median_ms": round(statistics.median(times), 4), "p99_ms": round(times[int(len(times) * 0.99) - 1], 4),
            "min_ms": round(times[0], 4)}


def compare(ref, out):
    """精度对齐看三项：最大绝对误差、平均绝对误差、余弦相似度（排查量化掉点的第一手证据）。"""
    ref, out = ref.float().flatten(), out.float().flatten()
    diff = (ref - out).abs()
    cos = torch.nn.functional.cosine_similarity(ref, out, dim=0).item()
    return {"max_abs": round(diff.max().item(), 6), "mean_abs": round(diff.mean().item(), 6), "cosine": round(cos, 6)}


def make_calibrator(trt, batches, cache_path):
    """INT8 熵校准器。真实项目必须用几百帧真实点云伪图像；这里用合成数据，只演示流程。"""

    class _Calib(trt.IInt8EntropyCalibrator2):
        def __init__(self):
            super().__init__()
            self.index = 0
            self.device = torch.empty_like(batches[0], device="cuda")

        def get_batch_size(self):
            return batches[0].shape[0]

        def get_batch(self, _names):
            if self.index >= len(batches):
                return None
            self.device.copy_(batches[self.index])
            self.index += 1
            return [int(self.device.data_ptr())]

        def read_calibration_cache(self):
            return cache_path.read_bytes() if cache_path.exists() else None

        def write_calibration_cache(self, cache):
            cache_path.write_bytes(bytes(cache))

    return _Calib()


def build_engine(trt, onnx_path, mode, calib=None):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        raise RuntimeError("; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    if mode in ("fp16", "int8"):
        config.set_flag(trt.BuilderFlag.FP16)
    if mode == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = calib
    t0 = time.time()
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError(f"TensorRT {mode} 构建失败")
    build_s = time.time() - t0
    engine = trt.Runtime(logger).deserialize_cuda_engine(plan)
    return engine, build_s, plan.nbytes


def run_engine(engine, inp, outputs_like):
    """TensorRT 10 推理：按名字绑定 device 指针，execute_async_v3 走当前 torch 流，零拷贝。"""
    ctx = engine.create_execution_context()
    outs = [torch.empty_like(o) for o in outputs_like]
    buffers = {"pillars": inp, "cls": outs[0], "box": outs[1]}  # 按名字绑定，不依赖 IO 张量顺序
    for name in (engine.get_tensor_name(i) for i in range(engine.num_io_tensors)):
        ctx.set_tensor_address(name, buffers[name].data_ptr())
    stream = torch.cuda.current_stream().cuda_stream

    def infer():
        ctx.execute_async_v3(stream)

    return infer, outs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--height", type=int, default=496)
    ap.add_argument("--width", type=int, default=432)
    ap.add_argument("--skip-int8", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=HERE / "logs")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA 不可用：请确认使用 ~/venvs/gpu/bin/python，且 nvidia-smi 正常")
    import tensorrt as trt

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    torch.manual_seed(0)
    model = BevBackbone().cuda().eval()
    # 随机初始化的 BN 统计量接近恒等，先用一批数据刷一次 running stats，让数值分布更像真实模型
    model.train()
    with torch.no_grad():
        for _ in range(5):
            model(torch.rand(1, 64, args.height, args.width, device="cuda"))
    model.eval()
    inp = torch.rand(1, 64, args.height, args.width, device="cuda")

    report = {"time": stamp, "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
              "tensorrt": trt.__version__, "input": list(inp.shape), "results": {}}
    print(f"[env] {report['gpu']} torch {report['torch']} TensorRT {report['tensorrt']} input {report['input']}")

    with torch.no_grad():
        ref = model(inp)
        report["results"]["torch_fp32"] = {"latency": cuda_time(lambda: model(inp), args.warmup, args.iters)}
        with torch.autocast("cuda", dtype=torch.float16):
            amp_out = model(inp)
            report["results"]["torch_amp_fp16"] = {
                "latency": cuda_time(lambda: model(inp), args.warmup, args.iters),
                "accuracy_cls": compare(ref[0], amp_out[0]), "accuracy_box": compare(ref[1], amp_out[1])}

    onnx_path = args.out_dir / "bev_backbone.onnx"
    torch.onnx.export(model, (inp,), str(onnx_path), input_names=["pillars"], output_names=["cls", "box"], opset_version=17)
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        ort_cls, ort_box = sess.run(None, {"pillars": inp.cpu().numpy()})
        report["results"]["onnxruntime_cpu"] = {"accuracy_cls": compare(ref[0].cpu(), torch.from_numpy(ort_cls)),
                                                "accuracy_box": compare(ref[1].cpu(), torch.from_numpy(ort_box))}
    except Exception as exc:  # onnxruntime 只是辅助校验，不阻塞主流程
        report["results"]["onnxruntime_cpu"] = {"error": str(exc)}

    modes = ["fp32", "fp16"] + ([] if args.skip_int8 else ["int8"])
    for mode in modes:
        calib = None
        if mode == "int8":
            batches = [torch.rand(1, 64, args.height, args.width) for _ in range(16)]
            calib = make_calibrator(trt, batches, args.out_dir / "int8_calib.cache")
        engine, build_s, size = build_engine(trt, onnx_path, mode, calib)
        # 套路：enqueueV3 用默认流时 TRT 会额外插 cudaStreamSynchronize，必须换独立流
        with torch.cuda.stream(torch.cuda.Stream()):
            infer, outs = run_engine(engine, inp, ref)
            latency = cuda_time(infer, args.warmup, args.iters)
        torch.cuda.synchronize()
        report["results"][f"trt_{mode}"] = {"latency": latency, "build_s": round(build_s, 1), "engine_mb": round(size / 2**20, 2),
                                            "accuracy_cls": compare(ref[0], outs[0]), "accuracy_box": compare(ref[1], outs[1])}
        print(f"[trt_{mode}] build {build_s:.1f}s median {latency['median_ms']}ms")

    base = report["results"]["torch_fp32"]["latency"]["median_ms"]
    lines = [f"# TensorRT 本机基准 {stamp}", "", f"- GPU: {report['gpu']}", f"- torch {report['torch']} / TensorRT {report['tensorrt']}",
             f"- 输入: {report['input']}，迭代 {args.iters}，预热 {args.warmup}", "",
             "| 后端 | 中位数 ms | P99 ms | 加速比 | cls 余弦 | cls 最大误差 |", "|---|---|---|---|---|---|"]
    for name, r in report["results"].items():
        lat, acc = r.get("latency"), r.get("accuracy_cls", {})
        if lat:
            lines.append(f"| {name} | {lat['median_ms']} | {lat['p99_ms']} | {base / lat['median_ms']:.2f}x | {acc.get('cosine', '-')} | {acc.get('max_abs', '-')} |")
        else:
            lines.append(f"| {name} | - | - | - | {acc.get('cosine', '-')} | {acc.get('max_abs', '-')} |")
    md = "\n".join(lines) + "\n"
    (args.out_dir / f"bench_{stamp}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out_dir / f"bench_{stamp}.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
