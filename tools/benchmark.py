"""
RepELA-Net Benchmark: Lightweight Model Metrics.

Measures all metrics needed for paper:
  - Parameters (train vs deploy)
  - FLOPs / MACs (train vs deploy)
  - Inference latency & FPS (GPU warmup + averaged)
  - Peak GPU memory
  - Model file size on disk
  - Comparison across Tiny / Small / Base variants

Usage:
    python tools/benchmark.py
    python tools/benchmark.py --input_size 256   # test different resolutions
    python tools/benchmark.py --device cpu       # CPU benchmark
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
# Ensure cwd = project root so relative paths (Mos2_data/, splits/) work
import os as _os
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
_os.chdir(_PROJECT_ROOT)

import os
import time
import argparse
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.repela_net import (
    RepELANet, repela_net_tiny, repela_net_small, repela_net_base
)

# smp baselines (mirrors tools/train.py)
SMP_MODEL_SPECS = {
    # Tier 1: Lightweight (<5M params)
    'fpn_mnv3s':       ('FPN',            'timm-mobilenetv3_small_100'),
    'unet_mnv3s':      ('Unet',           'timm-mobilenetv3_small_100'),
    'fpn_mv2':         ('FPN',            'mobilenet_v2'),
    'deeplabv3p_mv2':  ('DeepLabV3Plus',  'mobilenet_v2'),
    'deeplabv3p_effb0':('DeepLabV3Plus',  'efficientnet-b0'),
    # Tier 2: Standard (>10M params)
    'unet_r18':        ('Unet',           'resnet18'),
    'unet_r34':        ('Unet',           'resnet34'),
    'deeplabv3p_r18':  ('DeepLabV3Plus',  'resnet18'),
    'pspnet_r18':      ('PSPNet',         'resnet18'),
    'fpn_r18':         ('FPN',            'resnet18'),
    'unet_mit_b0':     ('Unet',           'mit_b0'),
}


def _build_smp_model(model_name, num_classes):
    import segmentation_models_pytorch as smp
    arch, encoder = SMP_MODEL_SPECS[model_name]
    cls = getattr(smp, arch)
    return cls(encoder_name=encoder, encoder_weights='imagenet',
               in_channels=3, classes=num_classes)


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def count_flops(model, input_size, device):
    """Count FLOPs using thop (if available) or manual estimation."""
    dummy = torch.randn(1, 3, input_size, input_size).to(device)

    try:
        from thop import profile, clever_format
        flops, params = profile(model, inputs=(dummy,), verbose=False)
        return flops, params
    except ImportError:
        # Fallback: use fvcore
        try:
            from fvcore.nn import FlopCountAnalysis
            fca = FlopCountAnalysis(model, dummy)
            flops = fca.total()
            params = sum(p.numel() for p in model.parameters())
            return flops, params
        except ImportError:
            print("  [Warning] Install 'thop' or 'fvcore' for FLOPs counting:")
            print("    pip install thop  OR  pip install fvcore")
            return None, None


def measure_latency(model, input_size, device, warmup=50, runs=200):
    """Measure inference latency with GPU synchronization."""
    dummy = torch.randn(1, 3, input_size, input_size).to(device)
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(dummy)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)  # ms

    times.sort()
    # Remove top/bottom 10% outliers
    trim = int(len(times) * 0.1)
    times_trimmed = times[trim:-trim] if trim > 0 else times

    avg = sum(times_trimmed) / len(times_trimmed)
    median = times_trimmed[len(times_trimmed) // 2]
    fps = 1000.0 / avg

    return avg, median, fps, min(times), max(times)


def measure_memory(model, input_size, device):
    """Measure peak GPU memory during inference."""
    if device.type != 'cuda':
        return None, None

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    dummy = torch.randn(1, 3, input_size, input_size).to(device)
    model.eval()

    mem_before = torch.cuda.memory_allocated(device) / 1e6  # MB

    with torch.no_grad():
        _ = model(dummy)

    torch.cuda.synchronize()
    peak_mem = torch.cuda.max_memory_allocated(device) / 1e6  # MB
    mem_after = torch.cuda.memory_allocated(device) / 1e6

    return peak_mem, peak_mem - mem_before


def measure_model_size(model):
    """Measure model file size on disk."""
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=True) as f:
        torch.save(model.state_dict(), f.name)
        size_bytes = os.path.getsize(f.name)
    return size_bytes / (1024 * 1024)  # MB


def format_num(n):
    if n is None:
        return "N/A"
    if n >= 1e9:
        return f"{n/1e9:.2f}G"
    elif n >= 1e6:
        return f"{n/1e6:.2f}M"
    elif n >= 1e3:
        return f"{n/1e3:.2f}K"
    return str(int(n))


def benchmark_model(name, model_fn, args, device):
    """Run full benchmark on a model variant."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    # --- Training Mode ---
    model = model_fn(num_classes=args.num_classes).to(device)
    model.eval()

    total_params, trainable_params = count_parameters(model)
    print(f"\n  [Training Mode]")
    print(f"  Parameters:     {format_num(total_params)} ({total_params/1e6:.2f}M)")

    flops, _ = count_flops(model, args.input_size, device)
    if flops:
        print(f"  FLOPs:          {format_num(flops)} @{args.input_size}x{args.input_size}")
        # MACs = FLOPs / 2 (approximately)
        print(f"  MACs:           {format_num(flops/2)}")

    avg_lat, med_lat, fps, min_lat, max_lat = measure_latency(
        model, args.input_size, device, warmup=args.warmup, runs=args.runs
    )
    print(f"  Latency (avg):  {avg_lat:.2f} ms")
    print(f"  Latency (med):  {med_lat:.2f} ms")
    print(f"  FPS:            {fps:.1f}")

    peak_mem, delta_mem = measure_memory(model, args.input_size, device)
    if peak_mem:
        print(f"  Peak Memory:    {peak_mem:.1f} MB")
        print(f"  Inference Mem:  {delta_mem:.1f} MB")

    size_mb = measure_model_size(model)
    print(f"  Model Size:     {size_mb:.2f} MB")

    train_results = {
        'params': total_params, 'flops': flops,
        'latency': avg_lat, 'fps': fps,
        'memory': peak_mem, 'size': size_mb
    }

    # --- Deploy Mode (Reparameterized) ---
    model.switch_to_deploy()

    deploy_params, _ = count_parameters(model)
    print(f"\n  [Deploy Mode (Reparameterized)]")
    print(f"  Parameters:     {format_num(deploy_params)} ({deploy_params/1e6:.2f}M)")

    deploy_flops, _ = count_flops(model, args.input_size, device)
    if deploy_flops:
        print(f"  FLOPs:          {format_num(deploy_flops)} @{args.input_size}x{args.input_size}")
        print(f"  MACs:           {format_num(deploy_flops/2)}")

    avg_lat_d, med_lat_d, fps_d, _, _ = measure_latency(
        model, args.input_size, device, warmup=args.warmup, runs=args.runs
    )
    print(f"  Latency (avg):  {avg_lat_d:.2f} ms")
    print(f"  FPS:            {fps_d:.1f}")

    peak_mem_d, delta_mem_d = measure_memory(model, args.input_size, device)
    if peak_mem_d:
        print(f"  Peak Memory:    {peak_mem_d:.1f} MB")

    deploy_size = measure_model_size(model)
    print(f"  Model Size:     {deploy_size:.2f} MB")

    # Speedup
    speedup = fps_d / fps if fps > 0 else 0
    print(f"\n  [Deploy Speedup]")
    print(f"  FPS: {fps:.1f} → {fps_d:.1f} ({speedup:.2f}x)")
    if flops and deploy_flops:
        print(f"  FLOPs: {format_num(flops)} → {format_num(deploy_flops)} "
              f"({flops/deploy_flops:.2f}x reduction)")

    deploy_results = {
        'params': deploy_params, 'flops': deploy_flops,
        'latency': avg_lat_d, 'fps': fps_d,
        'memory': peak_mem_d, 'size': deploy_size
    }

    del model
    torch.cuda.empty_cache()

    return train_results, deploy_results


def benchmark_smp_model(name, model_name, args, device):
    """Run benchmark on an smp baseline model (no deploy mode)."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    model = _build_smp_model(model_name, args.num_classes).to(device)
    model.eval()

    total_params, _ = count_parameters(model)
    print(f"\n  Parameters:     {format_num(total_params)} ({total_params/1e6:.2f}M)")

    flops, _ = count_flops(model, args.input_size, device)
    if flops:
        print(f"  FLOPs:          {format_num(flops)} @{args.input_size}x{args.input_size}")
        print(f"  MACs:           {format_num(flops/2)}")

    avg_lat, med_lat, fps, min_lat, max_lat = measure_latency(
        model, args.input_size, device, warmup=args.warmup, runs=args.runs
    )
    print(f"  Latency (avg):  {avg_lat:.2f} ms")
    print(f"  FPS:            {fps:.1f}")

    peak_mem, delta_mem = measure_memory(model, args.input_size, device)
    if peak_mem:
        print(f"  Peak Memory:    {peak_mem:.1f} MB")

    size_mb = measure_model_size(model)
    print(f"  Model Size:     {size_mb:.2f} MB")

    results = {
        'params': total_params, 'flops': flops,
        'latency': avg_lat, 'fps': fps,
        'memory': peak_mem, 'size': size_mb
    }

    del model
    torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser(description='Model Benchmark')
    parser.add_argument('--input_size', type=int, default=512)
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--runs', type=int, default=200)
    parser.add_argument('--variants', type=str, nargs='+',
                        default=['small'],
                        choices=['tiny', 'small', 'base'])
    parser.add_argument('--baselines', type=str, nargs='*', default=None,
                        choices=list(SMP_MODEL_SPECS.keys()),
                        help='smp baselines to benchmark (omit for none, empty for all)')
    parser.add_argument('--no_cpu', action='store_true',
                        help='Skip CPU benchmark')
    parser.add_argument('--cpu_runs', type=int, default=20,
                        help='Number of runs for CPU benchmark (fewer since CPU is slower)')
    args = parser.parse_args()

    has_gpu = torch.cuda.is_available()
    gpu_device = torch.device('cuda') if has_gpu else None
    cpu_device = torch.device('cpu')

    if has_gpu:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Input: {args.input_size}x{args.input_size}, Classes: {args.num_classes}")

    repela_fns = {
        'tiny': ('RepELA-Net-Tiny', repela_net_tiny),
        'small': ('RepELA-Net-Small', repela_net_small),
        'base': ('RepELA-Net-Base', repela_net_base),
    }

    # Collect all results: { name: { 'params', 'flops', 'gpu_fps', 'gpu_lat', 'gpu_mem', 'cpu_fps', 'cpu_lat' } }
    rows = []

    # ── RepELA-Net variants ──
    for variant in args.variants:
        name, fn = repela_fns[variant]
        if has_gpu:
            train_r, deploy_r = benchmark_model(name, fn, args, gpu_device)
        # CPU pass for deploy mode
        cpu_fps, cpu_lat = None, None
        if not args.no_cpu:
            print(f"\n  [CPU benchmark: {name} deploy]")
            cpu_model = fn(num_classes=args.num_classes).to(cpu_device)
            cpu_model.eval()
            cpu_model.switch_to_deploy()
            cpu_args = argparse.Namespace(**vars(args))
            cpu_args.warmup = 5
            cpu_args.runs = args.cpu_runs
            cpu_lat_avg, _, cpu_fps_val, _, _ = measure_latency(
                cpu_model, args.input_size, cpu_device,
                warmup=5, runs=args.cpu_runs)
            cpu_fps, cpu_lat = cpu_fps_val, cpu_lat_avg
            print(f"  CPU FPS: {cpu_fps:.1f}, Latency: {cpu_lat:.1f}ms")
            del cpu_model

        if has_gpu:
            rows.append({
                'name': name + ' (train)', 'params': train_r['params'],
                'flops': train_r['flops'], 'gpu_fps': train_r['fps'],
                'gpu_lat': train_r['latency'], 'gpu_mem': train_r['memory'],
                'cpu_fps': None, 'cpu_lat': None,
            })
            rows.append({
                'name': name + ' (deploy)', 'params': deploy_r['params'],
                'flops': deploy_r['flops'], 'gpu_fps': deploy_r['fps'],
                'gpu_lat': deploy_r['latency'], 'gpu_mem': deploy_r['memory'],
                'cpu_fps': cpu_fps, 'cpu_lat': cpu_lat,
            })

    # ── smp baselines ──
    baseline_list = []
    if args.baselines is not None:
        baseline_list = args.baselines if args.baselines else list(SMP_MODEL_SPECS.keys())

    for bl in baseline_list:
        if has_gpu:
            r = benchmark_smp_model(bl, bl, args, gpu_device)
        else:
            r = {'params': 0, 'flops': None, 'fps': 0, 'latency': 0, 'memory': None}

        cpu_fps, cpu_lat = None, None
        if not args.no_cpu:
            print(f"\n  [CPU benchmark: {bl}]")
            cpu_model = _build_smp_model(bl, args.num_classes).to(cpu_device)
            cpu_model.eval()
            cpu_lat_avg, _, cpu_fps_val, _, _ = measure_latency(
                cpu_model, args.input_size, cpu_device,
                warmup=5, runs=args.cpu_runs)
            cpu_fps, cpu_lat = cpu_fps_val, cpu_lat_avg
            print(f"  CPU FPS: {cpu_fps:.1f}, Latency: {cpu_lat:.1f}ms")
            del cpu_model

        rows.append({
            'name': bl, 'params': r['params'],
            'flops': r['flops'], 'gpu_fps': r['fps'],
            'gpu_lat': r['latency'], 'gpu_mem': r['memory'],
            'cpu_fps': cpu_fps, 'cpu_lat': cpu_lat,
        })

    # ── Summary table ──
    print(f"\n{'='*90}")
    print(f"  SUMMARY TABLE @{args.input_size}x{args.input_size}")
    print(f"{'='*90}")
    header = (f"{'Model':<25} {'Params':>8} {'FLOPs':>8} "
              f"{'GPU FPS':>8} {'GPU ms':>8} {'Mem MB':>8} "
              f"{'CPU FPS':>8} {'CPU ms':>8}")
    print(header)
    print("-" * len(header))

    for row in rows:
        gpu_fps_s = f"{row['gpu_fps']:.1f}" if row['gpu_fps'] else 'N/A'
        gpu_lat_s = f"{row['gpu_lat']:.2f}" if row['gpu_lat'] else 'N/A'
        gpu_mem_s = f"{row['gpu_mem']:.1f}" if row['gpu_mem'] else 'N/A'
        cpu_fps_s = f"{row['cpu_fps']:.1f}" if row['cpu_fps'] else '-'
        cpu_lat_s = f"{row['cpu_lat']:.1f}" if row['cpu_lat'] else '-'
        print(f"{row['name']:<25} "
              f"{format_num(row['params']):>8} "
              f"{format_num(row['flops']) if row['flops'] else 'N/A':>8} "
              f"{gpu_fps_s:>8} "
              f"{gpu_lat_s:>8} "
              f"{gpu_mem_s:>8} "
              f"{cpu_fps_s:>8} "
              f"{cpu_lat_s:>8}")

    print(f"{'='*90}")


if __name__ == '__main__':
    main()

