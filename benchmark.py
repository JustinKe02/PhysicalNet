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
    python benchmark.py
    python benchmark.py --input_size 256   # test different resolutions
    python benchmark.py --device cpu       # CPU benchmark
"""

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


def main():
    parser = argparse.ArgumentParser(description='RepELA-Net Benchmark')
    parser.add_argument('--input_size', type=int, default=512)
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--runs', type=int, default=200)
    parser.add_argument('--variants', type=str, nargs='+',
                        default=['tiny', 'small', 'base'],
                        choices=['tiny', 'small', 'base'])
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Input: {args.input_size}x{args.input_size}, Classes: {args.num_classes}")

    model_fns = {
        'tiny': ('RepELA-Net-Tiny', repela_net_tiny),
        'small': ('RepELA-Net-Small', repela_net_small),
        'base': ('RepELA-Net-Base', repela_net_base),
    }

    all_results = {}
    for variant in args.variants:
        name, fn = model_fns[variant]
        train_r, deploy_r = benchmark_model(name, fn, args, device)
        all_results[variant] = (train_r, deploy_r)

    # Summary table
    print(f"\n{'='*60}")
    print(f"  SUMMARY TABLE @{args.input_size}x{args.input_size}")
    print(f"{'='*60}")
    header = f"{'Model':<22} {'Params':>8} {'FLOPs':>8} {'FPS':>8} {'Lat(ms)':>8} {'Mem(MB)':>8} {'Size(MB)':>8}"
    print(header)
    print("-" * len(header))

    for variant in args.variants:
        train_r, deploy_r = all_results[variant]
        name, _ = model_fns[variant]

        # Train mode row
        print(f"{name+' (train)':<22} "
              f"{format_num(train_r['params']):>8} "
              f"{format_num(train_r['flops']) if train_r['flops'] else 'N/A':>8} "
              f"{train_r['fps']:>8.1f} "
              f"{train_r['latency']:>8.2f} "
              f"{train_r['memory'] or 0:>8.1f} "
              f"{train_r['size']:>8.2f}")

        # Deploy mode row
        print(f"{name+' (deploy)':<22} "
              f"{format_num(deploy_r['params']):>8} "
              f"{format_num(deploy_r['flops']) if deploy_r['flops'] else 'N/A':>8} "
              f"{deploy_r['fps']:>8.1f} "
              f"{deploy_r['latency']:>8.2f} "
              f"{deploy_r['memory'] or 0:>8.1f} "
              f"{deploy_r['size']:>8.2f}")
        print()


if __name__ == '__main__':
    main()
