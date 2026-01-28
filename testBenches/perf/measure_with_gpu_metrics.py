# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Wrapper script to run measure.py while simultaneously collecting GPU metrics.

This script:
1. Starts GPU monitoring in background (nvidia-smi based)
2. Runs measure.py with all provided arguments
3. Collects and saves GPU metrics (utilization, power, clock, estimated FLOPS/BW)
4. Merges results into a combined JSON output

Usage:
    python3 measure_with_gpu_metrics.py [all measure.py arguments] --metrics_output gpu_metrics.json

Example:
    python3 measure_with_gpu_metrics.py \\
        -c testcases_avg_F08.json \\
        -u uc_avg_F08_TDD.json \\
        -s ../build \\
        -v /workspace/aerial-cuda-accelerated-ran/testVectors \\
        -g 0 -f 1500 -t 40 40 \\
        -p dddsuudddd \\
        --metrics_output gpu_metrics_combined.json
"""

import os
import sys
import json
import subprocess
import threading
import time
import argparse
from datetime import datetime
from collections import defaultdict

# GPU Specifications for FLOPS estimation
GPU_SPECS = {
    "H100_NVL": {
        "sm_count": 132,
        "fp32_cores_per_sm": 128,
        "max_clock_mhz": 1980,
        "memory_bandwidth_gbps": 3350,
    },
    "H100_PCIe": {
        "sm_count": 114,
        "fp32_cores_per_sm": 128,
        "max_clock_mhz": 1620,
        "memory_bandwidth_gbps": 2000,
    },
    "A100": {
        "sm_count": 108,
        "fp32_cores_per_sm": 64,
        "max_clock_mhz": 1410,
        "memory_bandwidth_gbps": 2039,
    }
}


def detect_gpu_type(gpu_id):
    """Detect GPU type from nvidia-smi."""
    try:
        result = subprocess.run(
            f"nvidia-smi -i {gpu_id} --query-gpu=name --format=csv,noheader",
            shell=True, capture_output=True, text=True
        )
        name = result.stdout.strip().upper()
        if "H100" in name:
            if "NVL" in name:
                return "H100_NVL"
            return "H100_PCIe"
        elif "A100" in name:
            return "A100"
        return "H100_NVL"  # Default
    except:
        return "H100_NVL"


class GPUMonitor:
    """Background GPU metrics collector using nvidia-smi."""
    
    def __init__(self, gpu_id, sample_interval=0.1):
        self.gpu_id = gpu_id
        self.sample_interval = sample_interval
        self.samples = []
        self.stop_event = threading.Event()
        self.monitor_thread = None
        self.gpu_type = detect_gpu_type(gpu_id)
        self.spec = GPU_SPECS.get(self.gpu_type, GPU_SPECS["H100_NVL"])
        
    def _collect_sample(self):
        """Collect a single GPU metrics sample."""
        try:
            result = subprocess.run(
                f"nvidia-smi -i {self.gpu_id} --query-gpu=utilization.gpu,utilization.memory,power.draw,clocks.sm,clocks.mem,memory.used --format=csv,noheader,nounits",
                shell=True, capture_output=True, text=True, timeout=2
            )
            parts = [p.strip() for p in result.stdout.strip().split(',')]
            if len(parts) >= 6:
                return {
                    "timestamp": time.time(),
                    "gpu_util_pct": float(parts[0]),
                    "mem_util_pct": float(parts[1]),
                    "power_w": float(parts[2]),
                    "sm_clock_mhz": float(parts[3]),
                    "mem_clock_mhz": float(parts[4]),
                    "memory_used_mb": float(parts[5]),
                }
        except:
            pass
        return None
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while not self.stop_event.is_set():
            sample = self._collect_sample()
            if sample:
                self.samples.append(sample)
            time.sleep(self.sample_interval)
    
    def start(self):
        """Start background monitoring."""
        self.samples = []
        self.stop_event.clear()
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print(f"  📊 GPU monitoring started (GPU {self.gpu_id}, {self.gpu_type})")
    
    def stop(self):
        """Stop monitoring and return collected samples."""
        self.stop_event.set()
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        print(f"  📊 GPU monitoring stopped ({len(self.samples)} samples collected)")
        return self.samples
    
    def calculate_metrics(self):
        """Calculate aggregated metrics from samples."""
        if not self.samples:
            return {}
        
        # Calculate averages and peaks
        metrics = {
            "sample_count": len(self.samples),
            "duration_sec": self.samples[-1]["timestamp"] - self.samples[0]["timestamp"] if len(self.samples) > 1 else 0,
            "gpu_type": self.gpu_type,
        }
        
        # Aggregate metrics
        for key in ["gpu_util_pct", "mem_util_pct", "power_w", "sm_clock_mhz", "memory_used_mb"]:
            values = [s[key] for s in self.samples if key in s]
            if values:
                metrics[f"avg_{key}"] = sum(values) / len(values)
                metrics[f"max_{key}"] = max(values)
                metrics[f"min_{key}"] = min(values)
        
        # Estimate FLOPS based on utilization
        avg_gpu_util = metrics.get("avg_gpu_util_pct", 0)
        avg_sm_clock = metrics.get("avg_sm_clock_mhz", self.spec["max_clock_mhz"])
        
        # Peak GFLOPS at max clock
        peak_gflops = (self.spec["sm_count"] * self.spec["fp32_cores_per_sm"] * 2 * 
                       self.spec["max_clock_mhz"] / 1000)
        
        # Estimated achieved GFLOPS
        clock_ratio = avg_sm_clock / self.spec["max_clock_mhz"]
        metrics["estimated_gflops"] = peak_gflops * (avg_gpu_util / 100) * clock_ratio
        metrics["theoretical_peak_gflops"] = peak_gflops
        
        # Estimated bandwidth
        avg_mem_util = metrics.get("avg_mem_util_pct", 0)
        if avg_mem_util > 1.0:
            metrics["estimated_bandwidth_gbps"] = self.spec["memory_bandwidth_gbps"] * (avg_mem_util / 100)
        else:
            # Rough estimate based on GPU utilization when mem_util is unreliable
            metrics["estimated_bandwidth_gbps"] = self.spec["memory_bandwidth_gbps"] * (avg_gpu_util / 100) * 0.15
        
        metrics["theoretical_peak_bandwidth_gbps"] = self.spec["memory_bandwidth_gbps"]
        
        return metrics
    
    def get_time_series(self):
        """Get time-series data for plotting."""
        if not self.samples:
            return {}
        
        start_time = self.samples[0]["timestamp"]
        return {
            "time_sec": [s["timestamp"] - start_time for s in self.samples],
            "gpu_util_pct": [s["gpu_util_pct"] for s in self.samples],
            "mem_util_pct": [s["mem_util_pct"] for s in self.samples],
            "power_w": [s["power_w"] for s in self.samples],
            "sm_clock_mhz": [s["sm_clock_mhz"] for s in self.samples],
        }


def run_measure_with_monitoring(measure_args, gpu_id, metrics_output):
    """Run measure.py while monitoring GPU metrics."""
    
    # Create GPU monitor
    monitor = GPUMonitor(gpu_id, sample_interval=0.1)
    
    # Build measure.py command
    measure_cmd = f"python3 measure.py {' '.join(measure_args)}"
    
    print("=" * 70)
    print("Running measure.py with GPU Metrics Collection")
    print("=" * 70)
    print(f"Command: {measure_cmd}")
    print(f"GPU: {gpu_id} ({monitor.gpu_type})")
    print(f"Metrics output: {metrics_output}")
    print("=" * 70)
    
    # Start monitoring
    monitor.start()
    start_time = datetime.now()
    
    try:
        # Run measure.py
        result = subprocess.run(measure_cmd, shell=True)
        return_code = result.returncode
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
        return_code = -1
    finally:
        # Stop monitoring
        samples = monitor.stop()
    
    end_time = datetime.now()
    
    # Calculate metrics
    metrics = monitor.calculate_metrics()
    time_series = monitor.get_time_series()
    
    # Prepare output
    output = {
        "timestamp": start_time.isoformat(),
        "duration_sec": (end_time - start_time).total_seconds(),
        "measure_py_return_code": return_code,
        "gpu_metrics": metrics,
        "time_series": time_series,
        "raw_samples": samples,  # Include raw samples for detailed analysis
    }
    
    # Save metrics
    with open(metrics_output, "w") as f:
        json.dump(output, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 70)
    print("GPU Metrics Summary")
    print("=" * 70)
    print(f"  Total Duration: {output['duration_sec']:.1f} sec")
    print(f"  Samples Collected: {metrics.get('sample_count', 0)}")
    print(f"  GPU Type: {metrics.get('gpu_type', 'Unknown')}")
    print()
    print(f"  GPU Utilization:")
    print(f"    Average: {metrics.get('avg_gpu_util_pct', 0):.1f}%")
    print(f"    Peak: {metrics.get('max_gpu_util_pct', 0):.1f}%")
    print()
    print(f"  Power Consumption:")
    print(f"    Average: {metrics.get('avg_power_w', 0):.1f}W")
    print(f"    Peak: {metrics.get('max_power_w', 0):.1f}W")
    print()
    print(f"  SM Clock:")
    print(f"    Average: {metrics.get('avg_sm_clock_mhz', 0):.0f} MHz")
    print()
    print(f"  Estimated Performance:")
    print(f"    GFLOPS: {metrics.get('estimated_gflops', 0):.1f} (peak: {metrics.get('theoretical_peak_gflops', 0):.0f})")
    print(f"    Bandwidth: {metrics.get('estimated_bandwidth_gbps', 0):.1f} GB/s (peak: {metrics.get('theoretical_peak_bandwidth_gbps', 0):.0f})")
    print()
    print(f"  Metrics saved to: {metrics_output}")
    print("=" * 70)
    
    return output


def main():
    # Find --metrics_output argument
    metrics_output = "gpu_metrics_combined.json"
    gpu_id = 0
    
    # Parse our custom arguments
    remaining_args = []
    i = 0
    while i < len(sys.argv[1:]):
        arg = sys.argv[i + 1]
        if arg == "--metrics_output":
            if i + 2 < len(sys.argv):
                metrics_output = sys.argv[i + 2]
                i += 2
                continue
        elif arg in ["-g", "--gpu"]:
            if i + 2 < len(sys.argv):
                gpu_id = int(sys.argv[i + 2])
        remaining_args.append(arg)
        i += 1
    
    if not remaining_args:
        print("Usage: python3 measure_with_gpu_metrics.py [measure.py args] --metrics_output output.json")
        print()
        print("This script runs measure.py while collecting GPU metrics.")
        print("All arguments except --metrics_output are passed to measure.py.")
        print()
        print("Example:")
        print("  python3 measure_with_gpu_metrics.py \\")
        print("      -c testcases_avg_F08.json \\")
        print("      -u uc_avg_F08_TDD.json \\")
        print("      -s ../build \\")
        print("      -v /workspace/aerial-cuda-accelerated-ran/testVectors \\")
        print("      -g 0 -f 1500 -t 40 40 \\")
        print("      -p dddsuudddd \\")
        print("      --metrics_output gpu_metrics.json")
        sys.exit(1)
    
    run_measure_with_monitoring(remaining_args, gpu_id, metrics_output)


if __name__ == "__main__":
    main()
