#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Direct Latency Measurement Script

This script directly measures latency by:
1. Running cubb_gpu_test_bench
2. Parsing raw timing data (t_send, t_recv) from output
3. Calculating latency = t_recv - t_send

This provides more control over the measurement process compared to analyze_kpi.py
which only reads pre-calculated values from JSON files.

Usage:
    python3 measure_latency_direct.py --buffer buffer-08.txt
    python3 measure_latency_direct.py --run --cells 8 --target 60 60
"""

import argparse
import subprocess
import os
import sys
import re
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class LatencyMeasurement:
    """Direct latency measurement from cubb_gpu_test_bench output."""
    
    def __init__(self):
        self.raw_measurements = {
            "dl": {},  # {channel: [(t_send, t_recv, latency), ...]}
            "ul": {}
        }
        self.statistics = {}
        
    def parse_buffer_file(self, filepath: str) -> Dict:
        """
        Parse buffer-XX.txt file and extract raw timing data.
        
        Log format examples:
            Average PUSCH run time: 1635.14 us from 548.48 (averaged over 1 iterations)
            Slot # 0: average PDSCH run time: 129.73 us from 2.72 (averaged over 1 iterations)
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Buffer file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        measurements = {
            "dl": {"PDSCH": [], "PDCCH": [], "SSB": [], "CSIRS": [], "DLBFW": []},
            "ul": {"PUSCH1": [], "PUSCH2": [], "PUCCH1": [], "PUCCH2": [], "PRACH": [], 
                   "SRS1": [], "SRS2": [], "ULBFW1": [], "ULBFW2": []}
        }
        
        pattern_count = 0
        
        for line in lines:
            parts = line.split()
            
            # Skip empty lines
            if not parts:
                continue
            
            # Count patterns
            if len(parts) >= 4 and parts[0] == "Slot" and parts[1] == "pattern":
                pattern_count += 1
                continue
            
            # UL Channels: "Average PUSCH run time: XXX us from YYY"
            # Format: [Average, CHANNEL, run, time:, t_recv, us, from, t_send, ...]
            if len(parts) > 7 and parts[0] == "Average" and parts[2] == "run":
                channel = parts[1]
                try:
                    t_recv = float(parts[4])  # End time (processing complete)
                    t_send = float(parts[7])  # Start time (processing start)
                    latency = t_recv - t_send
                    
                    # Map channel names
                    channel_map = {
                        "PUSCH": "PUSCH1",
                        "PUSCH2": "PUSCH2",
                        "PUCCH": "PUCCH1",
                        "PUCCH2": "PUCCH2",
                        "PRACH": "PRACH",
                        "SRS1": "SRS1",
                        "SRS2": "SRS2",
                        "ULBFW": "ULBFW1",
                        "ULBFW2": "ULBFW2"
                    }
                    
                    mapped_channel = channel_map.get(channel, channel)
                    if mapped_channel in measurements["ul"]:
                        measurements["ul"][mapped_channel].append({
                            "t_send_us": round(t_send, 2),
                            "t_recv_us": round(t_recv, 2),
                            "latency_us": round(latency, 2),
                            "pattern": pattern_count
                        })
                except (ValueError, IndexError):
                    continue
            
            # DL Channels: "Slot # X: average CHANNEL run time: XXX us from YYY"
            # Format: [Slot, #, X:, average, CHANNEL, run, time:, t_recv, us, from, t_send, ...]
            if len(parts) > 10 and parts[0] == "Slot" and parts[1] == "#" and parts[3] == "average":
                channel = parts[4]
                try:
                    slot_idx = int(parts[2].rstrip(':'))
                    t_recv = float(parts[7])  # End time
                    t_send = float(parts[10])  # Start time
                    latency = t_recv - t_send
                    
                    if channel in measurements["dl"]:
                        measurements["dl"][channel].append({
                            "slot": slot_idx,
                            "t_send_us": round(t_send, 2),
                            "t_recv_us": round(t_recv, 2),
                            "latency_us": round(latency, 2),
                            "pattern": pattern_count
                        })
                except (ValueError, IndexError):
                    continue
        
        self.raw_measurements = measurements
        self.pattern_count = pattern_count
        return measurements
    
    def calculate_statistics(self, latencies: List[float]) -> Dict:
        """Calculate comprehensive statistics for latency measurements."""
        if not latencies:
            return None
        
        if HAS_NUMPY:
            arr = np.array(latencies)
            stats = {
                "count": len(arr),
                "avg_us": round(np.mean(arr), 2),
                "std_us": round(np.std(arr), 2),
                "min_us": round(np.min(arr), 2),
                "max_us": round(np.max(arr), 2),
                "p50_us": round(np.percentile(arr, 50), 2),
                "p90_us": round(np.percentile(arr, 90), 2),
                "p95_us": round(np.percentile(arr, 95), 2),
                "p99_us": round(np.percentile(arr, 99), 2),
            }
        else:
            sorted_lat = sorted(latencies)
            n = len(sorted_lat)
            avg = sum(latencies) / n
            variance = sum((x - avg) ** 2 for x in latencies) / n
            stats = {
                "count": n,
                "avg_us": round(avg, 2),
                "std_us": round(variance ** 0.5, 2),
                "min_us": round(min(latencies), 2),
                "max_us": round(max(latencies), 2),
                "p50_us": round(sorted_lat[n // 2], 2),
                "p90_us": round(sorted_lat[int(n * 0.90)], 2),
                "p95_us": round(sorted_lat[int(n * 0.95)], 2),
                "p99_us": round(sorted_lat[min(int(n * 0.99), n - 1)], 2),
            }
        
        # Jitter metrics
        if stats["avg_us"] > 0:
            stats["jitter_cv_pct"] = round((stats["std_us"] / stats["avg_us"]) * 100, 2)
        else:
            stats["jitter_cv_pct"] = 0
        
        return stats
    
    def analyze(self) -> Dict:
        """Analyze all measurements and calculate statistics."""
        results = {
            "timestamp": datetime.now().isoformat(),
            "pattern_count": self.pattern_count,
            "dl": {},
            "ul": {},
            "summary": {}
        }
        
        all_dl_latencies = []
        all_ul_latencies = []
        
        # DL channels
        for channel, measurements in self.raw_measurements["dl"].items():
            if measurements:
                latencies = [m["latency_us"] for m in measurements]
                results["dl"][channel] = {
                    "raw_count": len(measurements),
                    "statistics": self.calculate_statistics(latencies),
                    "sample_data": measurements[:5]  # First 5 samples
                }
                all_dl_latencies.extend(latencies)
        
        # UL channels
        for channel, measurements in self.raw_measurements["ul"].items():
            if measurements:
                latencies = [m["latency_us"] for m in measurements]
                results["ul"][channel] = {
                    "raw_count": len(measurements),
                    "statistics": self.calculate_statistics(latencies),
                    "sample_data": measurements[:5]
                }
                all_ul_latencies.extend(latencies)
        
        # Combined statistics
        if all_dl_latencies:
            results["summary"]["dl_combined"] = self.calculate_statistics(all_dl_latencies)
        if all_ul_latencies:
            results["summary"]["ul_combined"] = self.calculate_statistics(all_ul_latencies)
        if all_dl_latencies or all_ul_latencies:
            results["summary"]["all_combined"] = self.calculate_statistics(
                all_dl_latencies + all_ul_latencies
            )
        
        self.statistics = results
        return results
    
    def print_report(self):
        """Print formatted measurement report."""
        results = self.statistics
        
        print("\n" + "=" * 100)
        print("              DIRECT LATENCY MEASUREMENT REPORT (t_recv - t_send)")
        print("=" * 100)
        print(f"\nTimestamp: {results['timestamp']}")
        print(f"Patterns Processed: {results['pattern_count']}")
        
        # DL Channels
        print("\n┌─ DOWNLINK CHANNELS ──────────────────────────────────────────────────────────────────────────┐")
        print(f"│  {'Channel':<12} {'Count':>8} {'t_send(μs)':>12} {'t_recv(μs)':>12} {'Latency(μs)':>12} │")
        print(f"│  {'─'*12} {'─'*8} {'─'*12} {'─'*12} {'─'*12} │")
        
        for channel, data in results["dl"].items():
            if data and data["sample_data"]:
                sample = data["sample_data"][0]
                stats = data["statistics"]
                print(f"│  {channel:<12} {data['raw_count']:>8} {sample['t_send_us']:>12.2f} {sample['t_recv_us']:>12.2f} {stats['avg_us']:>12.2f} │")
        print("└──────────────────────────────────────────────────────────────────────────────────────────────┘")
        
        # UL Channels
        print("\n┌─ UPLINK CHANNELS ────────────────────────────────────────────────────────────────────────────┐")
        print(f"│  {'Channel':<12} {'Count':>8} {'t_send(μs)':>12} {'t_recv(μs)':>12} {'Latency(μs)':>12} │")
        print(f"│  {'─'*12} {'─'*8} {'─'*12} {'─'*12} {'─'*12} │")
        
        for channel, data in results["ul"].items():
            if data and data["sample_data"]:
                sample = data["sample_data"][0]
                stats = data["statistics"]
                print(f"│  {channel:<12} {data['raw_count']:>8} {sample['t_send_us']:>12.2f} {sample['t_recv_us']:>12.2f} {stats['avg_us']:>12.2f} │")
        print("└──────────────────────────────────────────────────────────────────────────────────────────────┘")
        
        # Statistics Summary
        print("\n┌─ LATENCY STATISTICS (μs) ────────────────────────────────────────────────────────────────────┐")
        print(f"│  {'Category':<15} {'Avg':>10} {'Max':>10} {'P99':>10} {'P95':>10} {'Std(Jitter)':>12} │")
        print(f"│  {'─'*15} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*12} │")
        
        for category, label in [("dl_combined", "Downlink"), ("ul_combined", "Uplink"), ("all_combined", "Combined")]:
            if category in results["summary"]:
                s = results["summary"][category]
                print(f"│  {label:<15} {s['avg_us']:>10.2f} {s['max_us']:>10.2f} {s['p99_us']:>10.2f} {s['p95_us']:>10.2f} {s['std_us']:>12.2f} │")
        print("└──────────────────────────────────────────────────────────────────────────────────────────────┘")
        
        # Sample raw data
        print("\n┌─ SAMPLE RAW MEASUREMENTS ────────────────────────────────────────────────────────────────────┐")
        print(f"│  Pattern  │  Channel  │  Slot  │  t_send (μs)  │  t_recv (μs)  │  Latency (μs)  │")
        print(f"│  {'─'*7}  │  {'─'*7}  │  {'─'*4}  │  {'─'*11}  │  {'─'*11}  │  {'─'*12}  │")
        
        sample_count = 0
        for channel, data in results["dl"].items():
            if data and sample_count < 5:
                for m in data["sample_data"][:2]:
                    print(f"│  {m['pattern']:>7}  │  {channel:<7}  │  {m.get('slot', '-'):>4}  │  {m['t_send_us']:>11.2f}  │  {m['t_recv_us']:>11.2f}  │  {m['latency_us']:>12.2f}  │")
                    sample_count += 1
        
        for channel, data in results["ul"].items():
            if data and sample_count < 10:
                for m in data["sample_data"][:2]:
                    print(f"│  {m['pattern']:>7}  │  {channel:<7}  │  {'-':>4}  │  {m['t_send_us']:>11.2f}  │  {m['t_recv_us']:>11.2f}  │  {m['latency_us']:>12.2f}  │")
                    sample_count += 1
        
        print("└──────────────────────────────────────────────────────────────────────────────────────────────┘")
        print("\n" + "=" * 100)


def run_cubb_test(cells: int, target: Tuple[int, int], pattern: str = "dddsuudddd",
                  iterations: int = 10, delay: int = 0, cuphy_path: str = "../build") -> str:
    """
    Run cubb_gpu_test_bench and capture output.
    
    Returns path to the buffer file.
    """
    vectors_yaml = f"vectors-{str(cells).zfill(2)}.yaml"
    output_file = f"buffer-{str(cells).zfill(2)}.txt"
    
    if pattern == "dddsuudddd":
        u_mode = 5
    elif pattern == "dddsu":
        u_mode = 3
    else:
        u_mode = 5
    
    cmd = f"""CUDA_MPS_PIPE_DIRECTORY=. CUDA_LOG_DIRECTORY=. \\
{cuphy_path}/cubb_gpu_test_bench/cubb_gpu_test_bench \\
    -i {vectors_yaml} \\
    -r {iterations} -w {delay} -u {u_mode} -d 0 -m 1 \\
    --M {target[0]},{target[1]} \\
    --U --D"""
    
    print(f"\n🚀 Running cubb_gpu_test_bench...")
    print(f"   Cells: {cells}, Target: {target}, Pattern: {pattern}")
    print(f"   Command: {cmd[:80]}...")
    
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=300
        )
        
        # Save output to buffer file
        with open(output_file, 'w') as f:
            f.write(result.stdout)
            if result.stderr:
                f.write("\n--- STDERR ---\n")
                f.write(result.stderr)
        
        print(f"   ✅ Output saved to: {output_file}")
        return output_file
        
    except subprocess.TimeoutExpired:
        print("   ❌ Timeout!")
        return None
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Direct Latency Measurement (t_recv - t_send)"
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--buffer", "-b",
        type=str,
        help="Parse existing buffer file"
    )
    mode_group.add_argument(
        "--run", "-r",
        action="store_true",
        help="Run cubb_gpu_test_bench and measure"
    )
    
    # Run options
    parser.add_argument("--cells", "-c", type=int, default=8, help="Number of cells")
    parser.add_argument("--target", "-t", nargs=2, type=int, default=[60, 60], help="SM target [DL UL]")
    parser.add_argument("--pattern", "-p", type=str, default="dddsuudddd", help="TDD pattern")
    parser.add_argument("--iterations", "-i", type=int, default=10, help="Number of iterations")
    parser.add_argument("--delay", "-d", type=int, default=0, help="Delay between patterns (μs)")
    parser.add_argument("--cuphy", type=str, default="../build", help="cuPHY build path")
    
    # Output options
    parser.add_argument("--export", "-e", type=str, help="Export results to JSON file")
    
    args = parser.parse_args()
    
    measurer = LatencyMeasurement()
    
    if args.buffer:
        # Parse existing buffer file
        print(f"\n📂 Parsing buffer file: {args.buffer}")
        measurer.parse_buffer_file(args.buffer)
    
    elif args.run:
        # Run test and measure
        buffer_file = run_cubb_test(
            cells=args.cells,
            target=tuple(args.target),
            pattern=args.pattern,
            iterations=args.iterations,
            delay=args.delay,
            cuphy_path=args.cuphy
        )
        
        if buffer_file:
            measurer.parse_buffer_file(buffer_file)
        else:
            print("❌ Failed to run test")
            sys.exit(1)
    
    # Analyze and report
    results = measurer.analyze()
    measurer.print_report()
    
    # Export if requested
    if args.export:
        import json
        with open(args.export, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Results exported to: {args.export}")


if __name__ == "__main__":
    main()
