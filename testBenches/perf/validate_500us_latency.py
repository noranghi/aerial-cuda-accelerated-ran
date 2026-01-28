# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
500us Latency Validation Script

This script analyzes measure.py results and validates whether all slot latencies
meet the 500us (0.5ms) deadline requirement as per professor's specification.

Key Features:
- Validates each slot's latency against 500us threshold
- Reports deadline miss count and percentage
- Identifies worst-case latencies (jitter)
- Generates detailed reports and visualizations
- Supports multi-file comparison for different configurations

Usage:
    # Single file validation
    python3 validate_500us_latency.py <result_json_file> [options]
    
    # Multi-file comparison
    python3 validate_500us_latency.py file1.json file2.json file3.json --compare

Examples:
    python3 validate_500us_latency.py 040_040_sweep_graphs_avg_F08.json
    python3 validate_500us_latency.py 040_040_sweep_graphs_avg_F08.json --deadline 450
    python3 validate_500us_latency.py 040_040_sweep_graphs_avg_F08.json --plot
    
    # Compare multiple configurations
    python3 validate_500us_latency.py \\
        040_040_sweep_graphs_avg_F08.json \\
        060_060_sweep_graphs_avg_F08.json \\
        066_066_sweep_graphs_avg_F08.json \\
        --compare --plot
"""

import json
import argparse
import sys
import os
import glob
from datetime import datetime

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# Default deadline in microseconds (0.5ms = 500us)
DEFAULT_DEADLINE_US = 500

# Channels to validate
CHANNELS = ['PDSCH', 'PUSCH1', 'PUSCH2']


def load_result_file(filepath):
    """Load measure.py result JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def extract_cell_data(data):
    """Extract cell data from result JSON."""
    cells = {}
    
    for key, value in data.items():
        if key == "testConfig" or key == "constraints" or not isinstance(value, dict):
            continue
        
        # Extract cell count from key (e.g., "01+00" -> 1)
        try:
            cell_count = int(key.split("+")[0])
        except:
            continue
        
        cell_info = {
            "key": key,
            "cell_count": cell_count,
            "latencies": {},
            "stats": {}
        }
        
        # Extract latency arrays for each channel
        for channel in CHANNELS:
            latencies = value.get(channel, [])
            if latencies:
                cell_info["latencies"][channel] = latencies
        
        cells[cell_count] = cell_info
    
    return cells


def validate_latencies(cells, deadline_us):
    """Validate all latencies against the deadline."""
    results = {
        "deadline_us": deadline_us,
        "cells": {},
        "summary": {
            "total_slots": 0,
            "total_violations": 0,
            "overall_pass": True,
            "capacity_500us": 0
        }
    }
    
    all_pass_up_to = 0
    
    for cell_count in sorted(cells.keys()):
        cell = cells[cell_count]
        cell_result = {
            "cell_count": cell_count,
            "channels": {},
            "all_pass": True,
            "total_slots": 0,
            "total_violations": 0
        }
        
        for channel in CHANNELS:
            latencies = cell["latencies"].get(channel, [])
            if not latencies:
                continue
            
            # Calculate statistics
            if HAS_NUMPY:
                lat_array = np.array(latencies)
                avg = np.mean(lat_array)
                std = np.std(lat_array)
                min_lat = np.min(lat_array)
                max_lat = np.max(lat_array)
                p50 = np.percentile(lat_array, 50)
                p95 = np.percentile(lat_array, 95)
                p99 = np.percentile(lat_array, 99)
            else:
                latencies_sorted = sorted(latencies)
                avg = sum(latencies) / len(latencies)
                min_lat = min(latencies)
                max_lat = max(latencies)
                p50 = latencies_sorted[len(latencies) // 2]
                p95 = latencies_sorted[int(len(latencies) * 0.95)]
                p99 = latencies_sorted[int(len(latencies) * 0.99)]
                std = (sum((x - avg) ** 2 for x in latencies) / len(latencies)) ** 0.5
            
            # Find violations
            violations = [l for l in latencies if l > deadline_us]
            violation_count = len(violations)
            total_count = len(latencies)
            ontime_pct = (total_count - violation_count) / total_count * 100
            
            channel_result = {
                "total_slots": total_count,
                "violations": violation_count,
                "ontime_pct": ontime_pct,
                "pass": violation_count == 0,
                "stats": {
                    "avg_us": round(avg, 2),
                    "std_us": round(std, 2),
                    "min_us": round(min_lat, 2),
                    "max_us": round(max_lat, 2),
                    "p50_us": round(p50, 2),
                    "p95_us": round(p95, 2),
                    "p99_us": round(p99, 2)
                },
                "violation_latencies": sorted(violations, reverse=True)[:10]  # Top 10 worst
            }
            
            cell_result["channels"][channel] = channel_result
            cell_result["total_slots"] += total_count
            cell_result["total_violations"] += violation_count
            
            if violation_count > 0:
                cell_result["all_pass"] = False
        
        results["cells"][cell_count] = cell_result
        results["summary"]["total_slots"] += cell_result["total_slots"]
        results["summary"]["total_violations"] += cell_result["total_violations"]
        
        if cell_result["all_pass"]:
            all_pass_up_to = cell_count
        else:
            results["summary"]["overall_pass"] = False
    
    results["summary"]["capacity_500us"] = all_pass_up_to
    
    return results


def print_report(results, data):
    """Print formatted validation report."""
    deadline = results["deadline_us"]
    
    # Header
    print("\n" + "=" * 90)
    print(f"{'500us LATENCY VALIDATION REPORT':^90}")
    print("=" * 90)
    
    # Test Configuration
    config = data.get("testConfig", {})
    target = config.get("target", ["?", "?"])
    freq = config.get("freq", "?")
    pattern = config.get("pattern", "?")
    gpu = config.get("gpuName", "Unknown")
    
    print(f"\n📋 Test Configuration:")
    print(f"   Target SM (DL/UL): {target[0]}/{target[1]}")
    print(f"   GPU Frequency:     {freq} MHz")
    print(f"   TDD Pattern:       {pattern}")
    print(f"   GPU:               {gpu}")
    print(f"   Deadline:          {deadline} μs (0.{deadline//100}ms)")
    
    # Summary
    summary = results["summary"]
    print(f"\n📊 Overall Summary:")
    print(f"   Total Slots Analyzed:  {summary['total_slots']:,}")
    print(f"   Total Violations:      {summary['total_violations']:,}")
    violation_pct = (summary['total_violations'] / summary['total_slots'] * 100) if summary['total_slots'] > 0 else 0
    print(f"   Violation Rate:        {violation_pct:.4f}%")
    
    if summary['overall_pass']:
        print(f"\n   ✅ OVERALL RESULT: PASS (All slots under {deadline}μs)")
    else:
        print(f"\n   ❌ OVERALL RESULT: FAIL (Some slots exceeded {deadline}μs)")
    
    print(f"   📈 500μs Capacity:     {summary['capacity_500us']} cells")
    
    # Detailed per-cell results
    print("\n" + "-" * 90)
    print(f"{'CELL':<6} {'CHANNEL':<8} {'SLOTS':>8} {'VIOLATIONS':>12} {'ONTIME%':>10} {'AVG(μs)':>10} {'MAX(μs)':>10} {'STATUS':>8}")
    print("-" * 90)
    
    for cell_count in sorted(results["cells"].keys()):
        cell = results["cells"][cell_count]
        first_row = True
        
        for channel in CHANNELS:
            if channel not in cell["channels"]:
                continue
            
            ch = cell["channels"][channel]
            status = "✅ PASS" if ch["pass"] else "❌ FAIL"
            
            if first_row:
                cell_str = f"{cell_count:>4}"
                first_row = False
            else:
                cell_str = ""
            
            print(f"{cell_str:<6} {channel:<8} {ch['total_slots']:>8} {ch['violations']:>12} "
                  f"{ch['ontime_pct']:>9.2f}% {ch['stats']['avg_us']:>10.1f} "
                  f"{ch['stats']['max_us']:>10.1f} {status:>8}")
        
        # Cell summary row
        cell_status = "✅" if cell["all_pass"] else "❌"
        print(f"{'':6} {'TOTAL':<8} {cell['total_slots']:>8} {cell['total_violations']:>12} "
              f"{'':<10} {'':<10} {'':<10} {cell_status:>8}")
        print("-" * 90)
    
    # Worst violations details
    print(f"\n🔍 Worst Violations (Top 10 per channel):")
    print("-" * 60)
    
    found_violations = False
    for cell_count in sorted(results["cells"].keys()):
        cell = results["cells"][cell_count]
        for channel in CHANNELS:
            if channel not in cell["channels"]:
                continue
            
            ch = cell["channels"][channel]
            if ch["violation_latencies"]:
                found_violations = True
                violations_str = ", ".join([f"{v:.1f}" for v in ch["violation_latencies"][:5]])
                print(f"   {cell_count} cells / {channel}: {violations_str} μs")
    
    if not found_violations:
        print("   ✅ No violations found!")
    
    # Jitter Analysis
    print(f"\n📈 Jitter Analysis (P99 - P50):")
    print("-" * 60)
    
    for cell_count in sorted(results["cells"].keys()):
        cell = results["cells"][cell_count]
        jitters = []
        for channel in CHANNELS:
            if channel not in cell["channels"]:
                continue
            ch = cell["channels"][channel]
            jitter = ch["stats"]["p99_us"] - ch["stats"]["p50_us"]
            jitters.append(f"{channel}: {jitter:.1f}μs")
        
        print(f"   {cell_count} cells: {', '.join(jitters)}")
    
    print("\n" + "=" * 90)


def plot_results(results, data, output_prefix):
    """Generate visualization plots."""
    if not HAS_MATPLOTLIB:
        print("⚠️ matplotlib not available, skipping plots")
        return
    
    deadline = results["deadline_us"]
    config = data.get("testConfig", {})
    target = config.get("target", ["?", "?"])
    freq = config.get("freq", "?")
    
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(f"500μs Latency Validation Report\n"
                 f"Target: {target[0]}/{target[1]} | Freq: {freq} MHz | Deadline: {deadline}μs",
                 fontsize=14, fontweight='bold')
    
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    colors = {
        'PDSCH': '#2ecc71',
        'PUSCH1': '#3498db',
        'PUSCH2': '#e74c3c'
    }
    
    cell_counts = sorted(results["cells"].keys())
    
    # ===== Plot 1: Max Latency vs Cell Count =====
    ax1 = fig.add_subplot(gs[0, 0])
    
    for channel, color in colors.items():
        max_lats = []
        for c in cell_counts:
            if channel in results["cells"][c]["channels"]:
                max_lats.append(results["cells"][c]["channels"][channel]["stats"]["max_us"])
            else:
                max_lats.append(0)
        ax1.plot(cell_counts, max_lats, 'o-', label=channel, color=color, linewidth=2, markersize=6)
    
    ax1.axhline(y=deadline, color='red', linestyle='--', linewidth=2, label=f'{deadline}μs deadline')
    ax1.fill_between(cell_counts, 0, deadline, alpha=0.1, color='green')
    ax1.fill_between(cell_counts, deadline, ax1.get_ylim()[1] if ax1.get_ylim()[1] > deadline else deadline * 1.5, 
                     alpha=0.1, color='red')
    
    ax1.set_xlabel('Cell Count', fontsize=11)
    ax1.set_ylabel('Max Latency (μs)', fontsize=11)
    ax1.set_title('Max Latency vs Cell Count', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(cell_counts)
    
    # ===== Plot 2: P99 Latency vs Cell Count =====
    ax2 = fig.add_subplot(gs[0, 1])
    
    for channel, color in colors.items():
        p99_lats = []
        for c in cell_counts:
            if channel in results["cells"][c]["channels"]:
                p99_lats.append(results["cells"][c]["channels"][channel]["stats"]["p99_us"])
            else:
                p99_lats.append(0)
        ax2.plot(cell_counts, p99_lats, 'o-', label=channel, color=color, linewidth=2, markersize=6)
    
    ax2.axhline(y=deadline, color='red', linestyle='--', linewidth=2, label=f'{deadline}μs deadline')
    ax2.set_xlabel('Cell Count', fontsize=11)
    ax2.set_ylabel('P99 Latency (μs)', fontsize=11)
    ax2.set_title('P99 Latency vs Cell Count', fontsize=12)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(cell_counts)
    
    # ===== Plot 3: On-Time Percentage =====
    ax3 = fig.add_subplot(gs[0, 2])
    
    for channel, color in colors.items():
        ontime_pcts = []
        for c in cell_counts:
            if channel in results["cells"][c]["channels"]:
                ontime_pcts.append(results["cells"][c]["channels"][channel]["ontime_pct"])
            else:
                ontime_pcts.append(100)
        ax3.plot(cell_counts, ontime_pcts, 'o-', label=channel, color=color, linewidth=2, markersize=6)
    
    ax3.axhline(y=100, color='green', linestyle=':', alpha=0.5)
    ax3.set_xlabel('Cell Count', fontsize=11)
    ax3.set_ylabel('On-Time %', fontsize=11)
    ax3.set_title(f'On-Time % (< {deadline}μs)', fontsize=12)
    ax3.legend(loc='lower left')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-5, 110)
    ax3.set_xticks(cell_counts)
    
    # ===== Plot 4: Violation Count =====
    ax4 = fig.add_subplot(gs[1, 0])
    
    x = range(len(cell_counts))
    width = 0.25
    
    for i, (channel, color) in enumerate(colors.items()):
        violations = []
        for c in cell_counts:
            if channel in results["cells"][c]["channels"]:
                violations.append(results["cells"][c]["channels"][channel]["violations"])
            else:
                violations.append(0)
        ax4.bar([xi + i * width for xi in x], violations, width, label=channel, color=color, alpha=0.8)
    
    ax4.set_xticks([xi + width for xi in x])
    ax4.set_xticklabels(cell_counts)
    ax4.set_xlabel('Cell Count', fontsize=11)
    ax4.set_ylabel('Violation Count', fontsize=11)
    ax4.set_title('Deadline Violations per Cell', fontsize=12)
    ax4.legend(loc='upper left')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # ===== Plot 5: Jitter (P99 - P50) =====
    ax5 = fig.add_subplot(gs[1, 1])
    
    for channel, color in colors.items():
        jitters = []
        for c in cell_counts:
            if channel in results["cells"][c]["channels"]:
                stats = results["cells"][c]["channels"][channel]["stats"]
                jitters.append(stats["p99_us"] - stats["p50_us"])
            else:
                jitters.append(0)
        ax5.plot(cell_counts, jitters, 'o-', label=channel, color=color, linewidth=2, markersize=6)
    
    ax5.set_xlabel('Cell Count', fontsize=11)
    ax5.set_ylabel('Jitter (P99 - P50) μs', fontsize=11)
    ax5.set_title('Latency Jitter vs Cell Count', fontsize=12)
    ax5.legend(loc='upper left')
    ax5.grid(True, alpha=0.3)
    ax5.set_xticks(cell_counts)
    
    # ===== Plot 6: Summary Box =====
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    summary = results["summary"]
    capacity = summary["capacity_500us"]
    
    status_emoji = "✅ PASS" if summary["overall_pass"] else "❌ FAIL"
    violation_pct = (summary['total_violations'] / summary['total_slots'] * 100) if summary['total_slots'] > 0 else 0
    
    summary_text = f"""
╔═══════════════════════════════════════╗
║      500μs VALIDATION SUMMARY         ║
╠═══════════════════════════════════════╣
║                                       ║
║  Deadline:        {deadline:>6} μs            ║
║  Total Slots:     {summary['total_slots']:>6,}             ║
║  Violations:      {summary['total_violations']:>6,}             ║
║  Violation Rate:  {violation_pct:>6.3f}%            ║
║                                       ║
╠═══════════════════════════════════════╣
║                                       ║
║  500μs Capacity:  {capacity:>6} cells          ║
║                                       ║
║  Status: {status_emoji:<26}  ║
║                                       ║
╚═══════════════════════════════════════╝"""
    
    ax6.text(0.1, 0.95, summary_text, transform=ax6.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_file = f"{output_prefix}_500us_validation.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n📊 Plot saved: {output_file}")
    plt.close()


def save_json_report(results, data, output_prefix):
    """Save detailed JSON report."""
    output = {
        "validation_timestamp": datetime.now().isoformat(),
        "deadline_us": results["deadline_us"],
        "test_config": data.get("testConfig", {}),
        "summary": results["summary"],
        "per_cell_results": results["cells"]
    }
    
    output_file = f"{output_prefix}_500us_validation.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"💾 JSON report saved: {output_file}")


def get_unique_filename(base_path, extension):
    """Generate unique filename with incrementing counter."""
    counter = 1
    while True:
        filename = f"{base_path}_{counter}{extension}"
        if not os.path.exists(filename):
            return filename
        counter += 1


def process_multiple_files(filepaths, deadline_us):
    """Process multiple files and return comparison data."""
    all_results = []
    
    for filepath in filepaths:
        if not os.path.exists(filepath):
            print(f"⚠️ Skipping: {filepath} (not found)")
            continue
        
        try:
            data = load_result_file(filepath)
            cells = extract_cell_data(data)
            
            if not cells:
                print(f"⚠️ Skipping: {filepath} (no cell data)")
                continue
            
            results = validate_latencies(cells, deadline_us)
            
            # Extract config info
            config = data.get("testConfig", {})
            target = config.get("target", ["?", "?"])
            freq = config.get("freq", "?")
            pattern = config.get("pattern", "?")
            iterations = config.get("iterations", "?")
            delay = config.get("delay", "?")
            
            all_results.append({
                "filepath": filepath,
                "filename": os.path.basename(filepath),
                "target_label": f"{target[0]}/{target[1]}",
                "freq": freq,
                "pattern": pattern,
                "iterations": iterations,
                "delay": delay,
                "results": results,
                "data": data
            })
            
            print(f"  ✅ Processed: {os.path.basename(filepath)} (Target: {target[0]}/{target[1]}, Freq: {freq} MHz, Iter: {iterations}, Delay: {delay})")
            
        except Exception as e:
            print(f"⚠️ Error processing {filepath}: {e}")
    
    return all_results


def print_comparison_summary(all_results, deadline_us):
    """Print comparison summary table."""
    print("\n" + "=" * 140)
    print(f"{'500μs LATENCY VALIDATION - MULTI-FILE COMPARISON':^140}")
    print(f"{'Deadline: ' + str(deadline_us) + 'μs':^140}")
    print("=" * 140)
    
    # Header
    print(f"\n{'Target':<12} {'Freq':>8} {'Iters':>8} {'Delay':>12} {'500μs Cap':>10} {'Violations':>12} {'Rate':>10} {'Status':>10}")
    print(f"{'(DL/UL)':<12} {'(MHz)':>8} {'':>8} {'(μs)':>12} {'(cells)':>10} {'':>12} {'':>10} {'':>10}")
    print("-" * 140)
    
    best_capacity = 0
    best_result = None
    
    for r in sorted(all_results, key=lambda x: (x['freq'] if isinstance(x['freq'], (int, float)) else 0, x['iterations'] if isinstance(x['iterations'], (int, float)) else 0)):
        summary = r['results']['summary']
        capacity = summary['capacity_500us']
        violations = summary['total_violations']
        total = summary['total_slots']
        rate = (violations / total * 100) if total > 0 else 0
        status = "✅ PASS" if summary['overall_pass'] else "❌ FAIL"
        
        if capacity > best_capacity:
            best_capacity = capacity
            best_result = r
        
        freq_str = f"{r['freq']}" if r['freq'] != "?" else "N/A"
        iters_str = f"{r['iterations']}" if r['iterations'] != "?" else "N/A"
        delay_str = f"{r['delay']}" if r.get('delay', '?') != "?" else "N/A"
        
        print(f"{r['target_label']:<12} {freq_str:>8} {iters_str:>8} {delay_str:>12} {capacity:>10} {violations:>12,} {rate:>9.2f}% {status:>10}")
    
    print("-" * 140)
    
    if best_result:
        delay_info = f", Delay: {best_result.get('delay', '?')}μs" if best_result.get('delay', '?') != '?' else ""
        print(f"\n🏆 Best 500μs Capacity: Target {best_result['target_label']} @ {best_result['freq']} MHz, Iter: {best_result['iterations']}{delay_info} → {best_capacity} cells")
    
    print("=" * 140)


def plot_comparison(all_results, deadline_us, output_prefix):
    """Generate comparison plots for multiple files."""
    if not HAS_MATPLOTLIB:
        print("⚠️ matplotlib not available, skipping plots")
        return
    
    if not HAS_NUMPY:
        print("⚠️ numpy not available, skipping plots")
        return
    
    # Sort results by frequency and target
    all_results = sorted(all_results, key=lambda x: (x['freq'] if isinstance(x['freq'], (int, float)) else 0, x['target_label']))
    
    # Get unique frequencies for subtitle
    freqs = set(r['freq'] for r in all_results if r['freq'] != "?")
    freq_str = ", ".join([f"{f} MHz" for f in sorted(freqs)]) if freqs else "N/A"
    
    fig = plt.figure(figsize=(22, 16))
    fig.suptitle(f"500μs Latency Validation Comparison\n"
                 f"Deadline: {deadline_us}μs | Frequencies: {freq_str}",
                 fontsize=14, fontweight='bold', y=0.98)
    
    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
    
    # Generate color map
    cmap = plt.cm.viridis
    result_colors = [cmap(i / max(len(all_results) - 1, 1)) for i in range(len(all_results))]
    
    # Get all cell counts
    all_cell_counts = set()
    for r in all_results:
        all_cell_counts.update(r['results']['cells'].keys())
    cell_counts = sorted(all_cell_counts)
    
    # ===== Plot 1: 500μs Capacity Comparison =====
    ax1 = fig.add_subplot(gs[0, 0])
    
    labels = []
    capacities = []
    for r in all_results:
        # Create detailed label with iterations and delay
        iter_str = f"i={r['iterations']}" if r['iterations'] != "?" else ""
        delay_str = f"d={r.get('delay', '?')}" if r.get('delay', '?') != "?" else ""
        condition = ", ".join(filter(None, [iter_str, delay_str]))
        
        if len(freqs) > 1:
            labels.append(f"{r['target_label']}\n@{r['freq']}MHz\n{condition}")
        else:
            labels.append(f"{r['target_label']}\n{condition}")
        capacities.append(r['results']['summary']['capacity_500us'])
    
    bars = ax1.bar(range(len(all_results)), capacities, color=result_colors, edgecolor='black', alpha=0.8)
    ax1.set_xticks(range(len(all_results)))
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    ax1.set_xlabel('Target SM (DL/UL)', fontsize=11)
    ax1.set_ylabel('500μs Capacity (cells)', fontsize=11)
    ax1.set_title('500μs Capacity by Configuration', fontsize=11, pad=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    for bar, cap in zip(bars, capacities):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{cap}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # ===== Plot 2: Violation Rate Comparison =====
    ax2 = fig.add_subplot(gs[0, 1])
    
    rates = []
    for r in all_results:
        summary = r['results']['summary']
        rate = (summary['total_violations'] / summary['total_slots'] * 100) if summary['total_slots'] > 0 else 0
        rates.append(rate)
    
    bars = ax2.bar(range(len(all_results)), rates, color=result_colors, edgecolor='black', alpha=0.8)
    ax2.set_xticks(range(len(all_results)))
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    ax2.set_xlabel('Target SM (DL/UL)', fontsize=11)
    ax2.set_ylabel('Violation Rate (%)', fontsize=11)
    ax2.set_title('Deadline Violation Rate', fontsize=11, pad=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, rate in zip(bars, rates):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # ===== Plot 3: PUSCH1 Max Latency Comparison =====
    ax3 = fig.add_subplot(gs[0, 2])
    
    for idx, (r, color) in enumerate(zip(all_results, result_colors)):
        max_lats = []
        valid_cells = []
        for c in cell_counts:
            if c in r['results']['cells'] and 'PUSCH1' in r['results']['cells'][c]['channels']:
                max_lats.append(r['results']['cells'][c]['channels']['PUSCH1']['stats']['max_us'])
                valid_cells.append(c)
        
        # Create detailed label with iterations and delay
        iter_str = f"i={r['iterations']}" if r['iterations'] != "?" else ""
        delay_str = f"d={r.get('delay', '?')}" if r.get('delay', '?') != "?" else ""
        condition = ", ".join(filter(None, [iter_str, delay_str]))
        
        if len(freqs) > 1:
            label = f"{r['target_label']} @{r['freq']}MHz ({condition})"
        else:
            label = f"{r['target_label']} ({condition})"
        
        if max_lats:
            ax3.plot(valid_cells, max_lats, 'o-', label=label, color=color, linewidth=2, markersize=5)
    
    ax3.axhline(y=deadline_us, color='red', linestyle='--', linewidth=2, label=f'{deadline_us}μs deadline')
    ax3.set_xlabel('Cell Count', fontsize=11)
    ax3.set_ylabel('PUSCH1 Max Latency (μs)', fontsize=11)
    ax3.set_title('PUSCH1 Max Latency (Main Bottleneck)', fontsize=11, pad=10)
    ax3.legend(loc='upper left', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # ===== Plot 4: On-Time % for All Channels (at each config's capacity point) =====
    ax4 = fig.add_subplot(gs[1, 0])
    
    x = np.arange(len(all_results))
    n_channels = 3
    total_width = 0.7  # Total width for all 3 bars per group
    width = total_width / n_channels
    
    channels = ['PDSCH', 'PUSCH1', 'PUSCH2']
    channel_colors = {'PDSCH': '#27ae60', 'PUSCH1': '#2980b9', 'PUSCH2': '#c0392b'}
    
    for i, channel in enumerate(channels):
        ontime_at_cap = []
        for r in all_results:
            # Get on-time at highest cell count (or at a specific cell like 4 or 8)
            test_cell = min(8, max(r['results']['cells'].keys()))
            if test_cell in r['results']['cells'] and channel in r['results']['cells'][test_cell]['channels']:
                ontime_at_cap.append(r['results']['cells'][test_cell]['channels'][channel]['ontime_pct'])
            else:
                ontime_at_cap.append(100)
        
        # Offset each channel's bars side by side
        offset = (i - (n_channels - 1) / 2) * width
        bars = ax4.bar(x + offset, ontime_at_cap, width * 0.85, label=channel, 
                color=channel_colors[channel], alpha=0.9, edgecolor='black', linewidth=1)
        
        # Add value labels on each bar
        for bar, val in zip(bars, ontime_at_cap):
            if val < 100:  # Only label non-100% values
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{val:.0f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax4.axhline(y=100, color='green', linestyle='--', alpha=0.7, linewidth=1.5)
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels, rotation=45, ha='right')
    ax4.set_xlabel('Configuration', fontsize=11)
    ax4.set_ylabel('On-Time % @ 8 cells', fontsize=11)
    ax4.set_title('On-Time % at 8 Cells (All Channels)', fontsize=11, pad=10)
    ax4.legend(loc='upper right', fontsize=9, ncol=3)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim(0, 115)
    
    # ===== Plot 5: PUSCH1 On-Time % vs Cell Count =====
    ax5 = fig.add_subplot(gs[1, 1])
    
    for idx, (r, color) in enumerate(zip(all_results, result_colors)):
        ontime_pcts = []
        valid_cells = []
        for c in cell_counts:
            if c in r['results']['cells'] and 'PUSCH1' in r['results']['cells'][c]['channels']:
                ontime_pcts.append(r['results']['cells'][c]['channels']['PUSCH1']['ontime_pct'])
                valid_cells.append(c)
        
        # Create detailed label with iterations and delay
        iter_str = f"i={r['iterations']}" if r['iterations'] != "?" else ""
        delay_str = f"d={r.get('delay', '?')}" if r.get('delay', '?') != "?" else ""
        condition = ", ".join(filter(None, [iter_str, delay_str]))
        
        if len(freqs) > 1:
            label = f"{r['target_label']} @{r['freq']}MHz ({condition})"
        else:
            label = f"{r['target_label']} ({condition})"
        
        if ontime_pcts:
            ax5.plot(valid_cells, ontime_pcts, 'o-', label=label, color=color, linewidth=2, markersize=5)
    
    ax5.axhline(y=100, color='green', linestyle=':', alpha=0.5, label='100%')
    ax5.set_xlabel('Cell Count', fontsize=11)
    ax5.set_ylabel('PUSCH1 On-Time %', fontsize=11)
    ax5.set_title('PUSCH1 On-Time % vs Cell Count', fontsize=11, pad=10)
    ax5.legend(loc='lower left', fontsize=8)
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(-5, 110)
    
    # ===== Plot 6: Summary Table =====
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    # Find best result
    best_result = max(all_results, key=lambda x: x['results']['summary']['capacity_500us'])
    best_cap = best_result['results']['summary']['capacity_500us']
    
    best_iter = best_result['iterations'] if best_result['iterations'] != "?" else "N/A"
    best_delay = best_result.get('delay', '?') if best_result.get('delay', '?') != "?" else "N/A"
    
    summary_text = f"""
╔═══════════════════════════════════════════════════════════╗
║         500μs VALIDATION COMPARISON SUMMARY               ║
╠═══════════════════════════════════════════════════════════╣
║                                                           ║
║  Deadline:          {deadline_us:>6} μs                            ║
║  Configurations:    {len(all_results):>6}                                 ║
║                                                           ║
╠═══════════════════════════════════════════════════════════╣
║                                                           ║
║  🏆 Best Configuration:                                   ║
║     Target:     {best_result['target_label']:<10}                            ║
║     Freq:       {str(best_result['freq']) + ' MHz':<10}                            ║
║     Iterations: {str(best_iter):<10}                            ║
║     Delay:      {str(best_delay) + ' μs':<10}                            ║
║     500μs Cap:  {best_cap:>3} cells                               ║
║                                                           ║
╠═══════════════════════════════════════════════════════════╣
║                                                           ║
║  All Results (sorted by capacity):                        ║"""
    
    for r in sorted(all_results, key=lambda x: -x['results']['summary']['capacity_500us']):
        cap = r['results']['summary']['capacity_500us']
        status = "✅" if r['results']['summary']['overall_pass'] else "❌"
        iter_val = r['iterations'] if r['iterations'] != "?" else "?"
        delay_val = r.get('delay', '?') if r.get('delay', '?') != "?" else "?"
        summary_text += f"\n║  {r['target_label']:<6} i={str(iter_val):<5} d={str(delay_val):<7}: {cap:>2} cells {status}   ║"
    
    summary_text += """
║                                                           ║
╚═══════════════════════════════════════════════════════════╝"""
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.93])
    
    output_file = get_unique_filename(f"{output_prefix}_500us_comparison", ".png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n📊 Comparison plot saved: {output_file}")
    plt.close()


def save_comparison_json(all_results, deadline_us, output_prefix):
    """Save comparison JSON report."""
    output = {
        "validation_timestamp": datetime.now().isoformat(),
        "deadline_us": deadline_us,
        "total_configurations": len(all_results),
        "summary": [],
        "detailed_results": []
    }
    
    for r in sorted(all_results, key=lambda x: (x['freq'] if isinstance(x['freq'], (int, float)) else 0, x['target_label'])):
        summary = r['results']['summary']
        rate = (summary['total_violations'] / summary['total_slots'] * 100) if summary['total_slots'] > 0 else 0
        
        output["summary"].append({
            "filename": r['filename'],
            "target_label": r['target_label'],
            "freq_mhz": r['freq'],
            "iterations": r['iterations'],
            "delay": r.get('delay', '?'),
            "capacity_500us": summary['capacity_500us'],
            "total_slots": summary['total_slots'],
            "total_violations": summary['total_violations'],
            "violation_rate_pct": round(rate, 4),
            "overall_pass": summary['overall_pass']
        })
        
        output["detailed_results"].append({
            "filename": r['filename'],
            "target_label": r['target_label'],
            "cells": r['results']['cells']
        })
    
    output_file = get_unique_filename(f"{output_prefix}_500us_comparison", ".json")
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"💾 Comparison JSON saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate measure.py results against 500us latency deadline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single file validation
    python3 validate_500us_latency.py 040_040_sweep_graphs_avg_F08.json
    python3 validate_500us_latency.py result.json --deadline 450 --plot
    
    # Multi-file comparison
    python3 validate_500us_latency.py \\
        040_040_sweep_graphs_avg_F08.json \\
        060_060_sweep_graphs_avg_F08.json \\
        066_066_sweep_graphs_avg_F08.json \\
        --compare --plot
    
    # Using glob pattern
    python3 validate_500us_latency.py *_sweep_*.json --compare --plot
        """
    )
    
    parser.add_argument('input_files', nargs='+', help='measure.py result JSON file(s)')
    parser.add_argument('--deadline', '-d', type=int, default=DEFAULT_DEADLINE_US,
                       help=f'Deadline in microseconds (default: {DEFAULT_DEADLINE_US})')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output prefix for reports (default: based on input filename)')
    parser.add_argument('--plot', '-p', action='store_true',
                       help='Generate visualization plots')
    parser.add_argument('--json', '-j', action='store_true',
                       help='Save detailed JSON report')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress detailed output, show only summary')
    parser.add_argument('--compare', '-c', action='store_true',
                       help='Enable multi-file comparison mode')
    
    args = parser.parse_args()
    
    # Expand glob patterns and collect all files
    all_files = []
    for pattern in args.input_files:
        expanded = glob.glob(pattern)
        if expanded:
            all_files.extend(expanded)
        elif os.path.exists(pattern):
            all_files.append(pattern)
    
    # Remove duplicates and sort
    all_files = sorted(set(all_files))
    
    if not all_files:
        print(f"❌ Error: No files found!")
        sys.exit(1)
    
    # Determine mode: single file or comparison
    if len(all_files) > 1 or args.compare:
        # ===== MULTI-FILE COMPARISON MODE =====
        print("=" * 70)
        print("500μs Latency Validation - Multi-File Comparison Mode")
        print("=" * 70)
        print(f"  Files to process: {len(all_files)}")
        print(f"  Deadline: {args.deadline} μs")
        print()
        
        all_results = process_multiple_files(all_files, args.deadline)
        
        if not all_results:
            print("❌ No valid results to compare!")
            sys.exit(1)
        
        # Print comparison summary
        print_comparison_summary(all_results, args.deadline)
        
        # Set output prefix for comparison
        if args.output:
            output_prefix = args.output
        else:
            output_prefix = "500us_validation"
        
        # Generate comparison plots
        if args.plot:
            plot_comparison(all_results, args.deadline, output_prefix)
        
        # Save comparison JSON
        if args.json:
            save_comparison_json(all_results, args.deadline, output_prefix)
        
        # Also generate individual reports if requested
        if args.plot or args.json:
            print("\n📄 Individual file reports:")
            for r in all_results:
                individual_prefix = os.path.splitext(r['filepath'])[0]
                if args.plot:
                    plot_results(r['results'], r['data'], individual_prefix)
                if args.json:
                    save_json_report(r['results'], r['data'], individual_prefix)
        
        # Exit with success if any configuration passes
        any_pass = any(r['results']['summary']['overall_pass'] for r in all_results)
        sys.exit(0 if any_pass else 1)
        
    else:
        # ===== SINGLE FILE MODE =====
        input_file = all_files[0]
        
        # Set output prefix
        if args.output:
            output_prefix = args.output
        else:
            output_prefix = os.path.splitext(input_file)[0]
        
        # Load and process
        print(f"📂 Loading: {input_file}")
        data = load_result_file(input_file)
        
        print(f"🔍 Extracting cell data...")
        cells = extract_cell_data(data)
        
        if not cells:
            print("❌ No cell data found in file!")
            sys.exit(1)
        
        print(f"   Found {len(cells)} cell configurations")
        
        # Validate
        print(f"⏱️  Validating against {args.deadline}μs deadline...")
        results = validate_latencies(cells, args.deadline)
        
        # Output
        if not args.quiet:
            print_report(results, data)
        else:
            summary = results["summary"]
            status = "PASS ✅" if summary["overall_pass"] else "FAIL ❌"
            print(f"\n{status} | 500μs Capacity: {summary['capacity_500us']} cells | "
                  f"Violations: {summary['total_violations']}/{summary['total_slots']}")
        
        # Save outputs
        if args.plot:
            plot_results(results, data, output_prefix)
        
        if args.json:
            save_json_report(results, data, output_prefix)
        
        # Exit code based on result
        sys.exit(0 if results["summary"]["overall_pass"] else 1)


if __name__ == "__main__":
    main()
