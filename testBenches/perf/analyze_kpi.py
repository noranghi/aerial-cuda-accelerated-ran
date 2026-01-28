#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
KPI Analysis Script for measure.py Results

This script analyzes measure.py JSON output and provides detailed statistics for:
1. Latency (DL/UL separated): Average, Max, P99
2. Throughput: Slots per Second (TPS)
3. Jitter: Standard Deviation of latencies

Usage:
    python3 analyze_kpi.py <result_json_file> [options]
    
Examples:
    python3 analyze_kpi.py 060_060_sweep_graphs_avg_F08.json
    python3 analyze_kpi.py 060_060_sweep_graphs_avg_F08.json --cell 8
    python3 analyze_kpi.py 060_060_sweep_graphs_avg_F08.json --export kpi_report.json
"""

import json
import argparse
import sys
import os
from datetime import datetime

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("Warning: numpy not available. Using basic statistics.")

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# TDD Pattern: DDDSUUDDDD (10 slots per pattern)
# D = Downlink, S = Special, U = Uplink
TDD_PATTERN_DDDSUUDDDD = {
    "pattern": "dddsuudddd",
    "slots_per_pattern": 10,
    "dl_slots": 7,  # D(0,1,2) + D(6,7,8,9)
    "ul_slots": 2,  # U(4,5)
    "special_slots": 1,  # S(3)
    "slot_duration_us": 500,  # 500us per slot (30kHz SCS)
    "pattern_duration_us": 5000  # 5ms per pattern
}

# Channel Classification
DL_CHANNELS = ['PDSCH', 'PDCCH', 'SSB', 'CSI-RS', 'DLBFW']
UL_CHANNELS = ['PUSCH1', 'PUSCH2', 'PUCCH1', 'PUCCH2', 'PRACH', 'SRS1', 'SRS2', 'ULBFW1', 'ULBFW2']

import re
import glob


# =============================================================================
# TX TIMING JITTER CALCULATION
# =============================================================================

def parse_buffer_for_tx_timing(buffer_filepath):
    """
    Parse buffer-XX.txt file to extract t_send values for TX Timing Jitter calculation.
    
    TX Timing Jitter measures the accuracy of slot start timing.
    - Expected: Each slot starts exactly 500μs apart
    - Deviation = actual_interval - 500μs
    - TX Jitter = std(deviations)
    
    Returns:
        dict: {
            "dl_t_send": {pattern_idx: [t_send values for slots 0-7]},
            "ul_t_send": {pattern_idx: [PUSCH1 t_send, PUSCH2 t_send]},
            "tx_timing_jitter": {...}
        }
    """
    if not os.path.exists(buffer_filepath):
        return None
    
    with open(buffer_filepath, 'r') as f:
        lines = f.readlines()
    
    # Parse t_send values
    dl_t_send_by_pattern = {}  # {pattern_idx: {slot_idx: t_send}}
    ul_t_send_by_pattern = {}  # {pattern_idx: {"PUSCH1": t_send, "PUSCH2": t_send}}
    
    current_pattern = 0
    
    for i, line in enumerate(lines):
        parts = line.split()
        if not parts:
            continue
        
        # Detect pattern boundaries (look for "Slot pattern" or similar)
        if "Slot pattern" in line or (len(parts) >= 4 and parts[0] == "Slot" and parts[1] == "pattern"):
            current_pattern += 1
            if current_pattern not in dl_t_send_by_pattern:
                dl_t_send_by_pattern[current_pattern] = {}
            if current_pattern not in ul_t_send_by_pattern:
                ul_t_send_by_pattern[current_pattern] = {}
            continue
        
        # Initialize pattern 0 if not exists
        if current_pattern == 0:
            current_pattern = 1
            dl_t_send_by_pattern[1] = {}
            ul_t_send_by_pattern[1] = {}
        
        # DL: "Slot # X: average PDSCH run time: YYY us from ZZZ"
        if len(parts) > 10 and parts[0] == "Slot" and parts[1] == "#" and "PDSCH" in line:
            try:
                slot_idx = int(parts[2].rstrip(':'))
                # t_send is after "from"
                from_idx = parts.index("from")
                t_send = float(parts[from_idx + 1])
                dl_t_send_by_pattern[current_pattern][slot_idx] = t_send
            except (ValueError, IndexError):
                continue
        
        # UL: "Average PUSCH run time: YYY us from ZZZ"
        if len(parts) > 7 and parts[0] == "Average" and "PUSCH" in parts[1]:
            try:
                channel = parts[1]
                from_idx = parts.index("from")
                t_send = float(parts[from_idx + 1])
                
                if channel == "PUSCH":
                    ul_t_send_by_pattern[current_pattern]["PUSCH1"] = t_send
                elif channel == "PUSCH2":
                    ul_t_send_by_pattern[current_pattern]["PUSCH2"] = t_send
            except (ValueError, IndexError):
                continue
    
    # Calculate TX Timing Jitter
    tx_jitter_results = calculate_tx_timing_jitter(dl_t_send_by_pattern, ul_t_send_by_pattern)
    
    return {
        "dl_t_send_by_pattern": dl_t_send_by_pattern,
        "ul_t_send_by_pattern": ul_t_send_by_pattern,
        "tx_timing_jitter": tx_jitter_results,
        "pattern_count": current_pattern
    }


def calculate_tx_timing_jitter(dl_t_send_by_pattern, ul_t_send_by_pattern):
    """
    Calculate TX Timing Jitter from t_send values.
    
    TX Timing Jitter = std(actual_slot_interval - theoretical_slot_interval)
    
    For 5G NR with 30kHz SCS: theoretical_slot_interval = 500 μs
    """
    SLOT_DURATION_US = 500.0  # Theoretical slot duration
    
    results = {
        "dl": {
            "slot_intervals": [],  # Actual intervals between consecutive slots
            "deviations": [],      # Deviation from 500μs
            "jitter_std_us": 0,
            "jitter_max_us": 0,
            "jitter_p99_us": 0,
            "mean_interval_us": 0,
        },
        "combined": {}
    }
    
    all_dl_intervals = []
    all_dl_deviations = []
    
    # Calculate DL slot-to-slot intervals
    for pattern_idx, slots in dl_t_send_by_pattern.items():
        if len(slots) < 2:
            continue
        
        # Sort by slot index
        sorted_slots = sorted(slots.items())
        
        for i in range(1, len(sorted_slots)):
            prev_slot, prev_t_send = sorted_slots[i - 1]
            curr_slot, curr_t_send = sorted_slots[i]
            
            # Calculate actual interval
            actual_interval = curr_t_send - prev_t_send
            
            # Expected interval (considering slot index difference)
            slot_diff = curr_slot - prev_slot
            expected_interval = slot_diff * SLOT_DURATION_US
            
            # Deviation from expected
            deviation = actual_interval - expected_interval
            
            all_dl_intervals.append(actual_interval)
            all_dl_deviations.append(deviation)
    
    # Calculate statistics for DL
    if all_dl_deviations:
        if HAS_NUMPY:
            arr = np.array(all_dl_deviations)
            results["dl"]["jitter_std_us"] = round(float(np.std(arr)), 2)
            results["dl"]["jitter_max_us"] = round(float(np.max(np.abs(arr))), 2)
            results["dl"]["jitter_p99_us"] = round(float(np.percentile(np.abs(arr), 99)), 2)
            results["dl"]["mean_interval_us"] = round(float(np.mean(all_dl_intervals)), 2)
            results["dl"]["deviations"] = [round(d, 2) for d in all_dl_deviations[:20]]  # Sample
            results["dl"]["slot_intervals"] = [round(i, 2) for i in all_dl_intervals[:20]]
            results["dl"]["count"] = len(all_dl_deviations)
        else:
            n = len(all_dl_deviations)
            mean_dev = sum(all_dl_deviations) / n
            variance = sum((d - mean_dev) ** 2 for d in all_dl_deviations) / n
            results["dl"]["jitter_std_us"] = round(variance ** 0.5, 2)
            results["dl"]["jitter_max_us"] = round(max(abs(d) for d in all_dl_deviations), 2)
            sorted_abs = sorted(abs(d) for d in all_dl_deviations)
            results["dl"]["jitter_p99_us"] = round(sorted_abs[min(int(n * 0.99), n - 1)], 2)
            results["dl"]["mean_interval_us"] = round(sum(all_dl_intervals) / len(all_dl_intervals), 2)
            results["dl"]["count"] = n
    
    # Combined results
    results["combined"] = results["dl"].copy()
    results["theoretical_interval_us"] = SLOT_DURATION_US
    
    return results


def print_tx_timing_jitter_report(tx_jitter_data, cell_count=None):
    """Print TX Timing Jitter analysis report."""
    
    if not tx_jitter_data:
        print("\n⚠️ No TX Timing data available. Buffer file required.")
        return
    
    tx_jitter = tx_jitter_data.get("tx_timing_jitter", {})
    
    print("\n" + "=" * 100)
    print("                    TX TIMING JITTER ANALYSIS (교수님 정의)")
    print("=" * 100)
    print("""
    TX Timing Jitter = 슬롯 시작 타이밍의 불규칙성 (표준 편차)
    
    - 이론적 슬롯 간격: 500 μs (30kHz SCS)
    - Deviation = (실제 간격) - (이론적 간격)
    - TX Jitter = std(Deviation)
    
    ✅ "결과는 늦게 나와도 되지만, 시작 타이밍은 흔들리면 안 된다"
""")
    
    if cell_count:
        print(f"    Cell Count: {cell_count}")
    print(f"    Patterns Analyzed: {tx_jitter_data.get('pattern_count', 'N/A')}")
    
    print("\n┌─ TX TIMING JITTER RESULTS ───────────────────────────────────────────────────────────────────┐")
    print(f"│  {'Metric':<35} {'Value':>20} {'Unit':>10}  │")
    print(f"│  {'-'*35} {'-'*20} {'-'*10}  │")
    
    dl_jitter = tx_jitter.get("dl", {})
    
    print(f"│  {'Theoretical Slot Interval':<35} {tx_jitter.get('theoretical_interval_us', 500):>20.2f} {'μs':>10}  │")
    print(f"│  {'Mean Actual Interval':<35} {dl_jitter.get('mean_interval_us', 0):>20.2f} {'μs':>10}  │")
    print(f"│  {'-'*35} {'-'*20} {'-'*10}  │")
    print(f"│  {'TX Jitter (Std Dev)':<35} {dl_jitter.get('jitter_std_us', 0):>20.2f} {'μs':>10}  │")
    print(f"│  {'TX Jitter (Max Deviation)':<35} {dl_jitter.get('jitter_max_us', 0):>20.2f} {'μs':>10}  │")
    print(f"│  {'TX Jitter (P99 Deviation)':<35} {dl_jitter.get('jitter_p99_us', 0):>20.2f} {'μs':>10}  │")
    print(f"│  {'Sample Count':<35} {dl_jitter.get('count', 0):>20} {'':>10}  │")
    print("└──────────────────────────────────────────────────────────────────────────────────────────────┘")
    
    # Evaluation
    jitter_std = dl_jitter.get('jitter_std_us', 0)
    print("\n┌─ EVALUATION ────────────────────────────────────────────────────────────────────────────────┐")
    if jitter_std < 1:
        print(f"│  ✅ EXCELLENT: TX Jitter = {jitter_std:.2f} μs (< 1 μs)                                          │")
        print(f"│     슬롯 시작 타이밍이 매우 정확합니다.                                                    │")
    elif jitter_std < 5:
        print(f"│  ✅ GOOD: TX Jitter = {jitter_std:.2f} μs (< 5 μs)                                               │")
        print(f"│     슬롯 시작 타이밍이 안정적입니다.                                                        │")
    elif jitter_std < 10:
        print(f"│  ⚠️ ACCEPTABLE: TX Jitter = {jitter_std:.2f} μs (< 10 μs)                                        │")
        print(f"│     슬롯 시작 타이밍에 약간의 변동이 있습니다.                                              │")
    else:
        print(f"│  ❌ POOR: TX Jitter = {jitter_std:.2f} μs (>= 10 μs)                                             │")
        print(f"│     슬롯 시작 타이밍이 불안정합니다. 개선이 필요합니다.                                      │")
    print("└──────────────────────────────────────────────────────────────────────────────────────────────┘")
    
    # Sample deviations
    deviations = dl_jitter.get('deviations', [])
    intervals = dl_jitter.get('slot_intervals', [])
    if deviations:
        print("\n┌─ SAMPLE TX TIMING DATA ─────────────────────────────────────────────────────────────────────┐")
        print(f"│  {'#':>4}  {'Actual Interval (μs)':>22}  {'Expected (μs)':>15}  {'Deviation (μs)':>18}  │")
        print(f"│  {'-'*4}  {'-'*22}  {'-'*15}  {'-'*18}  │")
        for i, (interval, dev) in enumerate(zip(intervals[:10], deviations[:10])):
            expected = 500.0
            dev_mark = "✅" if abs(dev) < 5 else "⚠️" if abs(dev) < 10 else "❌"
            print(f"│  {i+1:>4}  {interval:>22.2f}  {expected:>15.2f}  {dev:>15.2f} {dev_mark}  │")
        print("└──────────────────────────────────────────────────────────────────────────────────────────────┘")
    
    print("\n" + "=" * 100)


def analyze_all_buffers(buffer_dir, output_prefix=None):
    """Analyze all buffer files in a directory and generate TX Timing Jitter report."""
    
    buffer_files = sorted(glob.glob(os.path.join(buffer_dir, "buffer-*.txt")))
    
    if not buffer_files:
        print(f"⚠️ No buffer files found in: {buffer_dir}")
        return None
    
    all_results = {}
    
    for buffer_file in buffer_files:
        # Extract cell count from filename (e.g., buffer-08.txt -> 8)
        basename = os.path.basename(buffer_file)
        match = re.search(r'buffer-(\d+)\.txt', basename)
        if match:
            cell_count = int(match.group(1))
        else:
            continue
        
        print(f"📂 Parsing: {basename} (Cell {cell_count})")
        result = parse_buffer_for_tx_timing(buffer_file)
        
        if result:
            all_results[cell_count] = result
    
    return all_results


def generate_tx_jitter_summary_table(all_tx_results):
    """Generate summary table for TX Timing Jitter across all cells."""
    
    if not all_tx_results:
        return
    
    print("\n" + "=" * 100)
    print("                    TX TIMING JITTER SUMMARY BY CELL COUNT")
    print("=" * 100)
    print(f"\n{'Cell':>6} {'Patterns':>10} {'Mean Int(μs)':>14} {'Jitter Std(μs)':>16} {'Jitter Max(μs)':>16} {'Jitter P99(μs)':>16} {'Status':>10}")
    print("-" * 100)
    
    for cell in sorted(all_tx_results.keys()):
        data = all_tx_results[cell]
        tx_jitter = data.get("tx_timing_jitter", {}).get("dl", {})
        
        patterns = data.get("pattern_count", 0)
        mean_int = tx_jitter.get("mean_interval_us", 0)
        jitter_std = tx_jitter.get("jitter_std_us", 0)
        jitter_max = tx_jitter.get("jitter_max_us", 0)
        jitter_p99 = tx_jitter.get("jitter_p99_us", 0)
        
        # Status evaluation
        if jitter_std < 1:
            status = "✅ EXCELLENT"
        elif jitter_std < 5:
            status = "✅ GOOD"
        elif jitter_std < 10:
            status = "⚠️ ACCEPTABLE"
        else:
            status = "❌ POOR"
        
        print(f"{cell:>6} {patterns:>10} {mean_int:>14.2f} {jitter_std:>16.2f} {jitter_max:>16.2f} {jitter_p99:>16.2f} {status:>10}")
    
    print("=" * 100)


def generate_tx_jitter_graph(all_tx_results, output_prefix, config=None):
    """
    Generate TX Timing Jitter analysis graph with detailed report.
    
    Creates a comprehensive PNG with:
    1. TX Jitter vs Cell Count
    2. Slot Interval Distribution
    3. Deviation Histogram
    4. Analysis Summary Table
    """
    
    if not HAS_MATPLOTLIB:
        print("⚠️ matplotlib not available. Skipping TX Jitter graph generation.")
        return
    
    if not all_tx_results:
        print("⚠️ No TX Timing data available for graph generation.")
        return
    
    cells = sorted(all_tx_results.keys())
    
    # Extract data
    jitter_std = []
    jitter_max = []
    jitter_p99 = []
    mean_intervals = []
    all_deviations = []
    all_intervals = []
    
    for cell in cells:
        data = all_tx_results[cell]
        tx = data.get("tx_timing_jitter", {}).get("dl", {})
        jitter_std.append(tx.get("jitter_std_us", 0))
        jitter_max.append(tx.get("jitter_max_us", 0))
        jitter_p99.append(tx.get("jitter_p99_us", 0))
        mean_intervals.append(tx.get("mean_interval_us", 500))
        
        # Collect sample deviations for histogram
        devs = tx.get("deviations", [])
        all_deviations.extend(devs)
        ints = tx.get("slot_intervals", [])
        all_intervals.extend(ints)
    
    # Create figure - 2x2 grid for 4 graphs only (no text box overlap issue)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.subplots_adjust(hspace=0.3, wspace=0.25)
    
    # Colors
    colors = {
        'primary': '#2E86AB',
        'secondary': '#A23B72',
        'accent': '#F18F01',
        'good': '#28965A',
        'bad': '#C73E1D'
    }
    
    # ============ Plot 1: TX Jitter vs Cell Count ============
    ax1 = axes[0, 0]
    
    ax1.plot(cells, jitter_std, 'o-', color=colors['primary'], linewidth=2.5, 
             markersize=8, label='TX Jitter (Std Dev)', zorder=3)
    ax1.fill_between(cells, 0, jitter_std, alpha=0.2, color=colors['primary'])
    
    # Threshold lines
    ax1.axhline(y=1, color=colors['good'], linestyle='--', linewidth=1.5, 
                alpha=0.7, label='Excellent (< 1 us)')
    ax1.axhline(y=5, color=colors['accent'], linestyle='--', linewidth=1.5, 
                alpha=0.7, label='Good (< 5 us)')
    ax1.axhline(y=10, color=colors['bad'], linestyle='--', linewidth=1.5, 
                alpha=0.7, label='Acceptable (< 10 us)')
    
    ax1.set_xlabel('Cell Count', fontsize=11, fontweight='bold')
    ax1.set_ylabel('TX Jitter (us)', fontsize=11, fontweight='bold')
    ax1.set_title('TX Timing Jitter vs Cell Count', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(cells)
    ax1.set_ylim(bottom=0)
    
    # ============ Plot 2: Mean Interval vs Theoretical ============
    ax2 = axes[0, 1]
    
    theoretical = [500] * len(cells)
    
    ax2.bar(cells, mean_intervals, color=colors['primary'], alpha=0.7, label='Actual Mean Interval')
    ax2.plot(cells, theoretical, 'r--', linewidth=2, label='Theoretical (500 us)')
    
    # Add deviation annotations
    for i, (cell, mean_int) in enumerate(zip(cells, mean_intervals)):
        diff = mean_int - 500
        color = colors['good'] if abs(diff) < 1 else colors['accent'] if abs(diff) < 5 else colors['bad']
        ax2.annotate(f'{diff:+.2f}', xy=(cell, mean_int), xytext=(0, 5),
                     textcoords='offset points', ha='center', fontsize=8, color=color)
    
    ax2.set_xlabel('Cell Count', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Slot Interval (us)', fontsize=11, fontweight='bold')
    ax2.set_title('Mean Slot Interval vs Theoretical (500 us)', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xticks(cells)
    
    # Set y-axis to show small deviations clearly
    if mean_intervals:
        y_min = min(min(mean_intervals), 500) - 5
        y_max = max(max(mean_intervals), 500) + 5
        ax2.set_ylim(y_min, y_max)
    
    # ============ Plot 3: Jitter Metrics Comparison ============
    ax3 = axes[1, 0]
    
    x = np.arange(len(cells))
    width = 0.25
    
    ax3.bar(x - width, jitter_std, width, label='Std Dev', color=colors['primary'], alpha=0.8)
    ax3.bar(x, jitter_p99, width, label='P99', color=colors['secondary'], alpha=0.8)
    ax3.bar(x + width, jitter_max, width, label='Max', color=colors['accent'], alpha=0.8)
    
    ax3.set_xlabel('Cell Count', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Deviation (us)', fontsize=11, fontweight='bold')
    ax3.set_title('TX Jitter Metrics Comparison (Std / P99 / Max)', fontsize=13, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(cells)
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # ============ Plot 4: Summary Table (as plot) ============
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Create table data
    table_data = []
    table_colors = []
    
    for i, cell in enumerate(cells):
        std = jitter_std[i]
        max_j = jitter_max[i]
        p99 = jitter_p99[i]
        mean_int = mean_intervals[i]
        
        if std < 1:
            status = "EXCELLENT"
            row_color = '#d4edda'  # Light green
        elif std < 5:
            status = "GOOD"
            row_color = '#fff3cd'  # Light yellow
        elif std < 10:
            status = "ACCEPTABLE"
            row_color = '#ffe5d0'  # Light orange
        else:
            status = "POOR"
            row_color = '#f8d7da'  # Light red
        
        table_data.append([
            f'{cell}',
            f'{std:.2f}',
            f'{max_j:.2f}',
            f'{p99:.2f}',
            f'{mean_int:.2f}',
            status
        ])
        table_colors.append([row_color] * 6)
    
    # Create table
    col_labels = ['Cell', 'Std Dev\n(us)', 'Max\n(us)', 'P99\n(us)', 'Mean Int\n(us)', 'Status']
    
    table = ax4.table(
        cellText=table_data,
        colLabels=col_labels,
        cellColours=table_colors,
        colColours=['#e9ecef'] * 6,
        loc='center',
        cellLoc='center'
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Style header
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight='bold')
            cell.set_facecolor('#343a40')
            cell.set_text_props(color='white', fontweight='bold')
    
    ax4.set_title('TX Jitter Results by Cell Count', fontsize=13, fontweight='bold', pad=20)
    
    # Main title
    if config:
        target = config.get('target', '?')
        if isinstance(target, list):
            target_str = f"{target[0]}/{target[1]}"
        else:
            target_str = str(target)
        title = f'TX Timing Jitter Analysis - Target SM: {target_str}, Freq: {config.get("freq", "?")} MHz'
    else:
        title = 'TX Timing Jitter Analysis'
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    
    # Save figure
    graph_path = f"{output_prefix}_tx_jitter_analysis.png"
    plt.savefig(graph_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"TX Jitter analysis graph saved to: {graph_path}")
    
    # Also generate separate detailed summary graph
    generate_tx_jitter_summary_graph(all_tx_results, output_prefix, config, cells, 
                                      jitter_std, jitter_max, jitter_p99, mean_intervals)
    
    return graph_path


def generate_tx_jitter_summary_graph(all_tx_results, output_prefix, config, cells, 
                                      jitter_std, jitter_max, jitter_p99, mean_intervals):
    """Generate separate detailed summary graph with explanation."""
    
    if not HAS_MATPLOTLIB:
        return
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('off')
    
    # Build summary text (English to avoid font issues)
    lines = []
    lines.append("=" * 90)
    lines.append("                    TX TIMING JITTER ANALYSIS SUMMARY")
    lines.append("=" * 90)
    lines.append("")
    lines.append("  [Definition - Professor's TX Jitter]")
    lines.append("    - TX Jitter = Standard deviation of slot start timing irregularity")
    lines.append("    - 'Results can be late, but TX start timing must NOT fluctuate'")
    lines.append("")
    lines.append("  [Calculation Method]")
    lines.append("    - Theoretical slot interval: 500 us (30kHz SCS, 5G NR)")
    lines.append("    - Deviation = (Actual slot interval) - 500 us")
    lines.append("    - TX Jitter = std(Deviation)")
    lines.append("")
    lines.append("  [Evaluation Criteria]")
    lines.append("    - EXCELLENT : < 1 us   (Very accurate slot start timing)")
    lines.append("    - GOOD      : < 5 us   (Stable timing)")
    lines.append("    - ACCEPTABLE: < 10 us  (Minor variations)")
    lines.append("    - POOR      : >= 10 us (Improvement needed)")
    lines.append("")
    lines.append("=" * 90)
    lines.append("  TX JITTER RESULTS BY CELL COUNT")
    lines.append("=" * 90)
    lines.append("")
    lines.append(f"  {'Cell':>6} | {'Std Dev':>10} | {'Max':>10} | {'P99':>10} | {'Mean Int':>12} | {'Status':<12}")
    lines.append(f"  {'-'*6} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*12} | {'-'*12}")
    
    for i, cell in enumerate(cells):
        std = jitter_std[i]
        max_j = jitter_max[i]
        p99 = jitter_p99[i]
        mean_int = mean_intervals[i]
        
        if std < 1:
            status = "[OK] EXCELLENT"
        elif std < 5:
            status = "[OK] GOOD"
        elif std < 10:
            status = "[--] ACCEPTABLE"
        else:
            status = "[!!] POOR"
        
        lines.append(f"  {cell:>6} | {std:>8.2f} us | {max_j:>8.2f} us | {p99:>8.2f} us | {mean_int:>10.2f} us | {status:<12}")
    
    lines.append("")
    lines.append("=" * 90)
    
    # Find best cell
    best_idx = jitter_std.index(min(jitter_std))
    worst_idx = jitter_std.index(max(jitter_std))
    
    lines.append(f"  Best Cell  : Cell {cells[best_idx]} (TX Jitter: {jitter_std[best_idx]:.2f} us)")
    lines.append(f"  Worst Cell : Cell {cells[worst_idx]} (TX Jitter: {jitter_std[worst_idx]:.2f} us)")
    lines.append(f"  Average    : {sum(jitter_std)/len(jitter_std):.2f} us")
    lines.append("")
    lines.append("=" * 90)
    
    # Join all lines
    summary_text = '\n'.join(lines)
    
    # Display text
    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='left',
            fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='#343a40', linewidth=2))
    
    # Title
    if config:
        target = config.get('target', '?')
        if isinstance(target, list):
            target_str = f"{target[0]}/{target[1]}"
        else:
            target_str = str(target)
        title = f'TX Timing Jitter Summary Report - SM: {target_str}'
    else:
        title = 'TX Timing Jitter Summary Report'
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    
    # Save
    summary_path = f"{output_prefix}_tx_jitter_summary.png"
    plt.savefig(summary_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"TX Jitter summary report saved to: {summary_path}")


# =============================================================================
# ORIGINAL FUNCTIONS
# =============================================================================

def calculate_statistics(latencies):
    """Calculate comprehensive statistics for latency array."""
    if not latencies:
        return None
    
    if HAS_NUMPY:
        arr = np.array(latencies)
        stats = {
            "count": len(arr),
            "avg_us": round(np.mean(arr), 2),
            "std_us": round(np.std(arr), 2),  # Jitter = Standard Deviation
            "min_us": round(np.min(arr), 2),
            "max_us": round(np.max(arr), 2),
            "p50_us": round(np.percentile(arr, 50), 2),
            "p90_us": round(np.percentile(arr, 90), 2),
            "p95_us": round(np.percentile(arr, 95), 2),
            "p99_us": round(np.percentile(arr, 99), 2),
            "variance_us2": round(np.var(arr), 2),
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
            "variance_us2": round(variance, 2),
        }
    
    # Jitter coefficient (CV = std/avg)
    if stats["avg_us"] > 0:
        stats["jitter_cv_pct"] = round((stats["std_us"] / stats["avg_us"]) * 100, 2)
    else:
        stats["jitter_cv_pct"] = 0
    
    return stats


def load_result_file(filepath):
    """Load measure.py result JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def extract_test_config(data):
    """Extract test configuration from result JSON."""
    config = data.get("testConfig", {})
    return {
        "target": config.get("target", "?"),
        "freq": config.get("freq", "?"),
        "iterations": config.get("iterations", 1),
        "sweeps": config.get("sweeps", 1),  # Number of slots (patterns)
        "delay": config.get("delay", 0),
        "pattern": config.get("pattern", "dddsuudddd"),
        "gpu": config.get("gpu", 0),
    }


def extract_cell_results(data):
    """Extract cell-level results from JSON."""
    cells = {}
    
    for key, value in data.items():
        if key == "testConfig" or key == "constraints" or not isinstance(value, dict):
            continue
        
        try:
            cell_count = int(key.split("+")[0])
        except:
            continue
        
        cell_data = {
            "key": key,
            "cell_count": cell_count,
            "structure": value.get("Structure", False),
            "mode": value.get("Mode", "Unknown"),
            "dl_latencies": {},
            "ul_latencies": {},
            "all_latencies": {},
            "memory_mb": value.get("memoryUseMB", {})
        }
        
        # Extract latencies by channel
        for channel in DL_CHANNELS + UL_CHANNELS:
            latencies = value.get(channel, [])
            if latencies:
                cell_data["all_latencies"][channel] = latencies
                if channel in DL_CHANNELS:
                    cell_data["dl_latencies"][channel] = latencies
                else:
                    cell_data["ul_latencies"][channel] = latencies
        
        # Extract ontime percent
        ontime = value.get("ontimePercent", {})
        cell_data["ontime"] = ontime
        
        cells[cell_count] = cell_data
    
    return cells


def analyze_cell_kpi(cell_data, config):
    """Analyze KPIs for a single cell configuration."""
    kpi = {
        "cell_count": cell_data["cell_count"],
        "structure_valid": cell_data["structure"],
        "mode": cell_data["mode"],
        "latency": {
            "dl": {},
            "ul": {},
            "combined": {}
        },
        "throughput": {},
        "jitter": {}
    }
    
    # 1. Latency Statistics
    # DL Channels
    all_dl_latencies = []
    for channel, latencies in cell_data["dl_latencies"].items():
        stats = calculate_statistics(latencies)
        if stats:
            kpi["latency"]["dl"][channel] = stats
            all_dl_latencies.extend(latencies)
    
    if all_dl_latencies:
        kpi["latency"]["dl"]["COMBINED"] = calculate_statistics(all_dl_latencies)
    
    # UL Channels
    all_ul_latencies = []
    for channel, latencies in cell_data["ul_latencies"].items():
        stats = calculate_statistics(latencies)
        if stats:
            kpi["latency"]["ul"][channel] = stats
            all_ul_latencies.extend(latencies)
    
    if all_ul_latencies:
        kpi["latency"]["ul"]["COMBINED"] = calculate_statistics(all_ul_latencies)
    
    # Combined (all channels)
    all_latencies = all_dl_latencies + all_ul_latencies
    if all_latencies:
        kpi["latency"]["combined"]["ALL"] = calculate_statistics(all_latencies)
    
    # 2. Throughput Calculation
    sweeps = config.get("sweeps", 1)
    iterations = config.get("iterations", 1)
    pattern = config.get("pattern", "dddsuudddd")
    
    if pattern == "dddsuudddd":
        slots_per_pattern = TDD_PATTERN_DDDSUUDDDD["slots_per_pattern"]
        pattern_duration_us = TDD_PATTERN_DDDSUUDDDD["pattern_duration_us"]
    elif pattern == "dddsu":
        slots_per_pattern = 5
        pattern_duration_us = 2500
    else:
        slots_per_pattern = 10
        pattern_duration_us = 5000
    
    total_patterns = sweeps * iterations
    total_slots = total_patterns * slots_per_pattern
    total_time_us = total_patterns * pattern_duration_us
    total_time_sec = total_time_us / 1_000_000
    
    # Slots per Second (TPS)
    if total_time_sec > 0:
        slots_per_second = total_slots / total_time_sec
        patterns_per_second = total_patterns / total_time_sec
    else:
        slots_per_second = 0
        patterns_per_second = 0
    
    kpi["throughput"] = {
        "total_patterns": total_patterns,
        "total_slots": total_slots,
        "total_time_us": total_time_us,
        "total_time_sec": round(total_time_sec, 4),
        "slots_per_second": round(slots_per_second, 2),
        "patterns_per_second": round(patterns_per_second, 2),
        "cell_slots_per_second": round(slots_per_second * cell_data["cell_count"], 2),
        "theoretical_slot_duration_us": 500,
        "theoretical_slots_per_second": 2000,  # 1000000 / 500 = 2000 slots/sec
    }
    
    # 3. Jitter Analysis
    jitter = {}
    
    # DL Jitter
    if "COMBINED" in kpi["latency"]["dl"]:
        dl_stats = kpi["latency"]["dl"]["COMBINED"]
        jitter["dl"] = {
            "std_us": dl_stats["std_us"],
            "cv_pct": dl_stats["jitter_cv_pct"],
            "max_deviation_us": round(dl_stats["max_us"] - dl_stats["avg_us"], 2),
            "p99_deviation_us": round(dl_stats["p99_us"] - dl_stats["avg_us"], 2),
        }
    
    # UL Jitter
    if "COMBINED" in kpi["latency"]["ul"]:
        ul_stats = kpi["latency"]["ul"]["COMBINED"]
        jitter["ul"] = {
            "std_us": ul_stats["std_us"],
            "cv_pct": ul_stats["jitter_cv_pct"],
            "max_deviation_us": round(ul_stats["max_us"] - ul_stats["avg_us"], 2),
            "p99_deviation_us": round(ul_stats["p99_us"] - ul_stats["avg_us"], 2),
        }
    
    # Combined Jitter
    if "ALL" in kpi["latency"]["combined"]:
        all_stats = kpi["latency"]["combined"]["ALL"]
        jitter["combined"] = {
            "std_us": all_stats["std_us"],
            "cv_pct": all_stats["jitter_cv_pct"],
            "max_deviation_us": round(all_stats["max_us"] - all_stats["avg_us"], 2),
            "p99_deviation_us": round(all_stats["p99_us"] - all_stats["avg_us"], 2),
        }
    
    kpi["jitter"] = jitter
    
    # Memory
    kpi["memory_mb"] = cell_data.get("memory_mb", {})
    
    # On-time percent
    kpi["ontime"] = cell_data.get("ontime", {})
    
    return kpi


def print_kpi_report(kpi, config, cell_count=None):
    """Print formatted KPI report."""
    
    print("\n" + "=" * 100)
    print("                              KPI ANALYSIS REPORT")
    print("=" * 100)
    
    # Test Configuration - convert list to string if necessary
    target = config.get('target', '?')
    if isinstance(target, list):
        target = ','.join(map(str, target))
    
    print("\n┌─ TEST CONFIGURATION ─────────────────────────────────────────────────────────────────────────┐")
    print(f"│  Target SM:        {str(target):<20}  │  Frequency:      {config.get('freq', '?')} MHz")
    print(f"│  Iterations:       {config.get('iterations', '?'):<20}  │  Slots/Sweeps:   {config.get('sweeps', '?')}")
    print(f"│  TDD Pattern:      {config.get('pattern', '?'):<20}  │  Delay:          {config.get('delay', '?')} μs")
    print("└──────────────────────────────────────────────────────────────────────────────────────────────┘")
    
    # Process each cell or specified cell
    cells_to_process = [cell_count] if cell_count else sorted(kpi.keys())
    
    for cell in cells_to_process:
        if cell not in kpi:
            print(f"\n⚠️  Cell {cell} not found in results")
            continue
        
        cell_kpi = kpi[cell]
        
        print(f"\n╔══════════════════════════════════════════════════════════════════════════════════════════════╗")
        print(f"║                                  CELL COUNT: {cell:02d}                                            ║")
        print(f"╠══════════════════════════════════════════════════════════════════════════════════════════════╣")
        
        # 1. LATENCY
        print(f"║  📊 LATENCY (μs)                                                                             ║")
        print(f"╠──────────────────────────────────────────────────────────────────────────────────────────────╣")
        print(f"║  {'Channel':<15} {'Count':>8} {'Avg':>10} {'Max':>10} {'P99':>10} {'P95':>10} {'Std':>10}   ║")
        print(f"║  {'-'*15} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}   ║")
        
        # DL Channels
        print(f"║  {'[DOWNLINK]':<15}                                                                        ║")
        dl_lat = cell_kpi["latency"]["dl"]
        for ch in sorted(dl_lat.keys()):
            if ch == "COMBINED":
                continue
            s = dl_lat[ch]
            print(f"║    {ch:<13} {s['count']:>8} {s['avg_us']:>10.2f} {s['max_us']:>10.2f} {s['p99_us']:>10.2f} {s['p95_us']:>10.2f} {s['std_us']:>10.2f}   ║")
        
        if "COMBINED" in dl_lat:
            s = dl_lat["COMBINED"]
            print(f"║  {'DL_TOTAL':<15} {s['count']:>8} {s['avg_us']:>10.2f} {s['max_us']:>10.2f} {s['p99_us']:>10.2f} {s['p95_us']:>10.2f} {s['std_us']:>10.2f}   ║")
        
        # UL Channels
        print(f"║  {'[UPLINK]':<15}                                                                          ║")
        ul_lat = cell_kpi["latency"]["ul"]
        for ch in sorted(ul_lat.keys()):
            if ch == "COMBINED":
                continue
            s = ul_lat[ch]
            print(f"║    {ch:<13} {s['count']:>8} {s['avg_us']:>10.2f} {s['max_us']:>10.2f} {s['p99_us']:>10.2f} {s['p95_us']:>10.2f} {s['std_us']:>10.2f}   ║")
        
        if "COMBINED" in ul_lat:
            s = ul_lat["COMBINED"]
            print(f"║  {'UL_TOTAL':<15} {s['count']:>8} {s['avg_us']:>10.2f} {s['max_us']:>10.2f} {s['p99_us']:>10.2f} {s['p95_us']:>10.2f} {s['std_us']:>10.2f}   ║")
        
        # Combined
        if "ALL" in cell_kpi["latency"]["combined"]:
            s = cell_kpi["latency"]["combined"]["ALL"]
            print(f"║  {'-'*15} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}   ║")
            print(f"║  {'ALL_COMBINED':<15} {s['count']:>8} {s['avg_us']:>10.2f} {s['max_us']:>10.2f} {s['p99_us']:>10.2f} {s['p95_us']:>10.2f} {s['std_us']:>10.2f}   ║")
        
        # 2. THROUGHPUT
        print(f"╠══════════════════════════════════════════════════════════════════════════════════════════════╣")
        print(f"║  ⚡ THROUGHPUT                                                                               ║")
        print(f"╠──────────────────────────────────────────────────────────────────────────────────────────────╣")
        tp = cell_kpi["throughput"]
        print(f"║  Total Patterns:       {tp['total_patterns']:>12}      │  Total Slots:        {tp['total_slots']:>12}      ║")
        print(f"║  Total Time:           {tp['total_time_sec']:>10.4f} sec  │  Slot Duration:      {tp['theoretical_slot_duration_us']:>10} μs      ║")
        print(f"║  ─────────────────────────────────────────────────────────────────────────────────────────   ║")
        print(f"║  📈 Slots/Second (TPS):              {tp['slots_per_second']:>12.2f}  (Theoretical: {tp['theoretical_slots_per_second']} slots/sec)   ║")
        print(f"║  📈 Patterns/Second:                 {tp['patterns_per_second']:>12.2f}                                         ║")
        print(f"║  📈 Cell-Slots/Second ({cell} cells):   {tp['cell_slots_per_second']:>12.2f}                                         ║")
        
        # 3. JITTER
        print(f"╠══════════════════════════════════════════════════════════════════════════════════════════════╣")
        print(f"║  📉 JITTER (Latency Variability)                                                             ║")
        print(f"╠──────────────────────────────────────────────────────────────────────────────────────────────╣")
        print(f"║  {'Direction':<12} {'Std Dev (μs)':>15} {'CV (%)':>12} {'Max Dev (μs)':>15} {'P99 Dev (μs)':>15} ║")
        print(f"║  {'-'*12} {'-'*15} {'-'*12} {'-'*15} {'-'*15} ║")
        
        jitter = cell_kpi["jitter"]
        for direction, label in [("dl", "Downlink"), ("ul", "Uplink"), ("combined", "Combined")]:
            if direction in jitter:
                j = jitter[direction]
                print(f"║  {label:<12} {j['std_us']:>15.2f} {j['cv_pct']:>12.2f} {j['max_deviation_us']:>15.2f} {j['p99_deviation_us']:>15.2f} ║")
        
        # On-time Summary
        if cell_kpi.get("ontime"):
            print(f"╠──────────────────────────────────────────────────────────────────────────────────────────────╣")
            print(f"║  ✅ ON-TIME RATE                                                                             ║")
            ontime = cell_kpi["ontime"]
            ontime_str = "  ".join([f"{k}: {v*100:.1f}%" for k, v in ontime.items() if isinstance(v, (int, float))])
            print(f"║  {ontime_str:<92} ║")
        
        print(f"╚══════════════════════════════════════════════════════════════════════════════════════════════╝")
    
    # Summary
    print("\n" + "=" * 100)
    print("                                    SUMMARY")
    print("=" * 100)
    
    # Find best performing cell (all on-time)
    best_cell = 0
    for cell in sorted(kpi.keys()):
        ontime = kpi[cell].get("ontime", {})
        all_ontime = all(v == 1.0 for v in ontime.values() if isinstance(v, (int, float)))
        if all_ontime:
            best_cell = cell
    
    print(f"\n  🏆 Maximum Cell Capacity (100% on-time): {best_cell} cells")
    
    if best_cell > 0 and best_cell in kpi:
        best_kpi = kpi[best_cell]
        
        # Best cell latency summary
        if "ALL" in best_kpi["latency"]["combined"]:
            s = best_kpi["latency"]["combined"]["ALL"]
            print(f"\n  📊 Latency @ {best_cell} cells:")
            print(f"     Average: {s['avg_us']:.2f} μs  |  Max: {s['max_us']:.2f} μs  |  P99: {s['p99_us']:.2f} μs")
        
        # Throughput
        tp = best_kpi["throughput"]
        print(f"\n  ⚡ Throughput @ {best_cell} cells:")
        print(f"     {tp['slots_per_second']:.2f} slots/sec  |  {tp['cell_slots_per_second']:.2f} cell-slots/sec")
        
        # Jitter
        if "combined" in best_kpi["jitter"]:
            j = best_kpi["jitter"]["combined"]
            print(f"\n  📉 Jitter (Std Dev) @ {best_cell} cells:")
            print(f"     {j['std_us']:.2f} μs  (CV: {j['cv_pct']:.2f}%)")
    
    print("\n" + "=" * 100)


def export_kpi_json(kpi, config, filepath):
    """Export KPI results to JSON file."""
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "kpi_by_cell": kpi
    }
    
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✅ KPI report exported to: {filepath}")


def generate_summary_table(kpi_results, config, output_prefix):
    """Generate summary table as CSV and formatted text."""
    
    # Prepare data for table
    table_data = []
    headers = ["Cell", "DL_Avg(μs)", "DL_Max(μs)", "DL_P99(μs)", "DL_Jitter(μs)",
               "UL_Avg(μs)", "UL_Max(μs)", "UL_P99(μs)", "UL_Jitter(μs)",
               "TPS", "Cell-TPS", "OnTime(%)"]
    
    for cell in sorted(kpi_results.keys()):
        k = kpi_results[cell]
        
        # DL stats
        dl_stats = k["latency"]["dl"].get("COMBINED", {})
        dl_avg = dl_stats.get("avg_us", 0)
        dl_max = dl_stats.get("max_us", 0)
        dl_p99 = dl_stats.get("p99_us", 0)
        dl_jitter = k["jitter"].get("dl", {}).get("std_us", 0)
        
        # UL stats
        ul_stats = k["latency"]["ul"].get("COMBINED", {})
        ul_avg = ul_stats.get("avg_us", 0)
        ul_max = ul_stats.get("max_us", 0)
        ul_p99 = ul_stats.get("p99_us", 0)
        ul_jitter = k["jitter"].get("ul", {}).get("std_us", 0)
        
        # Throughput
        tps = k["throughput"]["slots_per_second"]
        cell_tps = k["throughput"]["cell_slots_per_second"]
        
        # On-time
        ontime = k.get("ontime", {})
        ontime_values = [v for v in ontime.values() if isinstance(v, (int, float))]
        ontime_pct = min(ontime_values) * 100 if ontime_values else 0
        
        table_data.append([
            cell, dl_avg, dl_max, dl_p99, dl_jitter,
            ul_avg, ul_max, ul_p99, ul_jitter,
            tps, cell_tps, ontime_pct
        ])
    
    # Save as CSV
    csv_path = f"{output_prefix}_kpi_table.csv"
    with open(csv_path, 'w') as f:
        f.write(",".join(headers) + "\n")
        for row in table_data:
            f.write(",".join([f"{x:.2f}" if isinstance(x, float) else str(x) for x in row]) + "\n")
    
    print(f"\n✅ Summary table saved to: {csv_path}")
    
    # Print formatted table
    print("\n" + "=" * 140)
    print("                                        KPI SUMMARY TABLE")
    print("=" * 140)
    print(f"{'Cell':>5} │ {'DL_Avg':>8} {'DL_Max':>8} {'DL_P99':>8} {'DL_Jit':>8} │ {'UL_Avg':>8} {'UL_Max':>8} {'UL_P99':>8} {'UL_Jit':>8} │ {'TPS':>8} {'Cell-TPS':>10} {'OnTime':>8}")
    print("-" * 140)
    
    for row in table_data:
        cell, dl_avg, dl_max, dl_p99, dl_jit, ul_avg, ul_max, ul_p99, ul_jit, tps, ctps, ontime = row
        ontime_mark = "✅" if ontime >= 100 else "⚠️" if ontime >= 99 else "❌"
        print(f"{cell:>5} │ {dl_avg:>8.1f} {dl_max:>8.1f} {dl_p99:>8.1f} {dl_jit:>8.2f} │ {ul_avg:>8.1f} {ul_max:>8.1f} {ul_p99:>8.1f} {ul_jit:>8.2f} │ {tps:>8.0f} {ctps:>10.0f} {ontime:>6.1f}% {ontime_mark}")
    
    print("=" * 140)
    
    return table_data


def generate_kpi_graphs(kpi_results, config, output_prefix):
    """Generate KPI graphs using matplotlib."""
    
    if not HAS_MATPLOTLIB:
        print("⚠️ matplotlib not available. Skipping graph generation.")
        return
    
    cells = sorted(kpi_results.keys())
    
    # Extract data for plotting
    dl_avg, dl_max, dl_p99, dl_jitter = [], [], [], []
    ul_avg, ul_max, ul_p99, ul_jitter = [], [], [], []
    tps, cell_tps = [], []
    combined_avg, combined_max, combined_p99, combined_jitter = [], [], [], []
    
    for cell in cells:
        k = kpi_results[cell]
        
        # DL
        dl_stats = k["latency"]["dl"].get("COMBINED", {})
        dl_avg.append(dl_stats.get("avg_us", 0))
        dl_max.append(dl_stats.get("max_us", 0))
        dl_p99.append(dl_stats.get("p99_us", 0))
        dl_jitter.append(k["jitter"].get("dl", {}).get("std_us", 0))
        
        # UL
        ul_stats = k["latency"]["ul"].get("COMBINED", {})
        ul_avg.append(ul_stats.get("avg_us", 0))
        ul_max.append(ul_stats.get("max_us", 0))
        ul_p99.append(ul_stats.get("p99_us", 0))
        ul_jitter.append(k["jitter"].get("ul", {}).get("std_us", 0))
        
        # Combined
        comb_stats = k["latency"]["combined"].get("ALL", {})
        combined_avg.append(comb_stats.get("avg_us", 0))
        combined_max.append(comb_stats.get("max_us", 0))
        combined_p99.append(comb_stats.get("p99_us", 0))
        combined_jitter.append(k["jitter"].get("combined", {}).get("std_us", 0))
        
        # Throughput
        tps.append(k["throughput"]["slots_per_second"])
        cell_tps.append(k["throughput"]["cell_slots_per_second"])
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)
    
    # Color scheme
    colors = {
        'dl': '#2E86AB',      # Blue
        'ul': '#A23B72',      # Pink
        'combined': '#28965A', # Green
        'jitter': '#F18F01',   # Orange
        'tps': '#C73E1D'       # Red
    }
    
    # Get target info for title
    target = config.get('target', '?')
    if isinstance(target, list):
        target_str = f"{target[0]}/{target[1]}"
    else:
        target_str = str(target)
    
    # ============ Plot 1: Latency vs Cell Count ============
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(cells, dl_avg, 'o-', color=colors['dl'], linewidth=2, markersize=6, label='DL Avg')
    ax1.plot(cells, dl_p99, 's--', color=colors['dl'], linewidth=1.5, markersize=5, alpha=0.7, label='DL P99')
    ax1.plot(cells, ul_avg, 'o-', color=colors['ul'], linewidth=2, markersize=6, label='UL Avg')
    ax1.plot(cells, ul_p99, 's--', color=colors['ul'], linewidth=1.5, markersize=5, alpha=0.7, label='UL P99')
    ax1.axhline(y=500, color='red', linestyle=':', linewidth=2, label='500μs Deadline')
    
    ax1.set_xlabel('Cell Count', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Latency (μs)', fontsize=11, fontweight='bold')
    ax1.set_title('📊 Latency vs Cell Count', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(cells)
    
    # ============ Plot 2: Jitter (Std Dev) vs Cell Count ============
    ax2 = fig.add_subplot(gs[0, 1])
    
    width = 0.35
    x = np.array(cells)
    ax2.bar(x - width/2, dl_jitter, width, label='DL Jitter', color=colors['dl'], alpha=0.8)
    ax2.bar(x + width/2, ul_jitter, width, label='UL Jitter', color=colors['ul'], alpha=0.8)
    ax2.plot(cells, combined_jitter, 'D-', color=colors['jitter'], linewidth=2, markersize=7, label='Combined Jitter')
    
    ax2.set_xlabel('Cell Count', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Jitter / Std Dev (μs)', fontsize=11, fontweight='bold')
    ax2.set_title('📉 Jitter (Standard Deviation) vs Cell Count', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xticks(cells)
    
    # ============ Plot 3: Max Latency vs Cell Count ============
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.fill_between(cells, 0, dl_max, alpha=0.3, color=colors['dl'], label='DL Max')
    ax3.fill_between(cells, 0, ul_max, alpha=0.3, color=colors['ul'], label='UL Max')
    ax3.plot(cells, dl_max, 'o-', color=colors['dl'], linewidth=2, markersize=6)
    ax3.plot(cells, ul_max, 'o-', color=colors['ul'], linewidth=2, markersize=6)
    ax3.axhline(y=500, color='red', linestyle=':', linewidth=2, label='500μs Deadline')
    
    ax3.set_xlabel('Cell Count', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Max Latency (μs)', fontsize=11, fontweight='bold')
    ax3.set_title('⚡ Max Latency vs Cell Count', fontsize=13, fontweight='bold')
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(cells)
    
    # ============ Plot 4: Throughput vs Cell Count ============
    ax4 = fig.add_subplot(gs[1, 1])
    
    ax4_twin = ax4.twinx()
    
    bars = ax4.bar(cells, cell_tps, color=colors['tps'], alpha=0.7, label='Cell-Slots/sec')
    line, = ax4_twin.plot(cells, tps, 'D-', color=colors['combined'], linewidth=2, markersize=8, label='Slots/sec')
    
    ax4.set_xlabel('Cell Count', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Cell-Slots per Second', fontsize=11, fontweight='bold', color=colors['tps'])
    ax4_twin.set_ylabel('Slots per Second', fontsize=11, fontweight='bold', color=colors['combined'])
    ax4.set_title('🚀 Throughput vs Cell Count', fontsize=13, fontweight='bold')
    ax4.tick_params(axis='y', labelcolor=colors['tps'])
    ax4_twin.tick_params(axis='y', labelcolor=colors['combined'])
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_xticks(cells)
    
    # Combined legend
    lines = [bars, line]
    labels = ['Cell-Slots/sec', 'Slots/sec']
    ax4.legend(lines, labels, loc='upper left', fontsize=9)
    
    # Main title
    fig.suptitle(f'KPI Analysis Report - Target SM: {target_str}, Freq: {config.get("freq", "?")} MHz', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # Save figure
    graph_path = f"{output_prefix}_kpi_graphs.png"
    plt.savefig(graph_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ KPI graphs saved to: {graph_path}")
    
    # Generate additional detailed graph
    generate_detailed_latency_graph(kpi_results, config, output_prefix, cells)


def generate_detailed_latency_graph(kpi_results, config, output_prefix, cells):
    """Generate detailed latency distribution graph."""
    
    if not HAS_MATPLOTLIB:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Get target info for title
    target = config.get('target', '?')
    if isinstance(target, list):
        target_str = f"{target[0]}/{target[1]}"
    else:
        target_str = str(target)
    
    # Colors
    colors = plt.cm.viridis(np.linspace(0, 1, len(cells)))
    
    # ============ Plot 1: DL Latency Distribution by Cell ============
    ax1 = axes[0, 0]
    dl_data = []
    for cell in cells:
        k = kpi_results[cell]
        dl_stats = k["latency"]["dl"].get("COMBINED", {})
        if dl_stats:
            dl_data.append([dl_stats["min_us"], dl_stats["p50_us"], dl_stats["avg_us"], 
                           dl_stats["p95_us"], dl_stats["p99_us"], dl_stats["max_us"]])
    
    if dl_data:
        dl_data = np.array(dl_data)
        ax1.fill_between(cells, dl_data[:, 0], dl_data[:, 5], alpha=0.2, color='blue', label='Min-Max')
        ax1.fill_between(cells, dl_data[:, 1], dl_data[:, 4], alpha=0.3, color='blue', label='P50-P99')
        ax1.plot(cells, dl_data[:, 2], 'o-', color='blue', linewidth=2, label='Avg')
        ax1.axhline(y=500, color='red', linestyle=':', linewidth=2, label='500μs')
    
    ax1.set_xlabel('Cell Count', fontweight='bold')
    ax1.set_ylabel('DL Latency (μs)', fontweight='bold')
    ax1.set_title('Downlink Latency Distribution', fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(cells)
    
    # ============ Plot 2: UL Latency Distribution by Cell ============
    ax2 = axes[0, 1]
    ul_data = []
    for cell in cells:
        k = kpi_results[cell]
        ul_stats = k["latency"]["ul"].get("COMBINED", {})
        if ul_stats:
            ul_data.append([ul_stats["min_us"], ul_stats["p50_us"], ul_stats["avg_us"], 
                           ul_stats["p95_us"], ul_stats["p99_us"], ul_stats["max_us"]])
    
    if ul_data:
        ul_data = np.array(ul_data)
        ax2.fill_between(cells, ul_data[:, 0], ul_data[:, 5], alpha=0.2, color='purple', label='Min-Max')
        ax2.fill_between(cells, ul_data[:, 1], ul_data[:, 4], alpha=0.3, color='purple', label='P50-P99')
        ax2.plot(cells, ul_data[:, 2], 'o-', color='purple', linewidth=2, label='Avg')
        ax2.axhline(y=500, color='red', linestyle=':', linewidth=2, label='500μs')
    
    ax2.set_xlabel('Cell Count', fontweight='bold')
    ax2.set_ylabel('UL Latency (μs)', fontweight='bold')
    ax2.set_title('Uplink Latency Distribution', fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(cells)
    
    # ============ Plot 3: Latency Percentiles Comparison ============
    ax3 = axes[1, 0]
    
    percentiles = ['Avg', 'P50', 'P90', 'P95', 'P99', 'Max']
    x = np.arange(len(percentiles))
    width = 0.12
    
    # Select a few key cell counts for comparison
    key_cells = [c for c in [1, 4, 8, 12, 16] if c in cells]
    
    for i, cell in enumerate(key_cells):
        k = kpi_results[cell]
        comb_stats = k["latency"]["combined"].get("ALL", {})
        if comb_stats:
            values = [comb_stats["avg_us"], comb_stats["p50_us"], comb_stats["p90_us"],
                     comb_stats["p95_us"], comb_stats["p99_us"], comb_stats["max_us"]]
            ax3.bar(x + i * width, values, width, label=f'{cell} cells', alpha=0.8)
    
    ax3.axhline(y=500, color='red', linestyle=':', linewidth=2, label='500μs')
    ax3.set_xlabel('Percentile', fontweight='bold')
    ax3.set_ylabel('Latency (μs)', fontweight='bold')
    ax3.set_title('Latency Percentiles Comparison', fontweight='bold')
    ax3.set_xticks(x + width * (len(key_cells) - 1) / 2)
    ax3.set_xticklabels(percentiles)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # ============ Plot 4: Jitter Trend ============
    ax4 = axes[1, 1]
    
    dl_jitter = [kpi_results[c]["jitter"].get("dl", {}).get("std_us", 0) for c in cells]
    ul_jitter = [kpi_results[c]["jitter"].get("ul", {}).get("std_us", 0) for c in cells]
    dl_cv = [kpi_results[c]["jitter"].get("dl", {}).get("cv_pct", 0) for c in cells]
    ul_cv = [kpi_results[c]["jitter"].get("ul", {}).get("cv_pct", 0) for c in cells]
    
    ax4_twin = ax4.twinx()
    
    ax4.plot(cells, dl_jitter, 'o-', color='blue', linewidth=2, label='DL Std Dev')
    ax4.plot(cells, ul_jitter, 's-', color='purple', linewidth=2, label='UL Std Dev')
    ax4_twin.plot(cells, dl_cv, 'o--', color='blue', alpha=0.5, linewidth=1.5, label='DL CV%')
    ax4_twin.plot(cells, ul_cv, 's--', color='purple', alpha=0.5, linewidth=1.5, label='UL CV%')
    
    ax4.set_xlabel('Cell Count', fontweight='bold')
    ax4.set_ylabel('Jitter / Std Dev (μs)', fontweight='bold')
    ax4_twin.set_ylabel('Coefficient of Variation (%)', fontweight='bold')
    ax4.set_title('Jitter Trend Analysis', fontweight='bold')
    ax4.legend(loc='upper left', fontsize=8)
    ax4_twin.legend(loc='upper right', fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(cells)
    
    fig.suptitle(f'Detailed Latency Analysis - Target SM: {target_str}', fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    graph_path = f"{output_prefix}_latency_detail.png"
    plt.savefig(graph_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Detailed latency graph saved to: {graph_path}")


def generate_markdown_report(kpi_results, config, output_prefix):
    """Generate comprehensive Markdown report."""
    
    cells = sorted(kpi_results.keys())
    
    # Get target info
    target = config.get('target', '?')
    if isinstance(target, list):
        target_str = f"{target[0]}/{target[1]}"
    else:
        target_str = str(target)
    
    # Build markdown content
    md_lines = []
    
    # Header
    md_lines.append("# KPI Analysis Report")
    md_lines.append("")
    md_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md_lines.append("")
    
    # Test Configuration
    md_lines.append("## Test Configuration")
    md_lines.append("")
    md_lines.append("| Parameter | Value |")
    md_lines.append("|-----------|-------|")
    md_lines.append(f"| Target SM | {target_str} |")
    md_lines.append(f"| Frequency | {config.get('freq', '?')} MHz |")
    md_lines.append(f"| Iterations | {config.get('iterations', '?')} |")
    md_lines.append(f"| Slots/Sweeps | {config.get('sweeps', '?')} |")
    md_lines.append(f"| TDD Pattern | {config.get('pattern', '?')} |")
    md_lines.append(f"| Delay | {config.get('delay', '?')} μs |")
    md_lines.append(f"| GPU | {config.get('gpu', '?')} |")
    md_lines.append("")
    
    # Summary Section
    md_lines.append("## Summary")
    md_lines.append("")
    
    # Find best performing cell
    best_cell = 0
    for cell in cells:
        ontime = kpi_results[cell].get("ontime", {})
        all_ontime = all(v == 1.0 for v in ontime.values() if isinstance(v, (int, float)))
        if all_ontime:
            best_cell = cell
    
    md_lines.append(f"**Maximum Cell Capacity (100% on-time):** {best_cell} cells")
    md_lines.append("")
    
    if best_cell > 0 and best_cell in kpi_results:
        best_kpi = kpi_results[best_cell]
        if "ALL" in best_kpi["latency"]["combined"]:
            s = best_kpi["latency"]["combined"]["ALL"]
            md_lines.append(f"**Latency @ {best_cell} cells:** Avg: {s['avg_us']:.2f} μs | Max: {s['max_us']:.2f} μs | P99: {s['p99_us']:.2f} μs")
        tp = best_kpi["throughput"]
        md_lines.append(f"**Throughput @ {best_cell} cells:** {tp['slots_per_second']:.2f} slots/sec | {tp['cell_slots_per_second']:.2f} cell-slots/sec")
        if "combined" in best_kpi["jitter"]:
            j = best_kpi["jitter"]["combined"]
            md_lines.append(f"**Jitter @ {best_cell} cells:** {j['std_us']:.2f} μs (CV: {j['cv_pct']:.2f}%)")
    md_lines.append("")
    
    # KPI Summary Table
    md_lines.append("## KPI Summary Table")
    md_lines.append("")
    md_lines.append("| Cell | DL Avg (μs) | DL Max (μs) | DL P99 (μs) | DL Jitter (μs) | UL Avg (μs) | UL Max (μs) | UL P99 (μs) | UL Jitter (μs) | TPS | Cell-TPS | OnTime |")
    md_lines.append("|------|-------------|-------------|-------------|----------------|-------------|-------------|-------------|----------------|-----|----------|--------|")
    
    for cell in cells:
        k = kpi_results[cell]
        
        # DL stats
        dl_stats = k["latency"]["dl"].get("COMBINED", {})
        dl_avg = dl_stats.get("avg_us", 0)
        dl_max = dl_stats.get("max_us", 0)
        dl_p99 = dl_stats.get("p99_us", 0)
        dl_jitter = k["jitter"].get("dl", {}).get("std_us", 0)
        
        # UL stats
        ul_stats = k["latency"]["ul"].get("COMBINED", {})
        ul_avg = ul_stats.get("avg_us", 0)
        ul_max = ul_stats.get("max_us", 0)
        ul_p99 = ul_stats.get("p99_us", 0)
        ul_jitter = k["jitter"].get("ul", {}).get("std_us", 0)
        
        # Throughput
        tps = k["throughput"]["slots_per_second"]
        cell_tps = k["throughput"]["cell_slots_per_second"]
        
        # On-time
        ontime = k.get("ontime", {})
        ontime_values = [v for v in ontime.values() if isinstance(v, (int, float))]
        ontime_pct = min(ontime_values) * 100 if ontime_values else 0
        ontime_mark = "✅" if ontime_pct >= 100 else "⚠️" if ontime_pct >= 99 else "❌"
        
        md_lines.append(f"| {cell} | {dl_avg:.1f} | {dl_max:.1f} | {dl_p99:.1f} | {dl_jitter:.2f} | {ul_avg:.1f} | {ul_max:.1f} | {ul_p99:.1f} | {ul_jitter:.2f} | {tps:.0f} | {cell_tps:.0f} | {ontime_pct:.1f}% {ontime_mark} |")
    
    md_lines.append("")
    
    # Detailed Results per Cell
    md_lines.append("## Detailed Results per Cell")
    md_lines.append("")
    
    for cell in cells:
        cell_kpi = kpi_results[cell]
        
        md_lines.append(f"### Cell Count: {cell}")
        md_lines.append("")
        
        # Latency Table
        md_lines.append("#### Latency (μs)")
        md_lines.append("")
        md_lines.append("| Channel | Count | Avg | Max | P99 | P95 | P50 | Min | Std |")
        md_lines.append("|---------|-------|-----|-----|-----|-----|-----|-----|-----|")
        
        # DL Channels
        dl_lat = cell_kpi["latency"]["dl"]
        for ch in sorted(dl_lat.keys()):
            if ch == "COMBINED":
                continue
            s = dl_lat[ch]
            md_lines.append(f"| {ch} | {s['count']} | {s['avg_us']:.2f} | {s['max_us']:.2f} | {s['p99_us']:.2f} | {s['p95_us']:.2f} | {s['p50_us']:.2f} | {s['min_us']:.2f} | {s['std_us']:.2f} |")
        
        if "COMBINED" in dl_lat:
            s = dl_lat["COMBINED"]
            md_lines.append(f"| **DL_TOTAL** | {s['count']} | {s['avg_us']:.2f} | {s['max_us']:.2f} | {s['p99_us']:.2f} | {s['p95_us']:.2f} | {s['p50_us']:.2f} | {s['min_us']:.2f} | {s['std_us']:.2f} |")
        
        # UL Channels
        ul_lat = cell_kpi["latency"]["ul"]
        for ch in sorted(ul_lat.keys()):
            if ch == "COMBINED":
                continue
            s = ul_lat[ch]
            md_lines.append(f"| {ch} | {s['count']} | {s['avg_us']:.2f} | {s['max_us']:.2f} | {s['p99_us']:.2f} | {s['p95_us']:.2f} | {s['p50_us']:.2f} | {s['min_us']:.2f} | {s['std_us']:.2f} |")
        
        if "COMBINED" in ul_lat:
            s = ul_lat["COMBINED"]
            md_lines.append(f"| **UL_TOTAL** | {s['count']} | {s['avg_us']:.2f} | {s['max_us']:.2f} | {s['p99_us']:.2f} | {s['p95_us']:.2f} | {s['p50_us']:.2f} | {s['min_us']:.2f} | {s['std_us']:.2f} |")
        
        if "ALL" in cell_kpi["latency"]["combined"]:
            s = cell_kpi["latency"]["combined"]["ALL"]
            md_lines.append(f"| **ALL_COMBINED** | {s['count']} | {s['avg_us']:.2f} | {s['max_us']:.2f} | {s['p99_us']:.2f} | {s['p95_us']:.2f} | {s['p50_us']:.2f} | {s['min_us']:.2f} | {s['std_us']:.2f} |")
        
        md_lines.append("")
        
        # Throughput
        md_lines.append("#### Throughput")
        md_lines.append("")
        tp = cell_kpi["throughput"]
        md_lines.append(f"- **Total Patterns:** {tp['total_patterns']}")
        md_lines.append(f"- **Total Slots:** {tp['total_slots']}")
        md_lines.append(f"- **Total Time:** {tp['total_time_sec']:.4f} sec")
        md_lines.append(f"- **Slots per Second (TPS):** {tp['slots_per_second']:.2f} (Theoretical: {tp['theoretical_slots_per_second']})")
        md_lines.append(f"- **Patterns per Second:** {tp['patterns_per_second']:.2f}")
        md_lines.append(f"- **Cell-Slots per Second:** {tp['cell_slots_per_second']:.2f}")
        md_lines.append("")
        
        # Jitter
        md_lines.append("#### Jitter (Latency Variability)")
        md_lines.append("")
        md_lines.append("| Direction | Std Dev (μs) | CV (%) | Max Deviation (μs) | P99 Deviation (μs) |")
        md_lines.append("|-----------|--------------|--------|--------------------|--------------------|")
        
        jitter = cell_kpi["jitter"]
        for direction, label in [("dl", "Downlink"), ("ul", "Uplink"), ("combined", "Combined")]:
            if direction in jitter:
                j = jitter[direction]
                md_lines.append(f"| {label} | {j['std_us']:.2f} | {j['cv_pct']:.2f} | {j['max_deviation_us']:.2f} | {j['p99_deviation_us']:.2f} |")
        
        md_lines.append("")
        
        # On-time Rate
        if cell_kpi.get("ontime"):
            md_lines.append("#### On-Time Rate")
            md_lines.append("")
            ontime = cell_kpi["ontime"]
            for k, v in ontime.items():
                if isinstance(v, (int, float)):
                    md_lines.append(f"- **{k}:** {v*100:.1f}%")
            md_lines.append("")
        
        md_lines.append("---")
        md_lines.append("")
    
    # Graphs Reference (if generated)
    md_lines.append("## Graphs")
    md_lines.append("")
    md_lines.append("The following graphs are generated with the `--graph` option:")
    md_lines.append("")
    md_lines.append(f"- **KPI Summary:** `{output_prefix}_kpi_graphs.png`")
    md_lines.append(f"- **Latency Detail:** `{output_prefix}_latency_detail.png`")
    md_lines.append("")
    
    # Write to file
    md_path = f"{output_prefix}_kpi_report.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
    
    print(f"✅ Markdown report saved to: {md_path}")
    return md_path


def main():
    parser = argparse.ArgumentParser(
        description="Analyze KPIs (Latency, Throughput, Jitter) from measure.py results"
    )
    parser.add_argument(
        "result_file",
        help="Path to measure.py result JSON file"
    )
    parser.add_argument(
        "--cell", "-c",
        type=int,
        default=None,
        help="Analyze specific cell count only"
    )
    parser.add_argument(
        "--export", "-e",
        type=str,
        default=None,
        help="Export KPI results to JSON file"
    )
    parser.add_argument(
        "--brief", "-b",
        action="store_true",
        help="Show brief summary only"
    )
    parser.add_argument(
        "--graph", "-g",
        action="store_true",
        help="Generate KPI graphs (PNG)"
    )
    parser.add_argument(
        "--table", "-t",
        action="store_true",
        help="Generate summary table (CSV)"
    )
    parser.add_argument(
        "--markdown", "-m",
        action="store_true",
        help="Generate comprehensive Markdown report"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Generate all outputs (graph, table, markdown)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output prefix for graphs and tables (default: input filename)"
    )
    parser.add_argument(
        "--tx-jitter", "-j",
        action="store_true",
        help="Analyze TX Timing Jitter from buffer files (requires buffer-XX.txt files)"
    )
    parser.add_argument(
        "--buffer-dir",
        type=str,
        default=".",
        help="Directory containing buffer-XX.txt files (default: current directory)"
    )
    parser.add_argument(
        "--buffer",
        type=str,
        default=None,
        help="Analyze specific buffer file for TX Timing Jitter"
    )
    
    args = parser.parse_args()
    
    # Load result file
    if not os.path.exists(args.result_file):
        print(f"❌ Error: File not found: {args.result_file}")
        sys.exit(1)
    
    print(f"\n📂 Loading: {args.result_file}")
    data = load_result_file(args.result_file)
    
    # Extract config and cells
    config = extract_test_config(data)
    cells = extract_cell_results(data)
    
    if not cells:
        print("❌ Error: No cell data found in result file")
        sys.exit(1)
    
    print(f"✅ Found {len(cells)} cell configurations: {sorted(cells.keys())}")
    
    # Analyze KPIs for each cell
    kpi_results = {}
    for cell_count, cell_data in cells.items():
        kpi_results[cell_count] = analyze_cell_kpi(cell_data, config)
    
    # Determine output prefix
    if args.output:
        output_prefix = args.output
    else:
        output_prefix = os.path.splitext(args.result_file)[0]
    
    # Print report
    if not args.brief:
        print_kpi_report(kpi_results, config, args.cell)
    else:
        # Brief summary
        print("\n" + "=" * 80)
        print("                         BRIEF KPI SUMMARY")
        print("=" * 80)
        print(f"\n{'Cell':>6} {'Avg(μs)':>10} {'Max(μs)':>10} {'P99(μs)':>10} {'Jitter(μs)':>12} {'TPS':>10}")
        print("-" * 80)
        
        for cell in sorted(kpi_results.keys()):
            k = kpi_results[cell]
            if "ALL" in k["latency"]["combined"]:
                s = k["latency"]["combined"]["ALL"]
                j = k["jitter"].get("combined", {}).get("std_us", 0)
                tps = k["throughput"]["slots_per_second"]
                print(f"{cell:>6} {s['avg_us']:>10.2f} {s['max_us']:>10.2f} {s['p99_us']:>10.2f} {j:>12.2f} {tps:>10.2f}")
        print("=" * 80)
    
    # Handle --all option
    if args.all:
        args.graph = True
        args.table = True
        args.markdown = True
    
    # Generate summary table
    if args.table:
        generate_summary_table(kpi_results, config, output_prefix)
    
    # Generate graphs
    if args.graph:
        generate_kpi_graphs(kpi_results, config, output_prefix)
    
    # Generate Markdown report
    if args.markdown:
        generate_markdown_report(kpi_results, config, output_prefix)
    
    # Export if requested
    if args.export:
        export_kpi_json(kpi_results, config, args.export)
    
    # TX Timing Jitter Analysis
    if args.tx_jitter:
        print("\n" + "=" * 100)
        print("                    TX TIMING JITTER ANALYSIS")
        print("=" * 100)
        
        if args.buffer:
            # Analyze single buffer file
            print(f"\n📂 Analyzing buffer file: {args.buffer}")
            tx_data = parse_buffer_for_tx_timing(args.buffer)
            if tx_data:
                # Extract cell count from filename
                match = re.search(r'buffer-(\d+)\.txt', args.buffer)
                cell_count = int(match.group(1)) if match else None
                print_tx_timing_jitter_report(tx_data, cell_count)
                
                # Generate graph for single buffer
                single_result = {cell_count: tx_data} if cell_count else {1: tx_data}
                generate_tx_jitter_graph(single_result, output_prefix, config)
            else:
                print(f"❌ Failed to parse buffer file: {args.buffer}")
        else:
            # Analyze all buffer files in directory
            buffer_dir = args.buffer_dir
            if not os.path.isabs(buffer_dir):
                # Make relative to JSON file location
                json_dir = os.path.dirname(os.path.abspath(args.result_file))
                buffer_dir = os.path.join(json_dir, buffer_dir)
            
            print(f"\n📂 Scanning buffer files in: {buffer_dir}")
            all_tx_results = analyze_all_buffers(buffer_dir)
            
            if all_tx_results:
                # Print summary table
                generate_tx_jitter_summary_table(all_tx_results)
                
                # Generate TX Jitter graph (always when tx_jitter is enabled)
                generate_tx_jitter_graph(all_tx_results, output_prefix, config)
                
                # Print detailed report for each cell if specific cell requested
                if args.cell and args.cell in all_tx_results:
                    print_tx_timing_jitter_report(all_tx_results[args.cell], args.cell)
                elif not args.brief:
                    # Print first and last cell details
                    cells = sorted(all_tx_results.keys())
                    if cells:
                        print_tx_timing_jitter_report(all_tx_results[cells[0]], cells[0])
                        if len(cells) > 1:
                            print_tx_timing_jitter_report(all_tx_results[cells[-1]], cells[-1])
            else:
                print(f"⚠️ No buffer files found in: {buffer_dir}")
                print("   TX Timing Jitter analysis requires buffer-XX.txt files.")
                print("   Run measure.py with --save_buffers option to generate them.")
    
    # Print usage hint if no output options specified
    if not args.graph and not args.table and not args.export and not args.markdown and not args.tx_jitter:
        print("\n💡 Tip: Use --graph for graphs, --table for CSV, --markdown for MD report, --all for everything")
        print("        Use --tx-jitter for TX Timing Jitter analysis (requires buffer files)")


if __name__ == "__main__":
    main()
