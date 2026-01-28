# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Target Sweep Analysis Tool

This script analyzes measure.py results across different target SM allocations
and generates comprehensive graphs showing:
- Capacity vs Target SM allocation
- Latency vs Cell Count for different targets
- Latency distribution comparisons
- Per-channel latency breakdown

Usage:
    # Analyze multiple result files
    python3 analyze_target_sweep.py result1.json result2.json result3.json

    # Analyze all matching files in directory
    python3 analyze_target_sweep.py --pattern "*_sweep_*.json"

    # Single file detailed analysis
    python3 analyze_target_sweep.py --single 040_040_sweep_graphs_avg_F08.json

Example workflow:
    1. Run measure.py with different target values:
       python3 measure.py ... --target 20 20 ... --graph
       python3 measure.py ... --target 30 30 ... --graph  
       python3 measure.py ... --target 40 40 ... --graph
       python3 measure.py ... --target 50 50 ... --graph

    2. Analyze results:
       python3 analyze_target_sweep.py 020_020_*.json 030_030_*.json 040_040_*.json 050_050_*.json
"""

import json
import os
import sys
import glob
import argparse
import numpy as np
from datetime import datetime

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not found. Install with: pip install matplotlib")


def get_unique_filename(base_path, extension):
    """
    Generate a unique filename by appending an incrementing counter.
    
    Example:
        base_path="target_sweep", extension=".png"
        - If target_sweep_1.png doesn't exist → returns "target_sweep_1.png"
        - If target_sweep_1.png exists → returns "target_sweep_2.png"
        - etc.
    """
    counter = 1
    while True:
        filename = f"{base_path}_{counter}{extension}"
        if not os.path.exists(filename):
            return filename
        counter += 1


def parse_result_file(filepath):
    """Parse a measure.py result JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    result = {
        "filepath": filepath,
        "filename": os.path.basename(filepath),
        "testConfig": data.get("testConfig", {}),
        "cells": {}
    }
    
    # Extract target SM allocation
    target = data.get("testConfig", {}).get("target", ["?", "?"])
    if isinstance(target, list) and len(target) >= 2:
        result["target_dl"] = int(target[0]) if target[0] != "?" else 0
        result["target_ul"] = int(target[1]) if target[1] != "?" else 0
    else:
        result["target_dl"] = 0
        result["target_ul"] = 0
    
    result["target_label"] = f"{result['target_dl']:02d}/{result['target_ul']:02d}"
    result["target_total"] = result["target_dl"] + result["target_ul"]
    
    # Extract other config
    result["freq"] = data.get("testConfig", {}).get("freq", "?")
    result["delay"] = data.get("testConfig", {}).get("delay", "?")
    result["pattern"] = data.get("testConfig", {}).get("pattern", "?")
    result["gpu"] = data.get("testConfig", {}).get("gpuName", "Unknown GPU")
    result["iterations"] = data.get("testConfig", {}).get("iterations", "?")
    result["sweeps"] = data.get("testConfig", {}).get("sweeps", "?")
    
    # Parse cell results
    for key, value in data.items():
        if key == "testConfig" or not isinstance(value, dict):
            continue
        
        # Extract cell count from key (e.g., "01+00" -> 1)
        try:
            cell_count = int(key.split("+")[0])
        except:
            continue
        
        cell_data = {
            "key": key,
            "cell_count": cell_count,
            "structure": value.get("Structure", False),
            "mode": value.get("Mode", "Unknown"),
        }
        
        # Extract latency data
        for channel in ["PDSCH", "PUSCH1", "PUSCH2"]:
            latencies = value.get(channel, [])
            if latencies:
                cell_data[f"{channel}_latencies"] = latencies
                cell_data[f"{channel}_avg"] = np.mean(latencies)
                cell_data[f"{channel}_std"] = np.std(latencies)
                cell_data[f"{channel}_min"] = np.min(latencies)
                cell_data[f"{channel}_max"] = np.max(latencies)
                cell_data[f"{channel}_p50"] = np.percentile(latencies, 50)
                cell_data[f"{channel}_p95"] = np.percentile(latencies, 95)
                cell_data[f"{channel}_p99"] = np.percentile(latencies, 99)
        
        # Extract on-time percent
        ontime = value.get("ontimePercent", {})
        cell_data["ontime_pdsch"] = ontime.get("PDSCH", 0)
        cell_data["ontime_pusch1"] = ontime.get("PUSCH1", 0)
        cell_data["ontime_pusch2"] = ontime.get("PUSCH2", 0)
        
        # Check if all channels are on-time
        cell_data["all_ontime"] = (
            cell_data["ontime_pdsch"] == 1.0 and
            cell_data["ontime_pusch1"] == 1.0 and
            cell_data["ontime_pusch2"] == 1.0
        )
        
        # Memory usage
        mem = value.get("memoryUseMB", {})
        cell_data["memory_total_mb"] = mem.get("total", 0)
        
        result["cells"][cell_count] = cell_data
    
    # Calculate capacity
    capacity = 0
    for cell_count in sorted(result["cells"].keys()):
        if result["cells"][cell_count]["all_ontime"]:
            capacity = cell_count
        else:
            break
    result["capacity"] = capacity
    
    # Calculate capacity breakdown
    pdsch_cap = 0
    pusch1_cap = 0
    pusch2_cap = 0
    for cell_count in sorted(result["cells"].keys()):
        cell = result["cells"][cell_count]
        if cell["ontime_pdsch"] == 1.0:
            pdsch_cap = cell_count
        if cell["ontime_pusch1"] == 1.0:
            pusch1_cap = cell_count
        if cell["ontime_pusch2"] == 1.0:
            pusch2_cap = cell_count
    
    result["capacity_pdsch"] = pdsch_cap
    result["capacity_pusch1"] = pusch1_cap
    result["capacity_pusch2"] = pusch2_cap
    
    return result


def plot_single_file_analysis(result, output_prefix):
    """Generate detailed graphs for a single result file."""
    if not HAS_MATPLOTLIB:
        print("matplotlib required for plotting")
        return
    
    cells = result["cells"]
    cell_counts = sorted(cells.keys())
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(f"Target {result['target_label']} Analysis\n"
                 f"GPU: {result['gpu']} | Freq: {result['freq']} MHz | Pattern: {result['pattern']} | "
                 f"Iterations: {result['iterations']} | Capacity: {result['capacity']} cells",
                 fontsize=14, fontweight='bold')
    
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Color scheme
    colors = {
        'PDSCH': '#2ecc71',    # Green
        'PUSCH1': '#3498db',   # Blue  
        'PUSCH2': '#e74c3c',   # Red
    }
    
    # ===== Plot 1: Average Latency vs Cell Count =====
    ax1 = fig.add_subplot(gs[0, 0])
    for channel, color in colors.items():
        avg_key = f"{channel}_avg"
        latencies = [cells[c].get(avg_key, 0) for c in cell_counts]
        ax1.plot(cell_counts, latencies, 'o-', label=channel, color=color, linewidth=2, markersize=6)
    
    ax1.axvline(x=result['capacity'], color='gray', linestyle='--', alpha=0.7, label=f"Capacity ({result['capacity']})")
    ax1.set_xlabel('Cell Count', fontsize=11)
    ax1.set_ylabel('Average Latency (μs)', fontsize=11)
    ax1.set_title('Average Latency vs Cell Count', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(cell_counts)
    
    # ===== Plot 2: P95/P99 Latency vs Cell Count =====
    ax2 = fig.add_subplot(gs[0, 1])
    for channel, color in colors.items():
        p95_key = f"{channel}_p95"
        p99_key = f"{channel}_p99"
        p95_vals = [cells[c].get(p95_key, 0) for c in cell_counts]
        p99_vals = [cells[c].get(p99_key, 0) for c in cell_counts]
        ax2.plot(cell_counts, p95_vals, 'o--', label=f'{channel} P95', color=color, alpha=0.7, linewidth=1.5)
        ax2.plot(cell_counts, p99_vals, 's-', label=f'{channel} P99', color=color, linewidth=2)
    
    ax2.axvline(x=result['capacity'], color='gray', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Cell Count', fontsize=11)
    ax2.set_ylabel('Latency (μs)', fontsize=11)
    ax2.set_title('P95/P99 Latency vs Cell Count', fontsize=12)
    ax2.legend(loc='upper left', fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(cell_counts)
    
    # ===== Plot 3: On-Time Percentage vs Cell Count =====
    ax3 = fig.add_subplot(gs[0, 2])
    ontime_pdsch = [cells[c].get("ontime_pdsch", 0) * 100 for c in cell_counts]
    ontime_pusch1 = [cells[c].get("ontime_pusch1", 0) * 100 for c in cell_counts]
    ontime_pusch2 = [cells[c].get("ontime_pusch2", 0) * 100 for c in cell_counts]
    
    ax3.plot(cell_counts, ontime_pdsch, 'o-', label='PDSCH', color=colors['PDSCH'], linewidth=2, markersize=6)
    ax3.plot(cell_counts, ontime_pusch1, 's-', label='PUSCH1', color=colors['PUSCH1'], linewidth=2, markersize=6)
    ax3.plot(cell_counts, ontime_pusch2, '^-', label='PUSCH2', color=colors['PUSCH2'], linewidth=2, markersize=6)
    
    ax3.axhline(y=100, color='green', linestyle=':', alpha=0.5, label='100% threshold')
    ax3.axvline(x=result['capacity'], color='gray', linestyle='--', alpha=0.7, label=f"Capacity ({result['capacity']})")
    ax3.set_xlabel('Cell Count', fontsize=11)
    ax3.set_ylabel('On-Time (%)', fontsize=11)
    ax3.set_title('On-Time Percentage vs Cell Count', fontsize=12)
    ax3.legend(loc='lower left')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-5, 110)
    ax3.set_xticks(cell_counts)
    
    # ===== Plot 4: Latency Box Plot for Each Channel =====
    ax4 = fig.add_subplot(gs[1, :])
    
    box_data = []
    labels = []
    positions = []
    colors_list = []
    
    pos = 1
    for cell_count in cell_counts:
        cell = cells[cell_count]
        for i, channel in enumerate(['PDSCH', 'PUSCH1', 'PUSCH2']):
            lat_key = f"{channel}_latencies"
            if lat_key in cell and cell[lat_key]:
                box_data.append(cell[lat_key])
                labels.append(f"{cell_count}\n{channel[:2]}")
                positions.append(pos)
                colors_list.append(colors[channel])
            pos += 1
        pos += 1  # Gap between cell groups
    
    if box_data:
        bp = ax4.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
    
    # Add capacity line
    cap_pos = result['capacity'] * 4 + 2
    ax4.axvline(x=cap_pos, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f"Capacity ({result['capacity']})")
    
    ax4.set_xlabel('Cell Count / Channel', fontsize=11)
    ax4.set_ylabel('Latency (μs)', fontsize=11)
    ax4.set_title('Latency Distribution by Cell Count and Channel', fontsize=12)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Custom legend for channels
    legend_patches = [mpatches.Patch(color=colors[ch], alpha=0.6, label=ch) for ch in colors]
    ax4.legend(handles=legend_patches, loc='upper left')
    
    # ===== Plot 5: Memory Usage vs Cell Count =====
    ax5 = fig.add_subplot(gs[2, 0])
    memory_usage = [cells[c].get("memory_total_mb", 0) for c in cell_counts]
    ax5.bar(cell_counts, memory_usage, color='#9b59b6', alpha=0.7, edgecolor='black')
    ax5.axvline(x=result['capacity'], color='gray', linestyle='--', alpha=0.7)
    ax5.set_xlabel('Cell Count', fontsize=11)
    ax5.set_ylabel('Memory (MB)', fontsize=11)
    ax5.set_title('Memory Usage vs Cell Count', fontsize=12)
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.set_xticks(cell_counts)
    
    # ===== Plot 6: Latency Growth Rate =====
    ax6 = fig.add_subplot(gs[2, 1])
    
    for channel, color in colors.items():
        avg_key = f"{channel}_avg"
        latencies = [cells[c].get(avg_key, 0) for c in cell_counts]
        
        # Calculate growth rate (latency per cell)
        if len(cell_counts) > 1:
            growth_rates = []
            for i in range(1, len(latencies)):
                rate = (latencies[i] - latencies[i-1]) / (cell_counts[i] - cell_counts[i-1])
                growth_rates.append(rate)
            ax6.plot(cell_counts[1:], growth_rates, 'o-', label=channel, color=color, linewidth=2, markersize=6)
    
    ax6.axvline(x=result['capacity'], color='gray', linestyle='--', alpha=0.7)
    ax6.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax6.set_xlabel('Cell Count', fontsize=11)
    ax6.set_ylabel('Latency Growth Rate (μs/cell)', fontsize=11)
    ax6.set_title('Latency Growth Rate', fontsize=12)
    ax6.legend(loc='upper left')
    ax6.grid(True, alpha=0.3)
    ax6.set_xticks(cell_counts[1:])
    
    # ===== Plot 7: Summary Statistics Table =====
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')
    
    summary_text = f"""
╔══════════════════════════════════════╗
║         TEST CONFIGURATION           ║
╠══════════════════════════════════════╣
║  Target SM (DL/UL): {result['target_label']:>15} ║
║  GPU Frequency:     {str(result['freq']) + ' MHz':>15} ║
║  TDD Pattern:       {str(result['pattern']):>15} ║
║  Iterations:        {str(result['iterations']):>15} ║
║  Sweeps/Iter:       {str(result['sweeps']):>15} ║
║  Delay:             {str(result['delay']) + ' ns':>15} ║
╠══════════════════════════════════════╣
║           CAPACITY RESULTS           ║
╠══════════════════════════════════════╣
║  Overall Capacity:  {str(result['capacity']) + ' cells':>15} ║
║  PDSCH Capacity:    {str(result['capacity_pdsch']) + ' cells':>15} ║
║  PUSCH1 Capacity:   {str(result['capacity_pusch1']) + ' cells':>15} ║
║  PUSCH2 Capacity:   {str(result['capacity_pusch2']) + ' cells':>15} ║
╠══════════════════════════════════════╣
║       LATENCY @ CAPACITY ({result['capacity']:>2} cells)     ║
╠══════════════════════════════════════╣"""
    
    if result['capacity'] in cells:
        cap_cell = cells[result['capacity']]
        summary_text += f"""
║  PDSCH Avg:         {cap_cell.get('PDSCH_avg', 0):>12.1f} μs ║
║  PUSCH1 Avg:        {cap_cell.get('PUSCH1_avg', 0):>12.1f} μs ║
║  PUSCH2 Avg:        {cap_cell.get('PUSCH2_avg', 0):>12.1f} μs ║"""
    
    summary_text += """
╚══════════════════════════════════════╝"""
    
    ax7.text(0.1, 0.95, summary_text, transform=ax7.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_file = get_unique_filename(f"{output_prefix}_analysis", ".png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  📊 Saved: {output_file}")
    plt.close()


def plot_multi_target_comparison(results, output_prefix):
    """Generate comparison graphs across multiple target configurations."""
    if not HAS_MATPLOTLIB:
        print("matplotlib required for plotting")
        return
    
    # Sort results by frequency and target total SM
    results = sorted(results, key=lambda x: (x['freq'] if isinstance(x['freq'], (int, float)) else 0, x['target_total']))
    
    # Get unique frequencies for subtitle
    freqs = set(r['freq'] for r in results if r['freq'] != "?")
    freq_str = ", ".join([f"{f} MHz" for f in sorted(freqs)]) if freqs else "N/A"
    
    # Get unique patterns
    patterns = set(r['pattern'] for r in results if r['pattern'] != "?")
    pattern_str = ", ".join(sorted(patterns)) if patterns else "N/A"
    
    # Get iterations info
    iterations_set = set(r['iterations'] for r in results if r['iterations'] != "?")
    iterations_str = ", ".join([str(i) for i in sorted(iterations_set)]) if iterations_set else "N/A"
    
    # Get delay info
    delay_set = set(r['delay'] for r in results if r['delay'] != "?")
    delay_str = ", ".join([f"{d} ns" for d in sorted(delay_set)]) if delay_set else "N/A"
    
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(f'Target SM Allocation Sweep Analysis\n'
                 f'Frequency: {freq_str} | TDD Pattern: {pattern_str} | Iterations: {iterations_str} | Delay: {delay_str}', 
                 fontsize=16, fontweight='bold')
    
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Generate color gradient based on number of results
    cmap = plt.cm.viridis
    result_colors = [cmap(i / max(len(results) - 1, 1)) for i in range(len(results))]
    
    # ===== Plot 1: Capacity vs Target =====
    ax1 = fig.add_subplot(gs[0, 0])
    
    targets = [r['target_total'] for r in results]
    capacities = [r['capacity'] for r in results]
    # Include frequency in labels if multiple frequencies exist
    if len(freqs) > 1:
        target_labels = [f"{r['target_label']}\n@{r['freq']}MHz" for r in results]
    else:
        target_labels = [r['target_label'] for r in results]
    
    bars = ax1.bar(range(len(results)), capacities, color=result_colors, edgecolor='black', alpha=0.8)
    ax1.set_xticks(range(len(results)))
    ax1.set_xticklabels(target_labels, rotation=45, ha='right')
    ax1.set_xlabel('Target SM (DL/UL)', fontsize=11)
    ax1.set_ylabel('Capacity (cells)', fontsize=11)
    ax1.set_title('Capacity vs Target SM Allocation', fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, cap in zip(bars, capacities):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{cap}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # ===== Plot 2: Capacity Breakdown by Channel =====
    ax2 = fig.add_subplot(gs[0, 1])
    
    x = np.arange(len(results))
    width = 0.25
    
    pdsch_caps = [r['capacity_pdsch'] for r in results]
    pusch1_caps = [r['capacity_pusch1'] for r in results]
    pusch2_caps = [r['capacity_pusch2'] for r in results]
    
    ax2.bar(x - width, pdsch_caps, width, label='PDSCH', color='#2ecc71', alpha=0.8)
    ax2.bar(x, pusch1_caps, width, label='PUSCH1', color='#3498db', alpha=0.8)
    ax2.bar(x + width, pusch2_caps, width, label='PUSCH2', color='#e74c3c', alpha=0.8)
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(target_labels, rotation=45, ha='right')
    ax2.set_xlabel('Target SM (DL/UL)', fontsize=11)
    ax2.set_ylabel('Capacity (cells)', fontsize=11)
    ax2.set_title('Per-Channel Capacity vs Target', fontsize=12)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # ===== Plot 3: Efficiency (Capacity / Total SM) =====
    ax3 = fig.add_subplot(gs[0, 2])
    
    efficiency = [r['capacity'] / r['target_total'] if r['target_total'] > 0 else 0 for r in results]
    
    bars = ax3.bar(range(len(results)), efficiency, color=result_colors, edgecolor='black', alpha=0.8)
    ax3.set_xticks(range(len(results)))
    ax3.set_xticklabels(target_labels, rotation=45, ha='right')
    ax3.set_xlabel('Target SM (DL/UL)', fontsize=11)
    ax3.set_ylabel('Efficiency (cells/SM)', fontsize=11)
    ax3.set_title('SM Efficiency vs Target Allocation', fontsize=12)
    ax3.grid(True, alpha=0.3, axis='y')
    
    for bar, eff in zip(bars, efficiency):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{eff:.2f}', ha='center', va='bottom', fontsize=9)
    
    # ===== Plot 4: Latency vs Cell Count (Multi-Target Comparison) =====
    ax4 = fig.add_subplot(gs[1, 0])
    
    for idx, (r, color) in enumerate(zip(results, result_colors)):
        cells = r['cells']
        cell_counts = sorted(cells.keys())
        
        # Average across channels
        avg_latencies = []
        for c in cell_counts:
            cell = cells[c]
            lat_sum = 0
            lat_count = 0
            for ch in ['PDSCH', 'PUSCH1', 'PUSCH2']:
                if f'{ch}_avg' in cell:
                    lat_sum += cell[f'{ch}_avg']
                    lat_count += 1
            avg_latencies.append(lat_sum / lat_count if lat_count > 0 else 0)
        
        # Include frequency in legend if multiple frequencies
        if len(freqs) > 1:
            label = f"{r['target_label']} @{r['freq']}MHz"
        else:
            label = f"Target {r['target_label']}"
        
        ax4.plot(cell_counts, avg_latencies, 'o-', 
                label=label, 
                color=color, linewidth=2, markersize=5)
        
        # Mark capacity point
        ax4.axvline(x=r['capacity'], color=color, linestyle='--', alpha=0.5)
    
    ax4.set_xlabel('Cell Count', fontsize=11)
    ax4.set_ylabel('Average Latency (μs)', fontsize=11)
    ax4.set_title('Latency vs Cell Count (All Channels Avg)', fontsize=12)
    ax4.legend(loc='upper left', fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # ===== Plot 5: PUSCH2 Latency Comparison (Usually Bottleneck) =====
    ax5 = fig.add_subplot(gs[1, 1])
    
    for idx, (r, color) in enumerate(zip(results, result_colors)):
        cells = r['cells']
        cell_counts = sorted(cells.keys())
        
        pusch2_latencies = [cells[c].get('PUSCH2_avg', 0) for c in cell_counts]
        
        # Include frequency in legend if multiple frequencies
        if len(freqs) > 1:
            label = f"{r['target_label']} @{r['freq']}MHz"
        else:
            label = f"Target {r['target_label']}"
        
        ax5.plot(cell_counts, pusch2_latencies, 'o-', 
                label=label, 
                color=color, linewidth=2, markersize=5)
    
    ax5.set_xlabel('Cell Count', fontsize=11)
    ax5.set_ylabel('PUSCH2 Avg Latency (μs)', fontsize=11)
    ax5.set_title('PUSCH2 Latency vs Cell Count', fontsize=12)
    ax5.legend(loc='upper left', fontsize=9)
    ax5.grid(True, alpha=0.3)
    
    # ===== Plot 6: Latency at Capacity =====
    ax6 = fig.add_subplot(gs[1, 2])
    
    lat_at_cap = {'PDSCH': [], 'PUSCH1': [], 'PUSCH2': []}
    
    for r in results:
        cap = r['capacity']
        if cap in r['cells']:
            cell = r['cells'][cap]
            for ch in ['PDSCH', 'PUSCH1', 'PUSCH2']:
                lat_at_cap[ch].append(cell.get(f'{ch}_avg', 0))
        else:
            for ch in ['PDSCH', 'PUSCH1', 'PUSCH2']:
                lat_at_cap[ch].append(0)
    
    x = np.arange(len(results))
    width = 0.25
    
    ax6.bar(x - width, lat_at_cap['PDSCH'], width, label='PDSCH', color='#2ecc71', alpha=0.8)
    ax6.bar(x, lat_at_cap['PUSCH1'], width, label='PUSCH1', color='#3498db', alpha=0.8)
    ax6.bar(x + width, lat_at_cap['PUSCH2'], width, label='PUSCH2', color='#e74c3c', alpha=0.8)
    
    ax6.set_xticks(x)
    ax6.set_xticklabels(target_labels, rotation=45, ha='right')
    ax6.set_xlabel('Target SM (DL/UL)', fontsize=11)
    ax6.set_ylabel('Latency at Capacity (μs)', fontsize=11)
    ax6.set_title('Latency at Capacity Point', fontsize=12)
    ax6.legend(loc='upper left')
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_file = get_unique_filename(f"{output_prefix}_comparison", ".png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  📊 Saved: {output_file}")
    plt.close()


def print_summary_table(results):
    """Print a summary table of all results."""
    print("\n" + "=" * 120)
    print("TARGET SWEEP SUMMARY")
    print("=" * 120)
    print(f"{'Target':<12} {'Freq':>8} {'Iters':>8} {'Capacity':>10} {'PDSCH Cap':>10} {'PUSCH1 Cap':>11} {'PUSCH2 Cap':>11} {'Efficiency':>12}")
    print(f"{'(DL/UL)':<12} {'(MHz)':>8} {'':>8} {'(cells)':>10} {'(cells)':>10} {'(cells)':>11} {'(cells)':>11} {'(cells/SM)':>12}")
    print("-" * 120)
    
    for r in sorted(results, key=lambda x: (x['freq'] if isinstance(x['freq'], (int, float)) else 0, x['target_total'])):
        eff = r['capacity'] / r['target_total'] if r['target_total'] > 0 else 0
        freq_str = f"{r['freq']}" if r['freq'] != "?" else "N/A"
        iters_str = f"{r['iterations']}" if r['iterations'] != "?" else "N/A"
        print(f"{r['target_label']:<12} {freq_str:>8} {iters_str:>8} {r['capacity']:>10} {r['capacity_pdsch']:>10} "
              f"{r['capacity_pusch1']:>11} {r['capacity_pusch2']:>11} {eff:>12.3f}")
    
    print("=" * 120)
    
    # Find optimal target
    best = max(results, key=lambda x: x['capacity'])
    most_efficient = max(results, key=lambda x: x['capacity'] / x['target_total'] if x['target_total'] > 0 else 0)
    
    print(f"\n📈 Best Capacity:     Target {best['target_label']} → {best['capacity']} cells")
    
    eff = most_efficient['capacity'] / most_efficient['target_total'] if most_efficient['target_total'] > 0 else 0
    print(f"⚡ Most Efficient:    Target {most_efficient['target_label']} → {eff:.3f} cells/SM")


def save_combined_json(results, output_file):
    """Save combined analysis results to JSON."""
    output = {
        "analysis_timestamp": datetime.now().isoformat(),
        "total_files_analyzed": len(results),
        "summary": [],
        "detailed_results": []
    }
    
    for r in sorted(results, key=lambda x: x['target_total']):
        eff = r['capacity'] / r['target_total'] if r['target_total'] > 0 else 0
        
        summary_entry = {
            "target_label": r['target_label'],
            "target_dl": r['target_dl'],
            "target_ul": r['target_ul'],
            "target_total": r['target_total'],
            "capacity": r['capacity'],
            "capacity_pdsch": r['capacity_pdsch'],
            "capacity_pusch1": r['capacity_pusch1'],
            "capacity_pusch2": r['capacity_pusch2'],
            "efficiency_cells_per_sm": round(eff, 4),
            "freq_mhz": r['freq'],
            "iterations": r['iterations'],
            "sweeps": r['sweeps'],
            "pattern": r['pattern'],
            "source_file": r['filename']
        }
        output["summary"].append(summary_entry)
        
        # Detailed per-cell data
        cells_data = {}
        for cell_count, cell in r['cells'].items():
            cells_data[str(cell_count)] = {
                "ontime_pdsch": cell.get("ontime_pdsch", 0),
                "ontime_pusch1": cell.get("ontime_pusch1", 0),
                "ontime_pusch2": cell.get("ontime_pusch2", 0),
                "all_ontime": cell.get("all_ontime", False),
                "pdsch_avg_us": round(cell.get("PDSCH_avg", 0), 2),
                "pusch1_avg_us": round(cell.get("PUSCH1_avg", 0), 2),
                "pusch2_avg_us": round(cell.get("PUSCH2_avg", 0), 2),
                "memory_mb": round(cell.get("memory_total_mb", 0), 2)
            }
        
        output["detailed_results"].append({
            "target_label": r['target_label'],
            "cells": cells_data
        })
    
    # Generate unique filename
    base_name = os.path.splitext(output_file)[0]
    unique_output_file = get_unique_filename(base_name, ".json")
    
    with open(unique_output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"  💾 Saved: {unique_output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze measure.py results across different target SM allocations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze multiple specific files
    python3 analyze_target_sweep.py 020_020_*.json 040_040_*.json

    # Analyze all sweep files
    python3 analyze_target_sweep.py --pattern "*_sweep_*.json"

    # Single file detailed analysis
    python3 analyze_target_sweep.py --single 040_040_sweep_graphs_avg_F08.json
        """
    )
    
    parser.add_argument('files', nargs='*', help='JSON result files to analyze')
    parser.add_argument('--pattern', type=str, help='Glob pattern to find result files')
    parser.add_argument('--single', type=str, help='Single file for detailed analysis')
    parser.add_argument('--output', '-o', type=str, default='target_sweep',
                       help='Output prefix for generated files')
    parser.add_argument('--no-plot', action='store_true', help='Skip generating plots')
    
    args = parser.parse_args()
    
    # Collect files to analyze
    files = []
    
    if args.single:
        files = [args.single]
    elif args.pattern:
        files = glob.glob(args.pattern)
    elif args.files:
        for pattern in args.files:
            files.extend(glob.glob(pattern))
    else:
        # Default: find all sweep JSON files in current directory
        files = glob.glob("*_sweep_*.json")
    
    if not files:
        print("❌ No result files found!")
        print("   Provide files as arguments or use --pattern")
        sys.exit(1)
    
    # Remove duplicates and sort
    files = sorted(set(files))
    
    print("=" * 70)
    print("Target Sweep Analysis Tool")
    print("=" * 70)
    print(f"  Files to analyze: {len(files)}")
    
    # Parse all files
    results = []
    for filepath in files:
        try:
            print(f"  📄 Parsing: {filepath}")
            result = parse_result_file(filepath)
            results.append(result)
        except Exception as e:
            print(f"  ⚠️ Error parsing {filepath}: {e}")
    
    if not results:
        print("❌ No valid results to analyze!")
        sys.exit(1)
    
    print("=" * 70)
    
    # Print summary table
    print_summary_table(results)
    
    # Save combined JSON
    save_combined_json(results, f"{args.output}_analysis.json")
    
    # Generate plots
    if not args.no_plot and HAS_MATPLOTLIB:
        print("\n📊 Generating plots...")
        
        if args.single or len(results) == 1:
            # Single file detailed analysis
            plot_single_file_analysis(results[0], args.output)
        else:
            # Multi-target comparison
            plot_multi_target_comparison(results, args.output)
            
            # Also generate individual detailed plots
            for r in results:
                prefix = f"{args.output}_{r['target_dl']:02d}_{r['target_ul']:02d}"
                plot_single_file_analysis(r, prefix)
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
