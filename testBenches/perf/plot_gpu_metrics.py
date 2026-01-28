#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
GPU Metrics Visualization Script

Generates plots from gpu_metrics JSON files created by measure_with_gpu_metrics.py

Usage:
    python3 plot_gpu_metrics.py gpu_metrics_040_040_F08_8to16.json
    python3 plot_gpu_metrics.py gpu_metrics_040_040_F08_8to16.json --output my_plot.png
"""

import json
import argparse
import sys
import os

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Error: matplotlib and numpy are required. Install with:")
    print("  pip install matplotlib numpy")
    sys.exit(1)


def load_metrics(json_file):
    """Load metrics from JSON file."""
    with open(json_file, 'r') as f:
        return json.load(f)


def plot_metrics(data, output_file=None, show_plot=True):
    """Generate visualization plots from GPU metrics data."""
    
    metrics = data.get("gpu_metrics", {})
    time_series = data.get("time_series", {})
    
    if not time_series or not time_series.get("time_sec"):
        print("Error: No time series data found in JSON file")
        return None
    
    # Extract time series data
    time_sec = np.array(time_series.get("time_sec", []))
    gpu_util = np.array(time_series.get("gpu_util_pct", []))
    mem_util = np.array(time_series.get("mem_util_pct", []))
    power = np.array(time_series.get("power_w", []))
    sm_clock = np.array(time_series.get("sm_clock_mhz", []))
    
    # Convert time to minutes for readability
    time_min = time_sec / 60
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Title with summary info
    gpu_type = metrics.get("gpu_type", "Unknown")
    duration = data.get("duration_sec", 0)
    timestamp = data.get("timestamp", "")[:19]
    
    fig.suptitle(f'GPU Performance Metrics\n{gpu_type} | Duration: {duration/60:.1f} min | {timestamp}', 
                 fontsize=14, fontweight='bold')
    
    # Color scheme
    colors = {
        'gpu_util': '#2196F3',      # Blue
        'mem_util': '#9C27B0',      # Purple
        'power': '#F44336',         # Red
        'sm_clock': '#FF9800',      # Orange
        'gflops': '#4CAF50',        # Green
        'bandwidth': '#00BCD4',     # Cyan
    }
    
    # ============ Plot 1: GPU Utilization over time ============
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.fill_between(time_min, gpu_util, alpha=0.3, color=colors['gpu_util'])
    ax1.plot(time_min, gpu_util, color=colors['gpu_util'], linewidth=0.5, alpha=0.7)
    
    # Add moving average
    window = min(50, len(gpu_util) // 10) if len(gpu_util) > 100 else 5
    if window > 1:
        gpu_util_smooth = np.convolve(gpu_util, np.ones(window)/window, mode='valid')
        time_smooth = time_min[window-1:]
        ax1.plot(time_smooth, gpu_util_smooth, color='darkblue', linewidth=2, label=f'Moving Avg ({window})')
    
    ax1.axhline(y=metrics.get("avg_gpu_util_pct", 0), color='red', linestyle='--', 
                linewidth=2, label=f'Average: {metrics.get("avg_gpu_util_pct", 0):.1f}%')
    ax1.set_xlabel('Time (minutes)')
    ax1.set_ylabel('GPU Utilization (%)')
    ax1.set_title('GPU Utilization Over Time')
    ax1.set_ylim(0, 105)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # ============ Plot 2: Power Consumption over time ============
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.fill_between(time_min, power, alpha=0.3, color=colors['power'])
    ax2.plot(time_min, power, color=colors['power'], linewidth=0.5, alpha=0.7)
    
    if window > 1 and len(power) > window:
        power_smooth = np.convolve(power, np.ones(window)/window, mode='valid')
        ax2.plot(time_smooth, power_smooth, color='darkred', linewidth=2, label=f'Moving Avg')
    
    ax2.axhline(y=metrics.get("avg_power_w", 0), color='blue', linestyle='--', 
                linewidth=2, label=f'Average: {metrics.get("avg_power_w", 0):.1f}W')
    ax2.set_xlabel('Time (minutes)')
    ax2.set_ylabel('Power (W)')
    ax2.set_title('Power Consumption Over Time')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # ============ Plot 3: GPU Utilization Histogram ============
    ax3 = fig.add_subplot(2, 3, 3)
    
    # Create histogram
    bins = np.arange(0, 105, 5)
    counts, edges, patches = ax3.hist(gpu_util, bins=bins, color=colors['gpu_util'], 
                                       alpha=0.7, edgecolor='white', linewidth=0.5)
    
    # Color bars based on utilization level
    for i, patch in enumerate(patches):
        if edges[i] >= 90:
            patch.set_facecolor('#4CAF50')  # Green for high utilization
        elif edges[i] >= 50:
            patch.set_facecolor('#FF9800')  # Orange for medium
        else:
            patch.set_facecolor('#F44336')  # Red for low
    
    ax3.axvline(x=metrics.get("avg_gpu_util_pct", 0), color='blue', linestyle='--', 
                linewidth=2, label=f'Avg: {metrics.get("avg_gpu_util_pct", 0):.1f}%')
    ax3.set_xlabel('GPU Utilization (%)')
    ax3.set_ylabel('Sample Count')
    ax3.set_title('GPU Utilization Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # ============ Plot 4: Summary Metrics Bar Chart ============
    ax4 = fig.add_subplot(2, 3, 4)
    
    # Performance metrics
    categories = ['Avg GPU\nUtil %', 'Max GPU\nUtil %', 'Avg Power\n(W)', 'Peak Power\n(W)']
    values = [
        metrics.get("avg_gpu_util_pct", 0),
        metrics.get("max_gpu_util_pct", 0),
        metrics.get("avg_power_w", 0),
        metrics.get("max_power_w", 0)
    ]
    bar_colors = [colors['gpu_util'], colors['gpu_util'], colors['power'], colors['power']]
    
    bars = ax4.bar(categories, values, color=bar_colors, alpha=0.7, edgecolor='white', linewidth=2)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{val:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax4.set_ylabel('Value')
    ax4.set_title('Performance Summary')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # ============ Plot 5: Estimated GFLOPS & Bandwidth ============
    ax5 = fig.add_subplot(2, 3, 5)
    
    est_gflops = metrics.get("estimated_gflops", 0)
    peak_gflops = metrics.get("theoretical_peak_gflops", 66908)
    est_bw = metrics.get("estimated_bandwidth_gbps", 0)
    peak_bw = metrics.get("theoretical_peak_bandwidth_gbps", 3350)
    
    # Normalize to percentage for comparison
    gflops_pct = (est_gflops / peak_gflops * 100) if peak_gflops > 0 else 0
    bw_pct = (est_bw / peak_bw * 100) if peak_bw > 0 else 0
    
    x = np.arange(2)
    width = 0.35
    
    achieved = [gflops_pct, bw_pct]
    remaining = [100 - gflops_pct, 100 - bw_pct]
    
    bars1 = ax5.bar(x, achieved, width, label='Achieved', color=[colors['gflops'], colors['bandwidth']], alpha=0.8)
    bars2 = ax5.bar(x, remaining, width, bottom=achieved, label='Unused', color='lightgray', alpha=0.5)
    
    ax5.set_ylabel('% of Theoretical Peak')
    ax5.set_title('Performance vs Theoretical Peak')
    ax5.set_xticks(x)
    ax5.set_xticklabels([f'GFLOPS\n({est_gflops/1000:.1f}K)', f'Bandwidth\n({est_bw:.0f} GB/s)'])
    ax5.set_ylim(0, 110)
    ax5.legend(loc='upper right')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels
    for bar, pct in zip(bars1, achieved):
        ax5.text(bar.get_x() + bar.get_width()/2., bar.get_height()/2,
                f'{pct:.1f}%', ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    # ============ Plot 6: Key Statistics Text Box ============
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    stats_text = f"""
╔══════════════════════════════════════════════╗
║           GPU Performance Summary            ║
╠══════════════════════════════════════════════╣
║  GPU Type:        {gpu_type:<25} ║
║  Duration:        {duration/60:.1f} minutes{' '*17}║
║  Samples:         {metrics.get('sample_count', 0):<25} ║
╠══════════════════════════════════════════════╣
║  GPU Utilization                             ║
║    Average:       {metrics.get('avg_gpu_util_pct', 0):>6.1f}%                    ║
║    Peak:          {metrics.get('max_gpu_util_pct', 0):>6.1f}%                    ║
╠══════════════════════════════════════════════╣
║  Power Consumption                           ║
║    Average:       {metrics.get('avg_power_w', 0):>6.1f} W                   ║
║    Peak:          {metrics.get('max_power_w', 0):>6.1f} W                   ║
╠══════════════════════════════════════════════╣
║  Estimated Performance                       ║
║    GFLOPS:        {est_gflops:>10.1f} ({gflops_pct:>5.1f}% peak) ║
║    Bandwidth:     {est_bw:>7.1f} GB/s ({bw_pct:>5.1f}% peak) ║
╠══════════════════════════════════════════════╣
║  Theoretical Peak ({gpu_type})              ║
║    GFLOPS:        {peak_gflops:>10.0f}                  ║
║    Bandwidth:     {peak_bw:>7.0f} GB/s               ║
╚══════════════════════════════════════════════╝
"""
    
    ax6.text(0.5, 0.5, stats_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='center', horizontalalignment='center',
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='#f5f5f5', alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Save plot
    if output_file is None:
        output_file = os.path.splitext(sys.argv[1] if len(sys.argv) > 1 else "gpu_metrics")[0] + "_plot.png"
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✅ Plot saved to: {output_file}")
    
    if show_plot:
        plt.show()
    
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Plot GPU metrics from JSON file")
    parser.add_argument("json_file", help="GPU metrics JSON file")
    parser.add_argument("--output", "-o", help="Output PNG file (default: <input>_plot.png)")
    parser.add_argument("--no-show", action="store_true", help="Don't display plot, just save")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.json_file):
        print(f"Error: File not found: {args.json_file}")
        sys.exit(1)
    
    print(f"Loading metrics from: {args.json_file}")
    data = load_metrics(args.json_file)
    
    output_file = args.output
    if output_file is None:
        output_file = os.path.splitext(args.json_file)[0] + "_plot.png"
    
    plot_metrics(data, output_file, show_plot=not args.no_show)


if __name__ == "__main__":
    main()
