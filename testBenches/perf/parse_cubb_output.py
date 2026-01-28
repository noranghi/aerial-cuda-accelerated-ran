#!/usr/bin/env python3
"""
Parse cubb_gpu_test_bench output and generate throughput/latency graphs.

Usage:
    # From terminal output file:
    ./cubb_gpu_test_bench ... | tee output.txt
    python3 parse_cubb_output.py output.txt
    
    # Or pipe directly:
    ./cubb_gpu_test_bench ... 2>&1 | python3 parse_cubb_output.py -
"""

import re
import sys
import json
import argparse
from datetime import datetime

try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, plots will be skipped")


def parse_cubb_output(text):
    """Parse cubb_gpu_test_bench output text."""
    results = {
        "parse_time": datetime.now().isoformat(),
        "setup": {},
        "patterns": []
    }
    
    # Parse setup info
    sm_match = re.search(r'requested SMs for context \[0\].*?: (\d+)', text)
    if sm_match:
        results["setup"]["pdsch_sms"] = int(sm_match.group(1))
    
    sm_match = re.search(r'requested SMs for context \[1\].*?: (\d+)', text)
    if sm_match:
        results["setup"]["pusch_sms"] = int(sm_match.group(1))
    
    contexts_match = re.search(r'(\d+) CUDA contexts.*?(\d+) cuPHY.*?(\d+) cuMAC', text)
    if contexts_match:
        results["setup"]["total_contexts"] = int(contexts_match.group(1))
        results["setup"]["cuphy_contexts"] = int(contexts_match.group(2))
        results["setup"]["cumac_contexts"] = int(contexts_match.group(3))
    
    streams_match = re.search(r'Runs (\d+) stream\(s\) in parallel', text)
    if streams_match:
        results["setup"]["streams"] = int(streams_match.group(1))
    
    # Parse each slot pattern
    pattern_blocks = re.split(r'-{40,}\nSlot pattern # (\d+)', text)
    
    for i in range(1, len(pattern_blocks), 2):
        pattern_num = int(pattern_blocks[i])
        block = pattern_blocks[i + 1] if i + 1 < len(pattern_blocks) else ""
        
        pattern_data = {
            "pattern_num": pattern_num,
            "slot_pattern_time_us": None,
            "pusch_time_us": None,
            "pusch2_time_us": None,
            "pdsch_slots": []
        }
        
        # Parse slot pattern time
        slot_time_match = re.search(r'average slot pattern run time: ([\d.]+) us', block)
        if slot_time_match:
            pattern_data["slot_pattern_time_us"] = float(slot_time_match.group(1))
        
        # Parse PUSCH time
        pusch_match = re.search(r'Average PUSCH run time: ([\d.]+) us from ([\d.]+)', block)
        if pusch_match:
            pattern_data["pusch_time_us"] = float(pusch_match.group(1))
            pattern_data["pusch_start_us"] = float(pusch_match.group(2))
        
        # Parse PUSCH2 time
        pusch2_match = re.search(r'Average PUSCH2 run time: ([\d.]+) us from ([\d.]+)', block)
        if pusch2_match:
            pattern_data["pusch2_time_us"] = float(pusch2_match.group(1))
            pattern_data["pusch2_start_us"] = float(pusch2_match.group(2))
        
        # Parse PDSCH slots
        pdsch_matches = re.findall(r'Slot # (\d+): average PDSCH run time: ([\d.]+) us from ([\d.]+)', block)
        for match in pdsch_matches:
            pattern_data["pdsch_slots"].append({
                "slot": int(match[0]),
                "time_us": float(match[1]),
                "start_us": float(match[2])
            })
        
        results["patterns"].append(pattern_data)
    
    # Calculate summary statistics
    if results["patterns"]:
        pattern_times = [p["slot_pattern_time_us"] for p in results["patterns"] if p["slot_pattern_time_us"]]
        pusch_times = [p["pusch_time_us"] for p in results["patterns"] if p["pusch_time_us"]]
        pusch2_times = [p["pusch2_time_us"] for p in results["patterns"] if p["pusch2_time_us"]]
        
        results["summary"] = {
            "num_patterns": len(results["patterns"]),
            "avg_slot_pattern_time_us": np.mean(pattern_times) if pattern_times else 0,
            "avg_pusch_time_us": np.mean(pusch_times) if pusch_times else 0,
            "avg_pusch2_time_us": np.mean(pusch2_times) if pusch2_times else 0,
            "min_slot_pattern_time_us": min(pattern_times) if pattern_times else 0,
            "max_slot_pattern_time_us": max(pattern_times) if pattern_times else 0,
            "std_slot_pattern_time_us": np.std(pattern_times) if pattern_times else 0,
        }
        
        # Calculate throughput
        avg_pattern_time_ms = results["summary"]["avg_slot_pattern_time_us"] / 1000
        slots_per_pattern = 10  # DDDSUUDDDD = 10 slots
        
        results["summary"]["patterns_per_second"] = 1000 / avg_pattern_time_ms if avg_pattern_time_ms > 0 else 0
        results["summary"]["slots_per_second"] = results["summary"]["patterns_per_second"] * slots_per_pattern
        
        # Real-time requirement check (5ms for 10-slot pattern)
        realtime_requirement_us = 5000  # 10 slots × 0.5ms = 5ms
        results["summary"]["realtime_requirement_us"] = realtime_requirement_us
        results["summary"]["realtime_margin_us"] = realtime_requirement_us - results["summary"]["avg_slot_pattern_time_us"]
        results["summary"]["realtime_margin_percent"] = (results["summary"]["realtime_margin_us"] / realtime_requirement_us) * 100
        results["summary"]["realtime_satisfied"] = results["summary"]["avg_slot_pattern_time_us"] <= realtime_requirement_us
        
        # Cell throughput (assuming cell count from streams)
        num_cells = results["setup"].get("streams", 8)
        results["summary"]["num_cells"] = num_cells
        results["summary"]["cell_slots_per_second"] = results["summary"]["slots_per_second"] * num_cells
        
        # Estimated data throughput (rough estimate)
        # Assuming ~150 Mbps per cell for typical 5G config
        estimated_mbps_per_cell = 150
        results["summary"]["estimated_throughput_mbps"] = num_cells * estimated_mbps_per_cell
        results["summary"]["estimated_throughput_gbps"] = results["summary"]["estimated_throughput_mbps"] / 1000
        
        # Per-slot throughput
        avg_slot_time_us = results["summary"]["avg_slot_pattern_time_us"] / slots_per_pattern
        results["summary"]["avg_slot_time_us"] = avg_slot_time_us
        results["summary"]["slot_realtime_margin_us"] = 500 - avg_slot_time_us  # 500μs per slot
        results["summary"]["slot_realtime_satisfied"] = avg_slot_time_us <= 500
    
    return results


def print_summary(results):
    """Print summary table."""
    print("\n" + "=" * 90)
    print(f"{'cubb_gpu_test_bench THROUGHPUT/LATENCY ANALYSIS':^90}")
    print("=" * 90)
    
    setup = results.get("setup", {})
    print(f"\n📋 Setup:")
    print(f"   PDSCH SMs: {setup.get('pdsch_sms', 'N/A')}")
    print(f"   PUSCH SMs: {setup.get('pusch_sms', 'N/A')}")
    print(f"   Streams (Cells): {setup.get('streams', 'N/A')}")
    
    summary = results.get("summary", {})
    print(f"\n📊 Latency Summary (averaged over {summary.get('num_patterns', 0)} patterns):")
    print(f"   {'Metric':<30} {'Value':>15} {'500μs Check':>15}")
    print(f"   {'-'*60}")
    
    metrics = [
        ("Slot Pattern Time (10 slots)", summary.get("avg_slot_pattern_time_us"), 5000),
        ("Average per Slot", summary.get("avg_slot_time_us"), 500),
        ("PUSCH1 Time", summary.get("avg_pusch_time_us"), 500),
        ("PUSCH2 Time", summary.get("avg_pusch2_time_us"), 500),
    ]
    
    for name, value, deadline in metrics:
        if value:
            status = "✅ PASS" if value <= deadline else "❌ FAIL"
            print(f"   {name:<30} {value:>12.2f} μs {status:>15}")
    
    print(f"\n" + "=" * 90)
    print(f"{'📈 THROUGHPUT ANALYSIS':^90}")
    print("=" * 90)
    
    print(f"\n   {'Metric':<40} {'Value':>20} {'Unit':>15}")
    print(f"   {'-'*75}")
    print(f"   {'Patterns per Second':<40} {summary.get('patterns_per_second', 0):>20.2f} {'patterns/sec':>15}")
    print(f"   {'Slots per Second':<40} {summary.get('slots_per_second', 0):>20.2f} {'slots/sec':>15}")
    print(f"   {'Cell-Slots per Second':<40} {summary.get('cell_slots_per_second', 0):>20.2f} {'cell-slots/sec':>15}")
    print(f"   {'Estimated Data Throughput':<40} {summary.get('estimated_throughput_gbps', 0):>20.2f} {'Gbps':>15}")
    
    print(f"\n   📍 Real-time Requirement Check (10-slot pattern = 5ms):")
    realtime_status = "✅ SATISFIED" if summary.get('realtime_satisfied', False) else "❌ NOT SATISFIED"
    print(f"   Required Time:  5000 μs")
    print(f"   Actual Time:    {summary.get('avg_slot_pattern_time_us', 0):.2f} μs")
    print(f"   Margin:         {summary.get('realtime_margin_us', 0):.2f} μs ({summary.get('realtime_margin_percent', 0):.1f}%)")
    print(f"   Status:         {realtime_status}")
    
    print(f"\n   📍 Per-Slot Real-time Check (1 slot = 500μs TTI):")
    slot_status = "✅ SATISFIED" if summary.get('slot_realtime_satisfied', False) else "❌ NOT SATISFIED"
    print(f"   Required Time:  500 μs")
    print(f"   Actual Time:    {summary.get('avg_slot_time_us', 0):.2f} μs")
    print(f"   Margin:         {summary.get('slot_realtime_margin_us', 0):.2f} μs")
    print(f"   Status:         {slot_status}")
    
    print("\n" + "=" * 90)


def plot_results(results, output_prefix):
    """Generate visualization plots."""
    if not HAS_MATPLOTLIB:
        return
    
    patterns = results["patterns"]
    if not patterns:
        print("No patterns to plot")
        return
    
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("cubb_gpu_test_bench Performance Analysis\n"
                 f"PDSCH SMs: {results['setup'].get('pdsch_sms', 'N/A')}, "
                 f"PUSCH SMs: {results['setup'].get('pusch_sms', 'N/A')}, "
                 f"Cells: {results['setup'].get('streams', 'N/A')}",
                 fontsize=14, fontweight='bold')
    
    summary = results.get("summary", {})
    
    # Plot 1: Slot Pattern Time per Pattern (top-left)
    ax1 = fig.add_subplot(3, 2, 1)
    pattern_nums = [p["pattern_num"] for p in patterns]
    pattern_times = [p["slot_pattern_time_us"] for p in patterns]
    
    ax1.bar(pattern_nums, pattern_times, color='steelblue', alpha=0.8, edgecolor='black')
    ax1.axhline(y=5000, color='red', linestyle='--', linewidth=2, label='5ms real-time req')
    ax1.axhline(y=np.mean(pattern_times), color='green', linestyle=':', linewidth=2, label=f'Avg: {np.mean(pattern_times):.1f}μs')
    ax1.set_xlabel('Pattern #', fontsize=11)
    ax1.set_ylabel('Pattern Time (μs)', fontsize=11)
    ax1.set_title('10-Slot Pattern Run Time', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: PUSCH1 vs PUSCH2 Time (top-right)
    ax2 = fig.add_subplot(3, 2, 2)
    pusch1_times = [p["pusch_time_us"] for p in patterns if p["pusch_time_us"]]
    pusch2_times = [p["pusch2_time_us"] for p in patterns if p["pusch2_time_us"]]
    
    x = np.arange(len(pusch1_times))
    width = 0.35
    
    ax2.bar(x - width/2, pusch1_times, width, label='PUSCH1', color='#3498db', alpha=0.8)
    ax2.bar(x + width/2, pusch2_times, width, label='PUSCH2', color='#e74c3c', alpha=0.8)
    ax2.axhline(y=500, color='red', linestyle='--', linewidth=2, label='500μs deadline')
    ax2.set_xlabel('Pattern #', fontsize=11)
    ax2.set_ylabel('Latency (μs)', fontsize=11)
    ax2.set_title('PUSCH1 vs PUSCH2 Latency', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(x)
    
    # Plot 3: PDSCH Latency per Slot (middle-left)
    ax3 = fig.add_subplot(3, 2, 3)
    if patterns[0]["pdsch_slots"]:
        slots = [s["slot"] for s in patterns[0]["pdsch_slots"]]
        times = [s["time_us"] for s in patterns[0]["pdsch_slots"]]
        
        colors = ['#27ae60' if t <= 500 else '#c0392b' for t in times]
        ax3.bar(slots, times, color=colors, alpha=0.8, edgecolor='black')
        ax3.axhline(y=500, color='red', linestyle='--', linewidth=2, label='500μs deadline')
        ax3.set_xlabel('Slot #', fontsize=11)
        ax3.set_ylabel('PDSCH Latency (μs)', fontsize=11)
        ax3.set_title('PDSCH Latency per Slot (Pattern 0)', fontsize=12)
        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        for i, (slot, time) in enumerate(zip(slots, times)):
            ax3.text(slot, time + 50, f'{time:.0f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 4: Throughput Bar Chart (middle-right)
    ax4 = fig.add_subplot(3, 2, 4)
    
    throughput_labels = ['Patterns/sec', 'Slots/sec\n(÷100)', 'Cell-Slots/sec\n(÷1000)']
    throughput_values = [
        summary.get('patterns_per_second', 0),
        summary.get('slots_per_second', 0) / 100,
        summary.get('cell_slots_per_second', 0) / 1000
    ]
    
    colors = ['#2ecc71', '#3498db', '#9b59b6']
    bars = ax4.bar(throughput_labels, throughput_values, color=colors, alpha=0.8, edgecolor='black')
    
    for bar, val, orig in zip(bars, throughput_values, [
        summary.get('patterns_per_second', 0),
        summary.get('slots_per_second', 0),
        summary.get('cell_slots_per_second', 0)
    ]):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{orig:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax4.set_ylabel('Value (scaled)', fontsize=11)
    ax4.set_title('Throughput Metrics', fontsize=12)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Real-time Margin Gauge (bottom-left)
    ax5 = fig.add_subplot(3, 2, 5)
    
    categories = ['10-Slot Pattern\n(5ms req)', 'Per-Slot Avg\n(500μs req)']
    actual_times = [summary.get('avg_slot_pattern_time_us', 0), summary.get('avg_slot_time_us', 0)]
    requirements = [5000, 500]
    margins = [req - actual for req, actual in zip(requirements, actual_times)]
    margin_pcts = [(m / req) * 100 for m, req in zip(margins, requirements)]
    
    x_pos = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax5.bar(x_pos - width/2, actual_times, width, label='Actual Time', color='#e74c3c', alpha=0.8)
    bars2 = ax5.bar(x_pos + width/2, requirements, width, label='Requirement', color='#95a5a6', alpha=0.5)
    
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(categories)
    ax5.set_ylabel('Time (μs)', fontsize=11)
    ax5.set_title('Real-time Requirement vs Actual', fontsize=12)
    ax5.legend(loc='upper right')
    ax5.grid(True, alpha=0.3, axis='y')
    
    for bar, margin_pct, satisfied in zip(bars1, margin_pcts, [summary.get('realtime_satisfied'), summary.get('slot_realtime_satisfied')]):
        status = "✅" if satisfied else "❌"
        color = 'green' if satisfied else 'red'
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                f'{status} {margin_pct:.1f}%\nmargin', ha='center', va='bottom', fontsize=9, color=color)
    
    # Plot 6: Summary Box (bottom-right)
    ax6 = fig.add_subplot(3, 2, 6)
    ax6.axis('off')
    
    realtime_status = "✅ SATISFIED" if summary.get('realtime_satisfied', False) else "❌ NOT MET"
    slot_status = "✅ SATISFIED" if summary.get('slot_realtime_satisfied', False) else "❌ NOT MET"
    
    summary_text = f"""
╔═══════════════════════════════════════════════════════════════╗
║              THROUGHPUT & LATENCY SUMMARY                     ║
╠═══════════════════════════════════════════════════════════════╣
║                                                               ║
║  📊 Latency (averaged over {summary.get('num_patterns', 0):>2} patterns):                 ║
║     10-Slot Pattern:  {summary.get('avg_slot_pattern_time_us', 0):>8.2f} μs                      ║
║     Per-Slot Avg:     {summary.get('avg_slot_time_us', 0):>8.2f} μs                      ║
║     PUSCH1:           {summary.get('avg_pusch_time_us', 0):>8.2f} μs                      ║
║     PUSCH2:           {summary.get('avg_pusch2_time_us', 0):>8.2f} μs                      ║
║                                                               ║
║  📈 Throughput:                                               ║
║     Patterns/sec:     {summary.get('patterns_per_second', 0):>8.2f}                         ║
║     Slots/sec:        {summary.get('slots_per_second', 0):>8.2f}                         ║
║     Cell-Slots/sec:   {summary.get('cell_slots_per_second', 0):>8.2f}                         ║
║     Est. Throughput:  {summary.get('estimated_throughput_gbps', 0):>8.2f} Gbps                    ║
║                                                               ║
║  ⏱️  Real-time Check:                                          ║
║     Pattern (5ms):    {realtime_status:<15}                        ║
║     Slot (500μs):     {slot_status:<15}                        ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
"""
    
    ax6.text(0.02, 0.98, summary_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    
    output_file = f"{output_prefix}_throughput_analysis.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n📊 Plot saved: {output_file}")
    plt.close()


def save_json(results, output_prefix):
    """Save results to JSON file."""
    output_file = f"{output_prefix}_throughput_analysis.json"
    
    # Convert numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        return obj
    
    serializable_results = convert_to_serializable(results)
    
    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"💾 JSON saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Parse cubb_gpu_test_bench output and generate throughput/latency analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # From file:
    python3 parse_cubb_output.py output.txt
    
    # From pipe:
    ./cubb_gpu_test_bench ... 2>&1 | python3 parse_cubb_output.py -
    
    # With custom output prefix:
    python3 parse_cubb_output.py output.txt -o my_analysis
        """
    )
    
    parser.add_argument('input', help='Input file (use - for stdin)')
    parser.add_argument('-o', '--output', default='cubb_output',
                       help='Output prefix for generated files (default: cubb_output)')
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip generating plots')
    parser.add_argument('--no-json', action='store_true',
                       help='Skip saving JSON')
    
    args = parser.parse_args()
    
    # Read input
    if args.input == '-':
        text = sys.stdin.read()
    else:
        with open(args.input, 'r') as f:
            text = f.read()
    
    # Parse
    results = parse_cubb_output(text)
    
    # Print summary
    print_summary(results)
    
    # Generate outputs
    if not args.no_plot and HAS_MATPLOTLIB:
        plot_results(results, args.output)
    
    if not args.no_json:
        save_json(results, args.output)


if __name__ == "__main__":
    main()
