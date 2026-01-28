#!/bin/bash
#
# Cell Sweep Test Automation Script
# 다양한 Cell 수에 대해 throughput 테스트를 자동으로 실행
#
# Usage:
#   ./run_cell_sweep.sh [options]
#
# Options:
#   --cells "1 2 4 8"     테스트할 cell 수 목록 (기본: 1 2 4 8 16)
#   --sms "40 40"         SM 할당 (기본: 40 40)
#   --iterations 1000     반복 횟수 (기본: 1000)
#   --delay 0             delay 값 (기본: 0)
#   --output_dir ./results 출력 디렉토리
#   --no-mps              MPS 없이 실행 (SM 할당 비활성화)
#   --freq 1500           GPU 클럭 주파수 (MHz, 기본: 1500)
#   --gpu 0               사용할 GPU ID (기본: 0)
#   --monitor             CPU/GPU 리소스 모니터링 활성화
#   --monitor-interval 1  모니터링 간격 (초, 기본: 1)

set -e

# 기본 설정
CELLS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)
SM_DL=40
SM_UL=40
ITERATIONS=1000
DELAY=0
OUTPUT_DIR=""  # 자동 생성됨
USE_MPS=true   # --no-mps 옵션으로 비활성화 가능
GPU_FREQ=1500  # GPU 클럭 주파수 (MHz)
GPU_ID=0       # GPU ID
ENABLE_MONITOR=false  # --monitor 옵션으로 활성화
MONITOR_INTERVAL=1    # 모니터링 간격 (초)
PERF_DIR="/workspace/aerial-cuda-accelerated-ran/testBenches/perf"
BUILD_DIR="/workspace/aerial-cuda-accelerated-ran/testBenches/build/cubb_gpu_test_bench"

# 인자 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        --cells)
            IFS=' ' read -r -a CELLS <<< "$2"
            shift 2
            ;;
        --sms)
            SM_DL=$(echo $2 | cut -d' ' -f1)
            SM_UL=$(echo $2 | cut -d' ' -f2)
            shift 2
            ;;
        --iterations)
            ITERATIONS=$2
            shift 2
            ;;
        --delay)
            DELAY=$2
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR=$2
            shift 2
            ;;
        --no-mps)
            USE_MPS=false
            shift
            ;;
        --freq)
            GPU_FREQ=$2
            shift 2
            ;;
        --gpu)
            GPU_ID=$2
            shift 2
            ;;
        --monitor)
            ENABLE_MONITOR=true
            shift
            ;;
        --monitor-interval)
            MONITOR_INTERVAL=$2
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# 시작 시간
START_TIME=$(date +%Y%m%d_%H%M%S)

# 출력 디렉토리 자동 생성 (Cell 범위, SM 포함)
CELL_MIN=${CELLS[0]}
CELL_MAX=${CELLS[-1]}
if [ -z "$OUTPUT_DIR" ]; then
    if [ "$USE_MPS" = true ]; then
        OUTPUT_DIR="${PERF_DIR}/cell_sweep_${CELL_MIN}-${CELL_MAX}cell_SM${SM_DL}_${SM_UL}_${START_TIME}"
    else
        OUTPUT_DIR="${PERF_DIR}/cell_sweep_${CELL_MIN}-${CELL_MAX}cell_noMPS_${START_TIME}"
    fi
fi

# 절대 경로로 변환
OUTPUT_DIR=$(realpath -m "$OUTPUT_DIR")
mkdir -p "$OUTPUT_DIR"
SUMMARY_FILE="$OUTPUT_DIR/sweep_summary.txt"

echo "============================================================"
echo "  Cell Sweep Test Automation"
echo "============================================================"
echo "  Cells to test: ${CELLS[*]}"
if [ "$USE_MPS" = true ]; then
    echo "  SM allocation: DL=$SM_DL, UL=$SM_UL (MPS enabled)"
else
    echo "  SM allocation: None (MPS disabled)"
fi
echo "  GPU: $GPU_ID, Frequency: $GPU_FREQ MHz"
echo "  Iterations: $ITERATIONS"
echo "  Delay: $DELAY μs"
echo "  Output dir: $OUTPUT_DIR"
if [ "$ENABLE_MONITOR" = true ]; then
    echo "  Resource Monitor: ENABLED (interval: ${MONITOR_INTERVAL}s)"
else
    echo "  Resource Monitor: disabled (use --monitor to enable)"
fi
echo "============================================================"
echo ""

# =============================================================================
# GPU 클럭 주파수 설정
# =============================================================================
echo ""
echo "Configuring GPU clock frequency..."

# 원래 GPU 클럭 저장
ORIG_GPU_FREQ=$(nvidia-smi -i $GPU_ID --query-gpu=clocks.current.graphics --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' || echo "")
if [ -n "$ORIG_GPU_FREQ" ]; then
    echo "  Original GPU clock: $ORIG_GPU_FREQ MHz"
else
    echo "  ⚠️  Could not read original GPU clock"
    ORIG_GPU_FREQ=""
fi

# GPU 클럭 고정
if nvidia-smi -i $GPU_ID -lgc $GPU_FREQ 2>/dev/null; then
    echo "  ✅ GPU clock locked at $GPU_FREQ MHz"
else
    echo "  ⚠️  Could not lock GPU clock (may require sudo or persistence mode)"
fi

# =============================================================================
# 리소스 모니터링 시작 (옵션)
# =============================================================================
MONITOR_DIR="$OUTPUT_DIR/monitor"
if [ "$ENABLE_MONITOR" = true ]; then
    echo "Starting resource monitoring..."
    mkdir -p "$MONITOR_DIR"
    cd "$PERF_DIR"
    chmod +x start_monitor.sh 2>/dev/null || true
    ./start_monitor.sh "$MONITOR_DIR" "$MONITOR_INTERVAL"
    echo ""
fi

# 결과 요약 파일 헤더
echo "Cell Sweep Test Summary" > "$SUMMARY_FILE"
echo "Generated: $(date)" >> "$SUMMARY_FILE"
echo "GPU: $GPU_ID | Freq: $GPU_FREQ MHz | SM: DL=$SM_DL, UL=$SM_UL | Iterations: $ITERATIONS | Delay: $DELAY" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "Cells | Pattern Time (μs) | Slots/sec | Throughput (Gbps) | RT Check" >> "$SUMMARY_FILE"
echo "------|-------------------|-----------|-------------------|----------" >> "$SUMMARY_FILE"

# 각 cell 수에 대해 테스트 실행
for CELL_COUNT in "${CELLS[@]}"; do
    CELL_STR=$(printf "%02d" $CELL_COUNT)
    VECTORS_FILE="$PERF_DIR/vectors-${CELL_STR}.yaml"
    OUTPUT_FILE="$OUTPUT_DIR/cubb_${CELL_STR}cell_SM${SM_DL}_${SM_UL}.txt"
    JSON_OUTPUT="$OUTPUT_DIR/throughput_${CELL_STR}cell_SM${SM_DL}_${SM_UL}"
    
    echo ""
    echo "────────────────────────────────────────────────────────────"
    echo "  Testing $CELL_COUNT cells..."
    echo "────────────────────────────────────────────────────────────"
    
    # vectors 파일 존재 확인
    if [ ! -f "$VECTORS_FILE" ]; then
        echo "  ⚠️  Warning: $VECTORS_FILE not found, skipping..."
        echo "$CELL_STR    | N/A               | N/A       | N/A               | SKIPPED" >> "$SUMMARY_FILE"
        continue
    fi
    
    echo "  Input: $VECTORS_FILE"
    echo "  Output: $OUTPUT_FILE"
    
    # cubb_gpu_test_bench 실행
    echo "  Running cubb_gpu_test_bench..."
    cd "$BUILD_DIR"
    
    if [ "$USE_MPS" = true ]; then
        # MPS 활성화 상태: SM 할당 사용
        ./cubb_gpu_test_bench \
            -i "$VECTORS_FILE" \
            -r "$ITERATIONS" \
            -w "$DELAY" \
            -u 5 \
            -d 0 \
            -m 1 \
            --U \
            --D \
            --M "$SM_DL,$SM_UL" \
            2>&1 | tee "$OUTPUT_FILE"
    else
        # MPS 비활성화 상태: SM 할당 없이 실행
        ./cubb_gpu_test_bench \
            -i "$VECTORS_FILE" \
            -r "$ITERATIONS" \
            -w "$DELAY" \
            -u 5 \
            -d 0 \
            -m 1 \
            --U \
            --D \
            2>&1 | tee "$OUTPUT_FILE"
    fi
    
    echo ""
    echo "  Analyzing results..."
    
    # 결과 분석
    cd "$PERF_DIR"
    python3 parse_cubb_output.py "$OUTPUT_FILE" -o "$JSON_OUTPUT" --no-plot 2>/dev/null || true
    
    # JSON에서 결과 추출
    if [ -f "${JSON_OUTPUT}_throughput_analysis.json" ]; then
        PATTERN_TIME=$(python3 -c "import json; d=json.load(open('${JSON_OUTPUT}_throughput_analysis.json')); print(f\"{d['summary']['avg_slot_pattern_time_us']:.2f}\")" 2>/dev/null || echo "N/A")
        SLOTS_SEC=$(python3 -c "import json; d=json.load(open('${JSON_OUTPUT}_throughput_analysis.json')); print(f\"{d['summary']['slots_per_second']:.2f}\")" 2>/dev/null || echo "N/A")
        THROUGHPUT=$(python3 -c "import json; d=json.load(open('${JSON_OUTPUT}_throughput_analysis.json')); print(f\"{d['summary']['estimated_throughput_gbps']:.2f}\")" 2>/dev/null || echo "N/A")
        RT_CHECK=$(python3 -c "import json; d=json.load(open('${JSON_OUTPUT}_throughput_analysis.json')); print('✅ PASS' if d['summary']['realtime_satisfied'] else '❌ FAIL')" 2>/dev/null || echo "N/A")
        
        echo "$CELL_STR    | $PATTERN_TIME         | $SLOTS_SEC    | $THROUGHPUT             | $RT_CHECK" >> "$SUMMARY_FILE"
        
        echo "  ✅ Done: Pattern=${PATTERN_TIME}μs, Slots/s=${SLOTS_SEC}, RT=${RT_CHECK}"
    else
        echo "$CELL_STR    | ERROR             | ERROR     | ERROR             | ERROR" >> "$SUMMARY_FILE"
        echo "  ❌ Analysis failed"
    fi
done

echo ""
echo "============================================================"
echo "  Cell Sweep Complete!"
echo "============================================================"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo "Summary: $SUMMARY_FILE"
echo ""

# 요약 출력
echo "────────────────────────────────────────────────────────────"
cat "$SUMMARY_FILE"
echo "────────────────────────────────────────────────────────────"

# 그래프 생성 (선택적)
echo ""
echo "Generating comparison graphs..."

export OUTPUT_DIR
export SM_DL
export SM_UL

python3 << 'EOF'
import os
import json
import glob

try:
    import matplotlib.pyplot as plt
    import numpy as np
    
    output_dir = os.environ.get('OUTPUT_DIR', './cell_sweep_results')
    sm_dl = os.environ.get('SM_DL', '40')
    sm_ul = os.environ.get('SM_UL', '40')
    json_files = sorted(glob.glob(f"{output_dir}/throughput_*cell_SM*_throughput_analysis.json"))
    
    if not json_files:
        print("No JSON files found for graphing")
        exit(0)
    
    cells = []
    pattern_times = []
    slots_per_sec = []
    realtime_satisfied = []
    
    for f in json_files:
        with open(f) as fp:
            data = json.load(fp)
        cells.append(data['setup']['streams'])
        pattern_times.append(data['summary']['avg_slot_pattern_time_us'])
        slots_per_sec.append(data['summary']['slots_per_second'])
        realtime_satisfied.append(data['summary']['realtime_satisfied'])
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Pattern Time vs Cells
    colors = ['green' if rt else 'red' for rt in realtime_satisfied]
    axes[0].bar(cells, pattern_times, color=colors, alpha=0.8, edgecolor='black')
    axes[0].axhline(y=5000, color='red', linestyle='--', linewidth=2, label='5ms requirement')
    axes[0].set_xlabel('Number of Cells')
    axes[0].set_ylabel('Pattern Time (μs)')
    axes[0].set_title('10-Slot Pattern Time vs Cell Count')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Slots/sec vs Cells
    axes[1].bar(cells, slots_per_sec, color='steelblue', alpha=0.8, edgecolor='black')
    axes[1].axhline(y=2000, color='red', linestyle='--', linewidth=2, label='2000 slots/sec requirement')
    axes[1].set_xlabel('Number of Cells')
    axes[1].set_ylabel('Slots per Second')
    axes[1].set_title('Throughput vs Cell Count')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Scaling efficiency
    if len(cells) > 1:
        efficiency = [s / cells[i] for i, s in enumerate(slots_per_sec)]
        axes[2].plot(cells, efficiency, 'o-', color='purple', linewidth=2, markersize=8)
        axes[2].set_xlabel('Number of Cells')
        axes[2].set_ylabel('Slots/sec per Cell')
        axes[2].set_title('Scaling Efficiency')
        axes[2].grid(True, alpha=0.3)
    
    plt.suptitle(f'Cell Sweep Results (SM: DL={sm_dl}, UL={sm_ul})', fontsize=14, y=1.02)
    plt.tight_layout()
    graph_file = f"{output_dir}/cell_sweep_SM{sm_dl}_{sm_ul}_comparison.png"
    plt.savefig(graph_file, dpi=150, bbox_inches='tight')
    print(f"📊 Graph saved: {graph_file}")
    
except ImportError:
    print("matplotlib not available, skipping graphs")
except Exception as e:
    print(f"Graph generation failed: {e}")
EOF

# Markdown 보고서 생성
echo ""
echo "Generating analysis report..."

export ITERATIONS
export DELAY
export START_TIME
export GPU_FREQ
export GPU_ID

python3 << 'MDEOF'
import os
import json
import glob
from datetime import datetime

output_dir = os.environ.get('OUTPUT_DIR', './cell_sweep_results')
sm_dl = os.environ.get('SM_DL', '40')
sm_ul = os.environ.get('SM_UL', '40')
iterations = os.environ.get('ITERATIONS', '1000')
delay = os.environ.get('DELAY', '0')
start_time = os.environ.get('START_TIME', datetime.now().strftime('%Y%m%d_%H%M%S'))
gpu_freq = os.environ.get('GPU_FREQ', '1500')
gpu_id = os.environ.get('GPU_ID', '0')

json_files = sorted(glob.glob(f"{output_dir}/throughput_*cell_SM*_throughput_analysis.json"))

if not json_files:
    print("No JSON files found for report generation")
    exit(0)

# 데이터 수집
results = []
for f in json_files:
    with open(f) as fp:
        data = json.load(fp)
    results.append({
        'cells': data['setup']['streams'],
        'sms': data['setup'].get('sms', f'{sm_dl}/{sm_ul}'),
        'pattern_time': data['summary']['avg_slot_pattern_time_us'],
        'slots_per_sec': data['summary']['slots_per_second'],
        'cell_slots_per_sec': data['summary']['cell_slots_per_second'],
        'throughput': data['summary']['estimated_throughput_gbps'],
        'realtime': data['summary']['realtime_satisfied'],
        'pusch1_avg': data['summary'].get('avg_pusch_time_us', 0),
        'pusch2_avg': data['summary'].get('avg_pusch2_time_us', 0),
        'pdsch_avg': data['summary'].get('avg_slot_time_us', 0),  # Per-slot average
        'file': os.path.basename(f)
    })

# 최대 실시간 만족 Cell 수 찾기
max_rt_cells = 0
for r in results:
    if r['realtime']:
        max_rt_cells = max(max_rt_cells, r['cells'])

# Markdown 생성
md_file = f"{output_dir}/cell_sweep_analysis_report.md"
with open(md_file, 'w') as f:
    f.write(f"# Cell Sweep Test Analysis Report\n\n")
    f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    f.write("## 📋 Test Configuration\n\n")
    f.write("| Parameter | Value |\n")
    f.write("|-----------|-------|\n")
    f.write(f"| GPU | {gpu_id} |\n")
    f.write(f"| GPU Frequency | {gpu_freq} MHz |\n")
    f.write(f"| SM Allocation | DL: {sm_dl}, UL: {sm_ul} |\n")
    f.write(f"| Iterations | {iterations} |\n")
    f.write(f"| Delay | {delay} μs |\n")
    f.write(f"| Cells Tested | {', '.join([str(r['cells']) for r in results])} |\n")
    f.write(f"| Total Tests | {len(results)} |\n\n")
    
    f.write("## 📊 Results Summary\n\n")
    f.write("| Cells | Pattern Time (μs) | Slots/sec | Cell-Slots/sec | Throughput (Gbps) | Real-time |\n")
    f.write("|:-----:|:-----------------:|:---------:|:--------------:|:-----------------:|:---------:|\n")
    
    for r in results:
        rt_status = "✅ PASS" if r['realtime'] else "❌ FAIL"
        f.write(f"| {r['cells']} | {r['pattern_time']:.2f} | {r['slots_per_sec']:.2f} | {r['cell_slots_per_sec']:.2f} | {r['throughput']:.2f} | {rt_status} |\n")
    
    f.write(f"\n## 🎯 Key Findings\n\n")
    f.write(f"### Maximum Real-time Capacity\n\n")
    if max_rt_cells > 0:
        f.write(f"- **Maximum cells meeting 5ms requirement:** {max_rt_cells} cells\n")
        f.write(f"- **Real-time requirement:** 10-slot pattern ≤ 5000μs (5ms)\n")
    else:
        f.write(f"- ⚠️ No cell configuration met the real-time requirement\n")
    
    f.write(f"\n### Latency Analysis\n\n")
    f.write("| Cells | PUSCH1 Avg (μs) | PUSCH2 Avg (μs) | PDSCH Avg (μs) |\n")
    f.write("|:-----:|:---------------:|:---------------:|:--------------:|\n")
    for r in results:
        f.write(f"| {r['cells']} | {r['pusch1_avg']:.2f} | {r['pusch2_avg']:.2f} | {r['pdsch_avg']:.2f} |\n")
    
    f.write(f"\n### Throughput Scaling\n\n")
    if len(results) > 1:
        first = results[0]
        last = results[-1]
        scaling_factor = last['throughput'] / first['throughput'] if first['throughput'] > 0 else 0
        cell_scaling = last['cells'] / first['cells']
        efficiency = (scaling_factor / cell_scaling) * 100 if cell_scaling > 0 else 0
        
        f.write(f"- **Cell scaling:** {first['cells']} → {last['cells']} cells ({cell_scaling:.1f}x)\n")
        f.write(f"- **Throughput scaling:** {first['throughput']:.2f} → {last['throughput']:.2f} Gbps ({scaling_factor:.1f}x)\n")
        f.write(f"- **Scaling efficiency:** {efficiency:.1f}%\n")
    
    f.write(f"\n## ⚠️ Real-time Compliance Check\n\n")
    f.write("### Per-Slot Requirement (500μs TTI)\n\n")
    f.write("| Cells | Avg Slot Time (μs) | 500μs Margin | Status |\n")
    f.write("|:-----:|:------------------:|:------------:|:------:|\n")
    for r in results:
        avg_slot = r['pattern_time'] / 10
        margin = 500 - avg_slot
        status = "✅" if margin >= 0 else "❌"
        f.write(f"| {r['cells']} | {avg_slot:.2f} | {margin:.2f} | {status} |\n")
    
    # 500μs 기준 최대 Cell 수 찾기
    max_500us_cells = 0
    for r in results:
        if r['pattern_time'] / 10 <= 500:
            max_500us_cells = max(max_500us_cells, r['cells'])
    
    f.write(f"\n**Maximum cells meeting 500μs TTI requirement:** {max_500us_cells} cells\n")
    
    f.write(f"\n## 📁 Generated Files\n\n")
    f.write("| File | Description |\n")
    f.write("|------|-------------|\n")
    for r in results:
        f.write(f"| `{r['file']}` | {r['cells']}-cell throughput analysis |\n")
    f.write(f"| `cell_sweep_SM{sm_dl}_{sm_ul}_comparison.png` | Comparison graphs |\n")
    f.write(f"| `sweep_summary.txt` | Text summary |\n")
    
    f.write(f"\n## 📈 Conclusion\n\n")
    if max_rt_cells > 0 and max_500us_cells > 0:
        f.write(f"With SM allocation of {sm_dl}/{sm_ul} (DL/UL):\n\n")
        f.write(f"1. **5ms Pattern Requirement:** Up to **{max_rt_cells} cells** can meet the 10-slot (5ms) deadline\n")
        f.write(f"2. **500μs TTI Requirement:** Up to **{max_500us_cells} cells** can meet the per-slot (500μs) deadline\n")
        if max_rt_cells != max_500us_cells:
            f.write(f"\n⚠️ **Note:** The 500μs per-slot requirement is stricter than the 5ms pattern requirement.\n")
    else:
        f.write(f"⚠️ Current SM allocation ({sm_dl}/{sm_ul}) may need adjustment for real-time compliance.\n")
    
    f.write(f"\n---\n*Report generated by Cell Sweep Test Automation Script*\n")

print(f"📄 Report saved: {md_file}")
MDEOF

# =============================================================================
# 리소스 모니터링 종료 및 분석 (옵션)
# =============================================================================
if [ "$ENABLE_MONITOR" = true ]; then
    echo ""
    echo "Stopping resource monitoring..."
    cd "$PERF_DIR"
    chmod +x stop_monitor.sh 2>/dev/null || true
    ./stop_monitor.sh "$MONITOR_DIR/monitor.pid" 2>/dev/null || true
    
    # 잠시 대기 (로그 flush)
    sleep 2
    
    echo ""
    echo "Analyzing resource usage..."
    python3 analyze_monitor.py "$MONITOR_DIR" --output "$OUTPUT_DIR/resource_analysis"
fi

# =============================================================================
# GPU 클럭 주파수 복원
# =============================================================================
echo ""
echo "Restoring GPU configuration..."
if [ -n "$ORIG_GPU_FREQ" ] && [ "$ORIG_GPU_FREQ" != "$GPU_FREQ" ]; then
    if nvidia-smi -i $GPU_ID -lgc $ORIG_GPU_FREQ 2>/dev/null; then
        echo "  ✅ GPU clock restored to $ORIG_GPU_FREQ MHz"
    else
        # 클럭 잠금 해제 시도
        nvidia-smi -i $GPU_ID -rgc 2>/dev/null && echo "  ✅ GPU clock lock released" || true
    fi
else
    # 클럭 잠금 해제
    nvidia-smi -i $GPU_ID -rgc 2>/dev/null && echo "  ✅ GPU clock lock released" || true
fi

echo ""
echo "✅ All done!"

# 최종 출력 파일 목록
echo ""
echo "📁 Generated Files:"
echo "────────────────────────────────────────────────────────────"
ls -la "$OUTPUT_DIR"/*.txt "$OUTPUT_DIR"/*.png "$OUTPUT_DIR"/*.md 2>/dev/null || true
if [ "$ENABLE_MONITOR" = true ]; then
    echo ""
    echo "📊 Monitor Files:"
    ls -la "$MONITOR_DIR"/*.csv "$OUTPUT_DIR"/resource_*.* 2>/dev/null || true
fi
echo "────────────────────────────────────────────────────────────"
