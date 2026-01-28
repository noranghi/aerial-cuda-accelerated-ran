#!/bin/bash
# F14 Massive MIMO 시스템 모니터링 스크립트
# 사용법: ./monitor_system.sh <output_dir> <interval_seconds>

OUTPUT_DIR="${1:-./monitoring_results}"
INTERVAL="${2:-1}"

mkdir -p "$OUTPUT_DIR"

# 타임스탬프
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_PREFIX="${OUTPUT_DIR}/monitor_${TIMESTAMP}"

echo "=== F14 Massive MIMO 시스템 모니터링 시작 ==="
echo "출력 디렉토리: $OUTPUT_DIR"
echo "샘플링 간격: ${INTERVAL}초"
echo "로그 파일 prefix: $LOG_PREFIX"
echo ""

# GPU 모니터링 (백그라운드)
echo "GPU 모니터링 시작..."
nvidia-smi --query-gpu=timestamp,name,pci.bus_id,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,pcie.link.gen.current,pcie.link.width.current \
    --format=csv -l "$INTERVAL" > "${LOG_PREFIX}_gpu.csv" 2>&1 &
GPU_PID=$!

# GPU 상세 정보 (PCIe 대역폭 포함)
nvidia-smi dmon -s pucvmet -d "$INTERVAL" > "${LOG_PREFIX}_gpu_dmon.txt" 2>&1 &
DMON_PID=$!

# CPU 모니터링 (백그라운드)
echo "CPU 모니터링 시작..."
(while true; do
    echo "=== $(date '+%Y-%m-%d %H:%M:%S') ==="
    top -bn1 | head -20
    echo ""
    sleep "$INTERVAL"
done) > "${LOG_PREFIX}_cpu.txt" 2>&1 &
CPU_PID=$!

# 메모리 모니터링 (백그라운드)
echo "메모리 모니터링 시작..."
(while true; do
    echo "=== $(date '+%Y-%m-%d %H:%M:%S') ==="
    free -h
    echo ""
    sleep "$INTERVAL"
done) > "${LOG_PREFIX}_memory.txt" 2>&1 &
MEM_PID=$!

# vmstat 모니터링
echo "vmstat 모니터링 시작..."
vmstat "$INTERVAL" > "${LOG_PREFIX}_vmstat.txt" 2>&1 &
VMSTAT_PID=$!

# PID 저장
echo "$GPU_PID $DMON_PID $CPU_PID $MEM_PID $VMSTAT_PID" > "${LOG_PREFIX}_pids.txt"

echo ""
echo "모니터링 프로세스 PID:"
echo "  GPU (nvidia-smi): $GPU_PID"
echo "  GPU (dmon): $DMON_PID"
echo "  CPU (top): $CPU_PID"
echo "  Memory (free): $MEM_PID"
echo "  vmstat: $VMSTAT_PID"
echo ""
echo "모니터링 중... 중지하려면: kill $GPU_PID $DMON_PID $CPU_PID $MEM_PID $VMSTAT_PID"
echo "또는: ./stop_monitor.sh ${LOG_PREFIX}_pids.txt"

# PID 파일 경로 출력
echo "${LOG_PREFIX}_pids.txt"


