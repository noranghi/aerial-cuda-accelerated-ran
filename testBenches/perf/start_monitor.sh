#!/bin/bash
# =============================================================================
# GPU/CPU 리소스 모니터링 시작 스크립트
# =============================================================================
# Usage:
#   ./start_monitor.sh [output_dir] [interval_sec]
#
# Arguments:
#   output_dir   - 로그 저장 디렉토리 (기본: ./monitor_logs)
#   interval_sec - 모니터링 간격 (기본: 1초)
#
# Output Files:
#   - gpu_monitor.csv   : GPU 사용량 (utilization, memory, power, temp)
#   - cpu_monitor.csv   : CPU 사용량 (각 코어별 %)
#   - memory_monitor.csv: 시스템 메모리 사용량
#   - monitor.pid       : 모니터링 프로세스 PID (종료 시 사용)
#
# Stop:
#   ./stop_monitor.sh <output_dir>/monitor.pid
# =============================================================================

set -e

# 기본 설정
OUTPUT_DIR="${1:-./monitor_logs}"
INTERVAL="${2:-1}"  # 초 단위

# 타임스탬프
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 출력 디렉토리 생성
mkdir -p "$OUTPUT_DIR"

# 파일 경로
GPU_LOG="$OUTPUT_DIR/gpu_monitor.csv"
CPU_LOG="$OUTPUT_DIR/cpu_monitor.csv"
MEM_LOG="$OUTPUT_DIR/memory_monitor.csv"
PID_FILE="$OUTPUT_DIR/monitor.pid"

echo "============================================================"
echo "  Resource Monitoring Started"
echo "============================================================"
echo "  Output Dir  : $OUTPUT_DIR"
echo "  Interval    : ${INTERVAL}s"
echo "  GPU Log     : $GPU_LOG"
echo "  CPU Log     : $CPU_LOG"
echo "  Memory Log  : $MEM_LOG"
echo "  PID File    : $PID_FILE"
echo "============================================================"

# 기존 모니터링 종료
if [ -f "$PID_FILE" ]; then
    echo "Stopping existing monitors..."
    ./stop_monitor.sh "$PID_FILE" 2>/dev/null || true
fi

# GPU 모니터링 헤더
echo "timestamp,gpu_index,gpu_util_pct,mem_util_pct,mem_used_mb,mem_total_mb,power_w,temp_c,sm_clock_mhz,mem_clock_mhz" > "$GPU_LOG"

# CPU 모니터링 헤더
echo "timestamp,cpu_id,user_pct,system_pct,idle_pct,iowait_pct" > "$CPU_LOG"

# 메모리 모니터링 헤더
echo "timestamp,total_mb,used_mb,free_mb,available_mb,buff_cache_mb,used_pct" > "$MEM_LOG"

# PID 저장용
PIDS=""

# =============================================================================
# GPU 모니터링 (백그라운드)
# =============================================================================
(
    while true; do
        TS=$(date +%Y-%m-%d_%H:%M:%S.%3N)
        
        # nvidia-smi 쿼리
        nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm,clocks.mem \
            --format=csv,noheader,nounits 2>/dev/null | while read -r line; do
            echo "$TS,$line" >> "$GPU_LOG"
        done
        
        sleep "$INTERVAL"
    done
) &
GPU_PID=$!
PIDS="$GPU_PID"
echo "GPU Monitor PID: $GPU_PID"

# =============================================================================
# CPU 모니터링 (백그라운드)
# =============================================================================
(
    while true; do
        TS=$(date +%Y-%m-%d_%H:%M:%S.%3N)
        
        # mpstat이 있으면 사용, 없으면 /proc/stat 파싱
        if command -v mpstat &> /dev/null; then
            mpstat -P ALL 1 1 2>/dev/null | tail -n +4 | head -n -1 | while read -r line; do
                CPU_ID=$(echo "$line" | awk '{print $2}')
                USER=$(echo "$line" | awk '{print $3}')
                SYS=$(echo "$line" | awk '{print $5}')
                IDLE=$(echo "$line" | awk '{print $12}')
                IOWAIT=$(echo "$line" | awk '{print $6}')
                if [ "$CPU_ID" != "" ] && [ "$CPU_ID" != "CPU" ]; then
                    echo "$TS,$CPU_ID,$USER,$SYS,$IDLE,$IOWAIT" >> "$CPU_LOG"
                fi
            done
        else
            # Fallback: /proc/stat 사용
            grep "^cpu" /proc/stat | while read -r line; do
                CPU_ID=$(echo "$line" | awk '{print $1}')
                USER=$(echo "$line" | awk '{print $2}')
                NICE=$(echo "$line" | awk '{print $3}')
                SYS=$(echo "$line" | awk '{print $4}')
                IDLE=$(echo "$line" | awk '{print $5}')
                IOWAIT=$(echo "$line" | awk '{print $6}')
                TOTAL=$((USER + NICE + SYS + IDLE + IOWAIT))
                if [ "$TOTAL" -gt 0 ]; then
                    USER_PCT=$(echo "scale=2; $USER * 100 / $TOTAL" | bc)
                    SYS_PCT=$(echo "scale=2; $SYS * 100 / $TOTAL" | bc)
                    IDLE_PCT=$(echo "scale=2; $IDLE * 100 / $TOTAL" | bc)
                    IOWAIT_PCT=$(echo "scale=2; $IOWAIT * 100 / $TOTAL" | bc)
                    echo "$TS,$CPU_ID,$USER_PCT,$SYS_PCT,$IDLE_PCT,$IOWAIT_PCT" >> "$CPU_LOG"
                fi
            done
        fi
        
        sleep "$INTERVAL"
    done
) &
CPU_PID=$!
PIDS="$PIDS $CPU_PID"
echo "CPU Monitor PID: $CPU_PID"

# =============================================================================
# 메모리 모니터링 (백그라운드)
# =============================================================================
(
    while true; do
        TS=$(date +%Y-%m-%d_%H:%M:%S.%3N)
        
        # free 명령어 사용
        MEM_INFO=$(free -m | grep "^Mem:")
        TOTAL=$(echo "$MEM_INFO" | awk '{print $2}')
        USED=$(echo "$MEM_INFO" | awk '{print $3}')
        FREE=$(echo "$MEM_INFO" | awk '{print $4}')
        AVAILABLE=$(echo "$MEM_INFO" | awk '{print $7}')
        BUFF_CACHE=$(echo "$MEM_INFO" | awk '{print $6}')
        
        if [ "$TOTAL" -gt 0 ]; then
            USED_PCT=$(echo "scale=2; $USED * 100 / $TOTAL" | bc)
        else
            USED_PCT=0
        fi
        
        echo "$TS,$TOTAL,$USED,$FREE,$AVAILABLE,$BUFF_CACHE,$USED_PCT" >> "$MEM_LOG"
        
        sleep "$INTERVAL"
    done
) &
MEM_PID=$!
PIDS="$PIDS $MEM_PID"
echo "Memory Monitor PID: $MEM_PID"

# PID 파일 저장
echo "$PIDS" > "$PID_FILE"

echo ""
echo "✅ Monitoring started!"
echo "   To stop: ./stop_monitor.sh $PID_FILE"
echo ""
