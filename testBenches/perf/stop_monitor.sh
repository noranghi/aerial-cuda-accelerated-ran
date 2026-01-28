#!/bin/bash
# =============================================================================
# Resource Monitor Stop Script
# =============================================================================
# Usage: ./stop_monitor.sh <pid_file>
# =============================================================================

PID_FILE="$1"

if [ -z "$PID_FILE" ]; then
    echo "Usage: ./stop_monitor.sh <pid_file>"
    echo "Example: ./stop_monitor.sh ./monitor_logs/monitor.pid"
    exit 1
fi

if [ -f "$PID_FILE" ]; then
    PIDS=$(cat "$PID_FILE")
    echo "Stopping monitor processes: $PIDS"
    
    for PID in $PIDS; do
        if ps -p $PID > /dev/null 2>&1; then
            kill $PID 2>/dev/null
            echo "  - Stopped PID: $PID"
        else
            echo "  - PID $PID already stopped"
        fi
    done
    
    rm -f "$PID_FILE"
    echo "✅ Monitoring stopped"
else
    echo "⚠️ PID file not found: $PID_FILE"
    echo "   Monitoring may not be running or already stopped."
fi


