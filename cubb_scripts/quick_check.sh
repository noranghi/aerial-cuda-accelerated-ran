#!/bin/bash
# 빠른 진행 상황 확인 (원라이너 버전)

INPUT_DIR="${1:-../5GModel/aerial_mcore/examples/GPU_test_input}"
OUTPUT_DIR="${2:-$INPUT_DIR}"
OUT_FILE="$OUTPUT_DIR/out.txt"

PROCESSED=$(wc -l < "$OUT_FILE" 2>/dev/null || echo "0")
TOTAL=$(find "$INPUT_DIR" -name "*_gNB_FAPI_*.h5" -type f 2>/dev/null | wc -l)
SKIP=$(find "$INPUT_DIR" -name "*_gNB_FAPI_*.h5" -type f 2>/dev/null | grep -E "PUSCH_HARQ|F01|F08|F13|F14|CP" | wc -l)
TARGET=$((TOTAL - SKIP))
REMAINING=$((TARGET - PROCESSED))

if [ "$TARGET" -gt 0 ]; then
    PERCENT=$(awk "BEGIN {printf \"%.1f\", ($PROCESSED / $TARGET) * 100}")
    PID=$(pgrep -f "auto_lp.py" | head -1)
    
    if [ -n "$PID" ] && [ "$PROCESSED" -gt 0 ]; then
        ELAPSED=$(ps -o etime= -p "$PID" 2>/dev/null | awk -F: '{if (NF==2) print $1*60+$2; else if (NF==3) print $1*3600+$2*60+$3; else print 0}')
        if [ "$ELAPSED" -gt 0 ]; then
            RATE=$(awk "BEGIN {printf \"%.4f\", $PROCESSED / $ELAPSED}")
            REMAINING_SEC=$(awk "BEGIN {printf \"%.0f\", $REMAINING / $RATE}")
            HOURS=$((REMAINING_SEC / 3600))
            MINS=$(((REMAINING_SEC % 3600) / 60))
            
            echo "진행: $PROCESSED/$TARGET ($PERCENT%) | 남은 작업: $REMAINING | 예상 시간: ${HOURS}시간 ${MINS}분"
            
            CURRENT_HOUR=$(date +%H | sed 's/^0//')
            CURRENT_HOUR=${CURRENT_HOUR:-0}
            if [ "$HOURS" -lt $((24 - CURRENT_HOUR)) ]; then
                echo "✓ 오늘 안에 완료 예상"
            else
                echo "✗ 오늘 안에 완료 어려울 수 있음"
            fi
        else
            echo "진행: $PROCESSED/$TARGET ($PERCENT%) | 남은 작업: $REMAINING"
        fi
    else
        echo "진행: $PROCESSED/$TARGET ($PERCENT%) | 남은 작업: $REMAINING"
    fi
else
    echo "파일을 찾을 수 없습니다"
fi
