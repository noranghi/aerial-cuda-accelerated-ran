#!/bin/bash

# 진행 상황 확인 및 예상 완료 시간 계산 스크립트

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 기본값 설정
INPUT_DIR="../5GModel/aerial_mcore/examples/GPU_test_input"
OUTPUT_DIR="../5GModel/aerial_mcore/examples/GPU_test_input"
PROCESS_NAME="auto_lp.py"

# 인자 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input_dir)
            INPUT_DIR="$2"
            shift 2
            ;;
        -o|--output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "사용법: $0 [-i INPUT_DIR] [-o OUTPUT_DIR]"
            echo "  -i, --input_dir    입력 디렉토리 (기본값: $INPUT_DIR)"
            echo "  -o, --output_dir   출력 디렉토리 (기본값: $OUTPUT_DIR)"
            exit 0
            ;;
        *)
            echo "알 수 없는 옵션: $1"
            exit 1
            ;;
    esac
done

# 절대 경로로 변환
INPUT_DIR=$(realpath "$INPUT_DIR" 2>/dev/null || echo "$INPUT_DIR")
OUTPUT_DIR=$(realpath "$OUTPUT_DIR" 2>/dev/null || echo "$OUTPUT_DIR")

OUT_FILE="$OUTPUT_DIR/out.txt"

echo -e "${BLUE}=== 채널 생성 진행 상황 확인 ===${NC}\n"

# 1. 프로세스 확인
PID=$(pgrep -f "$PROCESS_NAME" | head -1)
if [ -z "$PID" ]; then
    echo -e "${YELLOW}경고: $PROCESS_NAME 프로세스를 찾을 수 없습니다.${NC}"
    echo "프로세스가 실행 중이 아니거나 이미 완료되었을 수 있습니다."
    echo ""
else
    # 프로세스 실행 시간 확인
    START_TIME=$(ps -o lstart= -p "$PID" 2>/dev/null)
    ELAPSED=$(ps -o etime= -p "$PID" 2>/dev/null | tr -d ' ')
    echo -e "${GREEN}프로세스 상태:${NC} 실행 중 (PID: $PID)"
    echo -e "${GREEN}시작 시간:${NC} $START_TIME"
    echo -e "${GREEN}경과 시간:${NC} $ELAPSED"
    echo ""
fi

# 2. 현재까지 처리된 파일 수 확인
if [ -f "$OUT_FILE" ]; then
    PROCESSED_COUNT=$(wc -l < "$OUT_FILE" 2>/dev/null || echo "0")
    echo -e "${GREEN}처리 완료:${NC} $PROCESSED_COUNT 개"
else
    PROCESSED_COUNT=0
    echo -e "${YELLOW}진행 로그 파일을 찾을 수 없습니다: $OUT_FILE${NC}"
    echo -e "${YELLOW}처리 완료:${NC} 0 개 (예상)"
fi

# 3. 전체 파일 수 확인
if [ -d "$INPUT_DIR" ]; then
    TOTAL_FILES=$(find "$INPUT_DIR" -name "*_gNB_FAPI_*.h5" -type f 2>/dev/null | wc -l)
    
    # 스킵될 파일 패턴 확인
    SKIP_PATTERN="PUSCH_HARQ|F01|F08|F13|F14|CP"
    SKIP_COUNT=$(find "$INPUT_DIR" -name "*_gNB_FAPI_*.h5" -type f 2>/dev/null | grep -E "$SKIP_PATTERN" | wc -l)
    
    # 실제 처리 대상 파일 수
    TARGET_FILES=$((TOTAL_FILES - SKIP_COUNT))
    
    echo -e "${GREEN}전체 파일:${NC} $TOTAL_FILES 개"
    echo -e "${YELLOW}스킵 파일:${NC} $SKIP_COUNT 개"
    echo -e "${GREEN}처리 대상:${NC} $TARGET_FILES 개"
else
    echo -e "${RED}오류: 입력 디렉토리를 찾을 수 없습니다: $INPUT_DIR${NC}"
    exit 1
fi

# 4. 진행률 계산
if [ "$TARGET_FILES" -gt 0 ]; then
    PROGRESS_PERCENT=$(awk "BEGIN {printf \"%.2f\", ($PROCESSED_COUNT / $TARGET_FILES) * 100}")
    REMAINING=$((TARGET_FILES - PROCESSED_COUNT))
    
    echo ""
    echo -e "${BLUE}=== 진행 상황 ===${NC}"
    echo -e "${GREEN}진행률:${NC} $PROGRESS_PERCENT% ($PROCESSED_COUNT / $TARGET_FILES)"
    echo -e "${YELLOW}남은 작업:${NC} $REMAINING 개"
    
    # 5. 처리 속도 및 예상 완료 시간 계산
    if [ -n "$PID" ] && [ "$PROCESSED_COUNT" -gt 0 ]; then
        # 경과 시간을 초로 변환
        ELAPSED_SEC=$(ps -o etime= -p "$PID" 2>/dev/null | awk -F: '{if (NF==2) print $1*60+$2; else if (NF==3) print $1*3600+$2*60+$3; else print 0}')
        
        if [ "$ELAPSED_SEC" -gt 0 ]; then
            # 처리 속도 (파일/초)
            RATE=$(awk "BEGIN {printf \"%.4f\", $PROCESSED_COUNT / $ELAPSED_SEC}")
            # 처리 속도 (파일/시간)
            RATE_PER_HOUR=$(awk "BEGIN {printf \"%.2f\", $PROCESSED_COUNT / ($ELAPSED_SEC / 3600)}")
            
            echo ""
            echo -e "${BLUE}=== 처리 속도 ===${NC}"
            echo -e "${GREEN}처리 속도:${NC} $RATE 파일/초 (약 $RATE_PER_HOUR 파일/시간)"
            
            # 예상 완료 시간 계산
            if [ "$REMAINING" -gt 0 ] && [ "$RATE" != "0" ]; then
                REMAINING_SEC=$(awk "BEGIN {printf \"%.0f\", $REMAINING / $RATE}")
                REMAINING_HOUR=$((REMAINING_SEC / 3600))
                REMAINING_MIN=$(((REMAINING_SEC % 3600) / 60))
                REMAINING_SEC_REMAIN=$((REMAINING_SEC % 60))
                
                # 현재 시간에 남은 시간 추가
                if command -v date >/dev/null 2>&1; then
                    ESTIMATED_COMPLETION=$(date -d "+$REMAINING_SEC seconds" "+%Y-%m-%d %H:%M:%S" 2>/dev/null || \
                        date -v+${REMAINING_SEC}S "+%Y-%m-%d %H:%M:%S" 2>/dev/null || \
                        echo "계산 불가")
                else
                    ESTIMATED_COMPLETION="계산 불가"
                fi
                
                echo ""
                echo -e "${BLUE}=== 예상 완료 시간 ===${NC}"
                if [ "$REMAINING_HOUR" -gt 0 ]; then
                    echo -e "${YELLOW}남은 시간:${NC} 약 ${REMAINING_HOUR}시간 ${REMAINING_MIN}분"
                else
                    echo -e "${YELLOW}남은 시간:${NC} 약 ${REMAINING_MIN}분 ${REMAINING_SEC_REMAIN}초"
                fi
                echo -e "${GREEN}예상 완료:${NC} $ESTIMATED_COMPLETION"
                
                # 오늘 안에 끝날지 확인
                CURRENT_HOUR=$(date +%H 2>/dev/null || echo "0")
                CURRENT_HOUR=${CURRENT_HOUR#0}  # leading zero 제거
                CURRENT_HOUR=${CURRENT_HOUR:-0}
                
                if [ "$REMAINING_HOUR" -lt $((24 - CURRENT_HOUR)) ]; then
                    echo -e "${GREEN}✓ 오늘 안에 완료될 것으로 예상됩니다!${NC}"
                else
                    DAYS_NEEDED=$((REMAINING_HOUR / 24 + 1))
                    echo -e "${RED}✗ 오늘 안에 완료되지 않을 것으로 예상됩니다.${NC}"
                    echo -e "${YELLOW}  약 ${DAYS_NEEDED}일 정도 더 필요할 수 있습니다.${NC}"
                fi
            fi
        fi
    else
        echo ""
        echo -e "${YELLOW}처리 속도를 계산할 수 없습니다 (프로세스가 실행 중이 아니거나 아직 처리된 파일이 없음)${NC}"
    fi
else
    echo -e "${RED}오류: 처리 대상 파일 수를 확인할 수 없습니다.${NC}"
fi

echo ""
