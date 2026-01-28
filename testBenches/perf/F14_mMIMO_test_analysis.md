# F14 Massive MIMO 테스트 분석 보고서

## 테스트 개요

| 항목 | 값 |
|------|-----|
| **테스트 날짜** | 2026-01-26 |
| **Use Case** | F14 (64T64R Massive MIMO) |
| **TDD 패턴** | `dddsuudddd_mMIMO` (15슬롯) |
| **GPU** | NVIDIA H100 NVL |
| **SM 할당** | 60 (DL) + 60 (UL) |
| **테스트 셀 수** | 1 ~ 6 셀 |
| **결과** | **5셀 성공** (Cell capacity: 05+00) |

---

## 1. MIMO 구성 분석

### 1.1 안테나 구성

| 구분 | 값 | 설명 |
|------|-----|------|
| **안테나 수** | 64T64R | 64 Transmit, 64 Receive |
| **실제 테스트 벡터** | 16x16 | 16 Layers × 16 Antennas |
| **빔포밍** | Digital Beamforming | DLBFW/ULBFW 사용 |

### 1.2 채널별 MIMO 레이어

| 채널 | MIMO 구성 | 의미 |
|------|-----------|------|
| **PDSCH (DL)** | 16×16 | 16 layers (최대 16 UE 동시 전송) |
| **PUSCH (UL)** | 8×16 | 8 layers × 16 antennas |
| **BFW** | 16×16 | 16 beams × 16 antennas |
| **SRS** | 8×16 | 8 layers × 16 antennas |

---

## 2. UE (사용자) 수 분석

### 2.1 Peak Traffic 기준 (F14-PP-00)

| 방향 | MIMO | 레이어 수 | **예상 최대 UE** |
|------|------|-----------|------------------|
| **Downlink (PDSCH)** | 16×16 | 16 layers | **최대 16 UE/셀** |
| **Uplink (PUSCH)** | 8×16 | 8 layers | **최대 8 UE/셀** |
| **PUCCH** | 1×4 | 1 layer | 제어 채널 |
| **PDCCH** | 4×4 | - | 제어 채널 |

### 2.2 5셀 성공 시 총 UE 추정

```
┌────────────────────────────────────────────────────────┐
│              5셀 기준 동시 서비스 가능 UE               │
├────────────────────────────────────────────────────────┤
│  Downlink: 5 셀 × 16 UE = 최대 80 UE 동시 전송         │
│  Uplink:   5 셀 × 8 UE  = 최대 40 UE 동시 수신         │
└────────────────────────────────────────────────────────┘
```

---

## 3. 테스트 벡터 상세

### 3.1 사용된 테스트 벡터 (F14-PP-00)

| 채널 | 파일명 | PRB | QAM |
|------|--------|-----|-----|
| **PDSCH** | TV_cuphy_F14-DS-01_slot0_MIMO16x16_PRB273_DataSyms11_qam256.h5 | 273 | 256-QAM |
| **PUSCH** | TV_cuphy_F14-US-01_snrdb40.00_MIMO8x16_PRB272_DataSyms12_qam256.h5 | 272 | 256-QAM |
| **DLBFW** | TV_cuphy_F14-BW-01_slot0_MIMO16x16_PRB273.h5 | 273 | - |
| **ULBFW** | TV_cuphy_F14-BW-01_slot0_MIMO16x16_PRB273.h5 | 273 | - |
| **SRS** | TV_cuphy_F14-SR-01_snrdb40.00_MIMO8x16_PRB272.h5 | 272 | - |
| **PRACH** | TV_cuphy_F14-RA-01.h5 | - | - |
| **PDCCH** | TV_cuphy_F14-DC-01_PRB273.h5 | 273 | - |
| **PUCCH** | TV_cuphy_F14-UC-01_PRB273.h5 | 273 | - |

### 3.2 대역폭 및 처리량

| 항목 | 값 |
|------|-----|
| **PRB (Physical Resource Block)** | 273 PRB |
| **대역폭** | 100 MHz (FR1 n78) |
| **Subcarrier Spacing** | 30 kHz |
| **변조 방식** | 256-QAM (최대) |
| **Data Symbols (DL)** | 11 symbols/slot |
| **Data Symbols (UL)** | 12 symbols/slot |

---

## 4. TDD 패턴 분석 (dddsuudddd_mMIMO)

### 4.1 15슬롯 패턴 구조

```
슬롯:    0   1   2   3   4   5   6   7   8   9  10  11  12  13  14
        ─────────────────────────────────────────────────────────────
타입:    D   D   D   S   U   U   D   D   D   D   D   D   D   S   U
        ─────────────────────────────────────────────────────────────
        └─── DL ───┘   └ UL ┘   └────────── DL ──────────┘   └ UL ┘
```

### 4.2 채널별 슬롯 활성화

| 채널 | 슬롯 0-14 활성화 패턴 | 총 활성 슬롯 |
|------|----------------------|-------------|
| **PDSCH** | `1 1 1 0 0 0 1 1 1 1 1 1 1 0 0` | 10 슬롯 |
| **PDCCH** | `1 1 1 1 0 0 1 1 1 1 1 1 1 1 0` | 12 슬롯 |
| **CSI-RS** | `0 0 0 0 0 0 1 1 1 1 1 1 0 0 0` | 6 슬롯 |
| **PBCH/SSB** | `1 1 1 1 0 0 0 0 0 0 0 0 0 0 0` | 4 슬롯 |
| **MAC** | `1 1 1 1 1 1 1 1 1 1 1 1 1 1 1` | 15 슬롯 |

### 4.3 DL/UL 비율

| 구분 | 슬롯 수 | 비율 |
|------|---------|------|
| **Downlink (D)** | 10 슬롯 | 66.7% |
| **Special (S)** | 2 슬롯 | 13.3% |
| **Uplink (U)** | 3 슬롯 | 20.0% |
| **합계** | 15 슬롯 | 100% |

---

## 5. 레이턴시 제약 조건

### 5.1 Massive MIMO 패턴 레이턴시 요구사항

| 채널 | 제한 시간 (μs) | 용도 |
|------|---------------|------|
| **DLBFW** | 250 | DL 빔포밍 가중치 |
| **PDSCH** | 300 | DL 데이터 |
| **PDCCH** | 300 | DL 제어 |
| **CSI-RS** | 300 | 채널 상태 참조 |
| **ULBFW** | 615 | UL 빔포밍 가중치 |
| **PUSCH1** | 2000 | UL 데이터 (첫 번째) |
| **PUSCH2** | 1855 | UL 데이터 (두 번째) |
| **PUCCH1** | 2000 | UL 제어 (첫 번째) |
| **PUCCH2** | 3000 | UL 제어 (두 번째) |
| **SRS** | 2500 | Sounding Reference Signal |
| **PRACH** | 3000 | Random Access |
| **MAC** | 500 | MAC 스케줄링 |

---

## 6. 테스트 결과 요약

### 6.1 실행 명령어

```bash
python3 measure.py \
    --cuphy ../build \
    --vectors ../../testVectors \
    --config testcases_avg_F14.json \
    --uc uc_avg_F14_TDD.json \
    --delay 0 \
    --gpu 0 \
    --freq 1500 \
    --start 1 \
    --cap 6 \
    --iterations 50 \
    --slots 45 \
    --target 60 60 \
    --tdd_pattern dddsuudddd_mMIMO \
    --save_buffers \
    --graph
```

### 6.2 결과

```
Cell capacity is 05+00 based on 45 slots run (100% on time for all channels)
```

| 테스트 셀 | 결과 |
|-----------|------|
| 1 셀 | ✅ Pass |
| 2 셀 | ✅ Pass |
| 3 셀 | ✅ Pass |
| 4 셀 | ✅ Pass |
| **5 셀** | ✅ **Pass (최대)** |
| 6 셀 | ❌ Fail (일부 채널 레이턴시 초과) |

---

## 7. 처리량 추정

### 7.1 단일 셀 Peak Throughput

```
DL Throughput (per cell) = 
    PRB × Subcarriers × Symbols × Modulation × Coding × MIMO_layers
    = 273 × 12 × 11 × 8 (256-QAM) × 0.9 × 16 layers
    ≈ 4.1 Gbps (이론적 최대)
```

### 7.2 5셀 총 처리량 추정

| 구분 | 단일 셀 | 5셀 합계 |
|------|---------|----------|
| **DL Peak** | ~4.1 Gbps | **~20.5 Gbps** |
| **UL Peak** | ~2.0 Gbps | **~10.0 Gbps** |

---

## 8. 결론

### 8.1 핵심 결과

1. **H100 NVL 1개 GPU**에서 **F14 Massive MIMO 5셀** 동시 처리 가능
2. **SM 60+60 할당**으로 안정적인 실시간 성능 달성
3. **모든 채널에서 100% on-time** 달성 (레이턴시 요구사항 충족)

### 8.2 시스템 용량

```
┌─────────────────────────────────────────────────────────────┐
│                 F14 Massive MIMO 시스템 용량                 │
├─────────────────────────────────────────────────────────────┤
│  • 최대 셀 수: 5 cells                                       │
│  • DL 동시 UE: 최대 80 UE (16 UE × 5 cells)                  │
│  • UL 동시 UE: 최대 40 UE (8 UE × 5 cells)                   │
│  • 총 DL 처리량: ~20 Gbps                                    │
│  • GPU 활용: H100 NVL × 1 (SM 60+60)                         │
└─────────────────────────────────────────────────────────────┘
```

### 8.3 추가 테스트 권장

- 더 많은 iterations (100+)로 안정성 확인
- SM 할당 변경 테스트 (50+50, 66+66 등)
- 6셀 실패 원인 상세 분석 (어떤 채널에서 레이턴시 초과)

---

## 부록: 테스트 환경

| 항목 | 값 |
|------|-----|
| **서버** | Supermicro SYS-741GE-TNRT |
| **CPU** | Intel Xeon Gold 6530 × 2 |
| **RAM** | 503 GB DDR5 |
| **GPU** | NVIDIA H100 NVL × 2 |
| **GPU VRAM** | 95.8 GB × 2 |
| **SSD** | Samsung PM9A3 1.92TB NVMe |
| **컨테이너** | aerial_25_3_cubb |

