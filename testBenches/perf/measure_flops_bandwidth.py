# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
GPU FLOPS & Bandwidth Measurement Script
Measures GPU metrics across different cell counts and generates visualization.

Measurement Modes:
    1. nvidia-smi (default): Uses MPS, fast but estimated values
    2. NCU hybrid (--use_ncu): nvidia-smi monitoring + NCU profiling, accurate FLOPS/BW
    3. NCU only (--ncu_only): NCU profiling only, most accurate but slowest

Usage Examples:

    # Fast measurement with nvidia-smi (estimated values, uses MPS)
    python3 measure_flops_bandwidth.py \\
        --cuphy ../build \\
        --vectors /workspace/aerial-cuda-accelerated-ran/testVectors \\
        --config testcases_avg_F08.json \\
        --uc uc_avg_F08_TDD.json \\
        --gpu 0 --freq 1500 --tdd_pattern dddsuudddd \\
        --start 1 --cap 16 --step 1 \\
        --target 40 40 --delay 100000 --iterations 100 \\
        --output flops_estimated.json \\
        --show_plot

    # Accurate measurement with NCU (measured values, no MPS)
    # Hybrid mode: nvidia-smi for utilization + NCU for FLOPS/BW
    python3 measure_flops_bandwidth.py \\
        --cuphy ../build \\
        --vectors /workspace/aerial-cuda-accelerated-ran/testVectors \\
        --config testcases_avg_F08.json \\
        --uc uc_avg_F08_TDD.json \\
        --gpu 0 --freq 1500 --tdd_pattern dddsuudddd \\
        --start 1 --cap 4 --step 1 \\
        --target 40 40 --delay 100000 \\
        --use_ncu \\
        --ncu_iterations 3 --ncu_timeout 600 \\
        --output flops_measured.json

    # NCU-only mode (most accurate, slowest)
    # Skip nvidia-smi monitoring, only use NCU for all metrics
    python3 measure_flops_bandwidth.py \\
        --cuphy ../build \\
        --vectors /workspace/aerial-cuda-accelerated-ran/testVectors \\
        --config testcases_avg_F08.json \\
        --uc uc_avg_F08_TDD.json \\
        --gpu 0 --freq 1500 --tdd_pattern dddsuudddd \\
        --start 1 --cap 4 --step 1 \\
        --target 40 40 --delay 100000 \\
        --ncu_only \\
        --ncu_iterations 5 --ncu_timeout 900 \\
        --output flops_ncu_only.json

Notes:
    - NCU requires MPS to be disabled. The script will automatically stop MPS.
    - NCU profiling is slow: expect 2-5 minutes per cell per iteration.
    - For quick capacity testing, use nvidia-smi mode first, then verify with NCU.
    - --ncu_iterations controls profiling depth (more = more accurate, slower)
"""

import os
import sys
import json
import argparse
import subprocess
import re
import time
import threading
from datetime import datetime

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib/numpy not available, plotting disabled")


# GPU Specifications (H100 NVL)
GPU_SPECS = {
    "H100_NVL": {
        "sm_count": 132,
        "fp32_cores_per_sm": 128,
        "fp16_cores_per_sm": 256,
        "max_clock_mhz": 1980,
        "memory_bandwidth_gbps": 3350,  # HBM3
        "tdp_w": 400,
    },
    "H100_PCIe": {
        "sm_count": 114,
        "fp32_cores_per_sm": 128,
        "fp16_cores_per_sm": 256,
        "max_clock_mhz": 1620,
        "memory_bandwidth_gbps": 2000,  # HBM2e
        "tdp_w": 350,
    },
    "A100": {
        "sm_count": 108,
        "fp32_cores_per_sm": 64,
        "fp16_cores_per_sm": 128,
        "max_clock_mhz": 1410,
        "memory_bandwidth_gbps": 2039,
        "tdp_w": 400,
    }
}


def parse_args():
    parser = argparse.ArgumentParser(description="Measure GPU FLOPS and Bandwidth")
    parser.add_argument("--cuphy", type=str, required=True, help="Path to cuPHY build directory")
    parser.add_argument("--vectors", type=str, required=True, help="Path to test vectors")
    parser.add_argument("--config", type=str, required=True, help="Test config JSON file")
    parser.add_argument("--uc", type=str, required=True, help="Use case JSON file")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument("--start", type=int, default=1, help="Starting cell count")
    parser.add_argument("--cap", type=int, default=16, help="Maximum cell count")
    parser.add_argument("--step", type=int, default=1, help="Cell count step size")
    parser.add_argument("--target", type=int, nargs=2, default=[60, 60], help="SM allocation (DL UL)")
    parser.add_argument("--freq", type=int, default=None, help="GPU clock frequency (MHz) - same as measure.py")
    parser.add_argument("--tdd_pattern", type=str, choices=["dddsu", "dddsuudddd", "dddsuudddd_mMIMO"],
                       default=None, help="TDD pattern - same as measure.py")
    parser.add_argument("--delay", type=int, default=100000, help="Delay between iterations (us)")
    parser.add_argument("--iterations", type=int, default=500, help="Number of test iterations for measurement")
    parser.add_argument("--output", type=str, default="flops_bandwidth_results.json", help="Output JSON file")
    parser.add_argument("--use_ncu", action="store_true", help="Use Nsight Compute for accurate FLOPS measurement (disables MPS)")
    parser.add_argument("--ncu_only", action="store_true", help="NCU-only mode: Skip nvidia-smi monitoring, use only NCU for measurement")
    parser.add_argument("--ncu_iterations", type=int, default=3, help="Number of iterations for NCU profiling (default: 3)")
    parser.add_argument("--ncu_timeout", type=int, default=600, help="NCU timeout in seconds per cell (default: 600)")
    parser.add_argument("--no_mps", action="store_true", help="Disable MPS (required for NCU profiling)")
    parser.add_argument("--gpu_type", type=str, default="H100_NVL", 
                       choices=list(GPU_SPECS.keys()), help="GPU type for theoretical calculations")
    parser.add_argument("--show_plot", action="store_true", help="Display plot immediately after generation")
    return parser.parse_args()


def get_gpu_info(gpu_id):
    """Get GPU name and memory info"""
    try:
        result = subprocess.run(
            f"nvidia-smi -i {gpu_id} --query-gpu=name,memory.total --format=csv,noheader",
            shell=True, capture_output=True, text=True
        )
        parts = result.stdout.strip().split(',')
        return {
            "name": parts[0].strip(),
            "memory_total_mb": int(parts[1].strip().replace("MiB", "").strip())
        }
    except:
        return {"name": "Unknown", "memory_total_mb": 0}


def parse_test_vector_info(filename):
    """
    테스트 벡터 파일명에서 설정 정보를 추출합니다.
    예: TV_cuphy_F08-DS-01_slot0_MIMO4x4_PRB273_DataSyms11_qam256.h5
    """
    info = {
        "frequency_range": None,
        "direction": None,
        "mimo": None,
        "prb": None,
        "data_symbols": None,
        "modulation": None,
        "snr_db": None
    }
    
    basename = os.path.basename(filename)
    
    # Frequency Range (F08 = FR1 3.5GHz, etc.)
    freq_match = re.search(r'F(\d+)', basename)
    if freq_match:
        freq_code = freq_match.group(1)
        freq_map = {
            "08": "FR1 100MHz (3.5GHz TDD)",
            "07": "FR1 40MHz (2.1GHz FDD)",
            "06": "FR1 20MHz (1.8GHz FDD)",
            "09": "FR2 100MHz (28GHz mmWave)"
        }
        info["frequency_range"] = freq_map.get(freq_code, f"F{freq_code}")
    
    # Direction (DS=Downlink/PDSCH, US=Uplink/PUSCH)
    if '-DS-' in basename:
        info["direction"] = "DL (PDSCH)"
    elif '-US-' in basename:
        info["direction"] = "UL (PUSCH)"
    elif '-DC-' in basename:
        info["direction"] = "DL (PDCCH)"
    elif '-UC-' in basename:
        info["direction"] = "UL (PUCCH)"
    elif '-RA-' in basename:
        info["direction"] = "UL (PRACH)"
    elif '-SS-' in basename:
        info["direction"] = "DL (SSB)"
    elif '-CR-' in basename:
        info["direction"] = "DL (CSI-RS)"
    
    # MIMO configuration
    mimo_match = re.search(r'MIMO(\d+)x(\d+)', basename)
    if mimo_match:
        info["mimo"] = f"{mimo_match.group(1)}x{mimo_match.group(2)}"
    
    # PRB (Physical Resource Block) count
    prb_match = re.search(r'PRB(\d+)', basename)
    if prb_match:
        info["prb"] = int(prb_match.group(1))
    
    # Data symbols
    sym_match = re.search(r'DataSyms(\d+)', basename)
    if sym_match:
        info["data_symbols"] = int(sym_match.group(1))
    
    # Modulation (QAM)
    qam_match = re.search(r'qam(\d+)', basename, re.IGNORECASE)
    if qam_match:
        info["modulation"] = f"QAM{qam_match.group(1)}"
    
    # SNR (for UL)
    snr_match = re.search(r'snrdb([\d.]+)', basename)
    if snr_match:
        info["snr_db"] = float(snr_match.group(1))
    
    return info


def parse_vectors_yaml(yaml_path):
    """
    vectors YAML 파일을 파싱하여 테스트 설정 정보를 추출합니다.
    """
    test_config = {
        "cell_count": 0,
        "total_slots": 0,
        "pdsch_slots": 0,
        "pusch_slots": 0,
        "pdsch_config": None,
        "pusch_config": None,
        "tdd_pattern": None
    }
    
    if not HAS_YAML:
        # YAML 파서가 없으면 간단한 텍스트 파싱
        try:
            with open(yaml_path, 'r') as f:
                content = f.read()
                
            # Cell count
            cell_match = re.search(r'cells:\s*(\d+)', content)
            if cell_match:
                test_config["cell_count"] = int(cell_match.group(1))
            
            # PDSCH/PUSCH 테스트 벡터 경로 찾기
            pdsch_files = re.findall(r'PDSCH:.*?-\s*(.*?\.h5)', content, re.DOTALL)
            pusch_files = re.findall(r'PUSCH:.*?-\s*(.*?\.h5)', content, re.DOTALL)
            
            # 슬롯 수 계산 (slots: 섹션의 - 항목 개수)
            slots_section = re.search(r'slots:\s*\n((?:- .*\n?|\s+.*\n?)+)', content)
            if slots_section:
                slot_entries = re.findall(r'^- ', slots_section.group(1), re.MULTILINE)
                test_config["total_slots"] = len(slot_entries)
            
            # 첫 번째 PDSCH/PUSCH 파일에서 설정 정보 추출
            if pdsch_files:
                test_config["pdsch_config"] = parse_test_vector_info(pdsch_files[0])
                test_config["pdsch_slots"] = len([f for f in content.split('\n') if 'PDSCH:' in f])
            
            if pusch_files:
                test_config["pusch_config"] = parse_test_vector_info(pusch_files[0])
                test_config["pusch_slots"] = len([f for f in content.split('\n') if 'PUSCH:' in f])
                
        except Exception as e:
            print(f"  Warning: Failed to parse YAML: {e}")
            return test_config
    else:
        try:
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
            
            test_config["cell_count"] = data.get("cells", 0)
            
            slots = data.get("slots", [])
            test_config["total_slots"] = len(slots)
            
            for slot in slots:
                if "PDSCH" in slot:
                    test_config["pdsch_slots"] += 1
                    if test_config["pdsch_config"] is None and slot["PDSCH"]:
                        test_config["pdsch_config"] = parse_test_vector_info(slot["PDSCH"][0])
                if "PUSCH" in slot:
                    test_config["pusch_slots"] += 1
                    if test_config["pusch_config"] is None and slot["PUSCH"]:
                        test_config["pusch_config"] = parse_test_vector_info(slot["PUSCH"][0])
                        
        except Exception as e:
            print(f"  Warning: Failed to parse YAML: {e}")
            return test_config
    
    # TDD 패턴 추정 (PDSCH only, PUSCH only, or TDD)
    if test_config["pdsch_slots"] > 0 and test_config["pusch_slots"] > 0:
        dl_ratio = test_config["pdsch_slots"] / test_config["total_slots"] * 100
        ul_ratio = test_config["pusch_slots"] / test_config["total_slots"] * 100
        test_config["tdd_pattern"] = f"TDD (DL:{dl_ratio:.0f}%, UL:{ul_ratio:.0f}%)"
    elif test_config["pdsch_slots"] > 0:
        test_config["tdd_pattern"] = "DL Only (FDD-like)"
    elif test_config["pusch_slots"] > 0:
        test_config["tdd_pattern"] = "UL Only (FDD-like)"
    
    return test_config


def print_test_configuration(test_config, target, gpu_freq=None, tdd_pattern=None):
    """테스트 설정 정보를 출력합니다."""
    print("\n" + "-" * 60)
    print("Test Configuration Details")
    print("-" * 60)
    
    # GPU Clock Frequency (from command line)
    if gpu_freq:
        print(f"  GPU Clock Frequency: {gpu_freq} MHz")
    
    # TDD Pattern (from command line)
    if tdd_pattern:
        pattern_desc = {
            "dddsu": "DDDSU (4:1 DL:UL)",
            "dddsuudddd": "DDDSUUDDDD (8:2 DL:UL)",
            "dddsuudddd_mMIMO": "DDDSUUDDDD mMIMO (8:2 DL:UL)"
        }
        print(f"  TDD Pattern: {pattern_desc.get(tdd_pattern, tdd_pattern)}")
    
    # Cell count
    print(f"  Cells: {test_config.get('cell_count', 'N/A')}")
    
    # Duplex Mode (estimated from YAML)
    if test_config.get("tdd_pattern") and not tdd_pattern:
        print(f"  Duplex Mode (from YAML): {test_config['tdd_pattern']}")
    
    # Total slots
    print(f"  Total Slots/Frame: {test_config.get('total_slots', 'N/A')}")
    print(f"    - PDSCH Slots: {test_config.get('pdsch_slots', 0)}")
    print(f"    - PUSCH Slots: {test_config.get('pusch_slots', 0)}")
    
    # PDSCH Configuration
    pdsch = test_config.get("pdsch_config")
    if pdsch:
        print(f"\n  [PDSCH Configuration]")
        if pdsch.get("frequency_range"):
            print(f"    Frequency: {pdsch['frequency_range']}")
        if pdsch.get("mimo"):
            print(f"    MIMO: {pdsch['mimo']} (TxAnt x RxAnt)")
        if pdsch.get("prb"):
            bw_mhz = pdsch["prb"] * 12 * 30 / 1000  # assuming 30kHz SCS
            print(f"    PRB: {pdsch['prb']} (~{bw_mhz:.0f}MHz @ 30kHz SCS)")
        if pdsch.get("data_symbols"):
            print(f"    Data Symbols: {pdsch['data_symbols']}")
        if pdsch.get("modulation"):
            print(f"    Modulation: {pdsch['modulation']}")
    
    # PUSCH Configuration
    pusch = test_config.get("pusch_config")
    if pusch:
        print(f"\n  [PUSCH Configuration]")
        if pusch.get("frequency_range"):
            print(f"    Frequency: {pusch['frequency_range']}")
        if pusch.get("mimo"):
            print(f"    MIMO: {pusch['mimo']} (TxAnt x RxAnt)")
        if pusch.get("prb"):
            bw_mhz = pusch["prb"] * 12 * 30 / 1000
            print(f"    PRB: {pusch['prb']} (~{bw_mhz:.0f}MHz @ 30kHz SCS)")
        if pusch.get("data_symbols"):
            print(f"    Data Symbols: {pusch['data_symbols']}")
        if pusch.get("modulation"):
            print(f"    Modulation: {pusch['modulation']}")
        if pusch.get("snr_db"):
            print(f"    SNR: {pusch['snr_db']} dB")
    
    # SM Allocation
    print(f"\n  [GPU SM Allocation]")
    print(f"    DL SMs: {target[0]}")
    print(f"    UL SMs: {target[1]}")
    print("-" * 60)


def calculate_theoretical_peak(gpu_type):
    """Calculate theoretical peak FLOPS and bandwidth"""
    spec = GPU_SPECS.get(gpu_type, GPU_SPECS["H100_NVL"])
    
    # Peak FP32 TFLOPS = SM count × FP32 cores × 2 (ops per cycle for FMA) × clock (GHz)
    peak_fp32_tflops = (spec["sm_count"] * spec["fp32_cores_per_sm"] * 2 * 
                        spec["max_clock_mhz"] / 1000) / 1000
    
    return {
        "peak_fp32_tflops": peak_fp32_tflops,
        "peak_fp32_gflops": peak_fp32_tflops * 1000,
        "peak_bandwidth_gbps": spec["memory_bandwidth_gbps"],
        "max_clock_mhz": spec["max_clock_mhz"],
        "sm_count": spec["sm_count"],
        "tdp_w": spec["tdp_w"]
    }


def check_mps_running():
    """Check if NVIDIA MPS is currently running."""
    try:
        result = subprocess.run("ps aux | grep nvidia-cuda-mps | grep -v grep", 
                               shell=True, capture_output=True, text=True)
        return len(result.stdout.strip()) > 0
    except:
        return False


def stop_mps(gpu_id):
    """Stop MPS daemon."""
    print("  Stopping MPS for NCU profiling...")
    os.system(f"echo quit | CUDA_VISIBLE_DEVICES={gpu_id} CUDA_MPS_PIPE_DIRECTORY=. CUDA_LOG_DIRECTORY=. nvidia-cuda-mps-control 2>/dev/null")
    os.system("pkill -9 nvidia-cuda-mps 2>/dev/null")
    time.sleep(2)  # Wait for MPS to fully stop


def find_ncu_path():
    """Find ncu executable path."""
    # Common NCU paths
    ncu_paths = [
        "ncu",  # In PATH
        "/usr/local/cuda/bin/ncu",
        "/opt/nvidia/nsight-compute/ncu",
        "/usr/local/NVIDIA-Nsight-Compute/ncu",
        "/opt/nvidia/nsight-compute-*/ncu",
    ]
    
    # Try to find ncu
    for path in ncu_paths:
        if '*' in path:
            # Handle glob pattern
            import glob
            matches = glob.glob(path)
            if matches:
                return matches[0]
        else:
            try:
                result = subprocess.run(f"which {path}", shell=True, capture_output=True, text=True)
                if result.returncode == 0 and result.stdout.strip():
                    return result.stdout.strip()
                # Check if path exists directly
                if os.path.exists(path):
                    return path
            except:
                continue
    
    return None


def check_ncu_available():
    """Check if NCU is available and return version info."""
    ncu_path = find_ncu_path()
    if not ncu_path:
        return None, "NCU not found in PATH or common locations"
    
    try:
        result = subprocess.run(f"{ncu_path} --version", shell=True, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return ncu_path, result.stdout.strip()
        else:
            return ncu_path, f"NCU found but error: {result.stderr}"
    except Exception as e:
        return ncu_path, f"NCU check failed: {e}"


def run_ncu_measurement(cuphy_path, vectors_yaml, gpu_id, target, delay, iterations=3, timeout_sec=600):
    """
    Run Nsight Compute to collect accurate FLOPS and bandwidth metrics.
    
    NCU requires MPS to be disabled. This function will automatically stop MPS if running.
    """
    print("  Running Nsight Compute profiling (this may take several minutes)...")
    
    # Check if NCU is available
    ncu_path, ncu_info = check_ncu_available()
    if not ncu_path:
        print(f"  ❌ NCU Error: {ncu_info}")
        print(f"     NCU (Nsight Compute) is required for accurate FLOPS measurement.")
        print(f"     Please install NVIDIA Nsight Compute or use nvidia-smi mode instead.")
        return None
    
    print(f"    NCU path: {ncu_path}")
    
    # Check and stop MPS if running
    if check_mps_running():
        stop_mps(gpu_id)
    
    # Key metrics for FLOPS and Bandwidth
    metrics = [
        # FP32 operations
        "sm__sass_thread_inst_executed_op_fadd_pred_on.sum",
        "sm__sass_thread_inst_executed_op_fmul_pred_on.sum", 
        "sm__sass_thread_inst_executed_op_ffma_pred_on.sum",
        # FP64 operations
        "sm__sass_thread_inst_executed_op_dadd_pred_on.sum",
        "sm__sass_thread_inst_executed_op_dmul_pred_on.sum",
        "sm__sass_thread_inst_executed_op_dfma_pred_on.sum",
        # Memory bandwidth
        "dram__bytes_read.sum",
        "dram__bytes_write.sum",
        # Duration
        "gpu__time_duration.sum",
        # Throughput percentages
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "dram__throughput.avg.pct_of_peak_sustained_elapsed"
    ]
    
    metrics_str = ",".join(metrics)
    
    log_file = f"ncu_output_{vectors_yaml.replace('.yaml', '').replace('/', '_')}.csv"
    
    # Use --page raw for easier parsing
    # Note: -u 5 is the correct uldl mode (not -u 2 which causes "Invalid uldl mode" error)
    # Note: --M (SM affinity) is NOT supported under NCU profiling (CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY)
    #       So we omit --M option when running under NCU
    command = f"""
    CUDA_VISIBLE_DEVICES={gpu_id} \
    {ncu_path} --target-processes all \
        --metrics {metrics_str} \
        --page raw \
        --csv \
        --log-file {log_file} \
        {cuphy_path}/cubb_gpu_test_bench/cubb_gpu_test_bench \
        -i {vectors_yaml} \
        -r {iterations} -w {delay} -u 5 -d 0 -m 1 \
        --U --D
    """
    
    try:
        print(f"    NCU command: {ncu_path} --metrics ... -r {iterations}")
        print(f"    Expected time: {iterations * 2}-{iterations * 5} minutes")
        
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=timeout_sec)
        
        # Always print stdout/stderr for debugging
        combined_output = result.stdout + result.stderr
        
        # Check for errors
        if result.returncode != 0:
            print(f"  ⚠️ NCU returned error code: {result.returncode}")
            
        # Always print stderr if present (important for debugging)
        if result.stderr and result.stderr.strip():
            print(f"    [NCU stderr]:")
            for line in result.stderr.strip().split('\n')[:20]:  # First 20 lines
                print(f"      {line}")
            
            # Check for common errors
            if "ERR_NVGPUCTRPERM" in result.stderr or "permission" in result.stderr.lower():
                print(f"\n  ❌ NCU Permission Error!")
                print(f"     Docker에서 NCU를 사용하려면 다음 옵션이 필요합니다:")
                print(f"     docker run --cap-add=CAP_SYS_ADMIN --cap-add=CAP_SYS_PTRACE ...")
                print(f"     또는 Docker 외부에서 sudo로 실행하세요.")
            elif "not found" in result.stderr.lower():
                print(f"  ❌ NCU or application not found")
            elif "No kernels" in result.stderr or "no kernel" in result.stderr.lower():
                print(f"  ❌ NCU가 커널을 찾지 못했습니다.")
                print(f"     cubb_gpu_test_bench가 너무 빨리 종료되었을 수 있습니다.")
        
        # Print stdout if stderr is empty but there's an error
        if result.returncode != 0 and not result.stderr and result.stdout:
            print(f"    [NCU stdout]:")
            for line in result.stdout.strip().split('\n')[:20]:
                print(f"      {line}")
        
        # Try to parse from log file first (more reliable)
        if os.path.exists(log_file):
            print(f"    NCU log file found: {log_file}")
            with open(log_file, 'r') as f:
                log_content = f.read()
            print(f"    Log file size: {len(log_content)} bytes")
            metrics_result = parse_ncu_output(log_content, is_csv_file=True)
            # Cleanup log file
            try:
                os.remove(log_file)
            except:
                pass
            return metrics_result
        else:
            print(f"    ⚠️ NCU log file not created")
        
        # Fallback to stdout parsing
        return parse_ncu_output(result.stdout + result.stderr, is_csv_file=False)
        
    except subprocess.TimeoutExpired:
        print(f"  ⚠️ Warning: NCU profiling timed out after {timeout_sec}s")
        print(f"     Try reducing iterations or using --ncu_iterations option")
        return None
    except Exception as e:
        print(f"  ⚠️ Warning: NCU profiling failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def parse_numeric_value(val_str):
    """Parse numeric value from NCU output (handles scientific notation, integers, floats, units)."""
    if not val_str:
        return None
    
    val_str = val_str.strip().replace('"', '').replace(',', '')
    
    if not val_str or val_str == 'N/A' or val_str == '' or val_str == '(null)':
        return None
    
    # Remove common units
    units = ['byte', 'bytes', 'nsecond', 'ns', 'cycle', 'inst', '%']
    for unit in units:
        val_str = val_str.replace(unit, '').strip()
    
    try:
        # Try integer first
        if val_str.isdigit():
            return int(val_str)
        # Try float (handles scientific notation like 1.23e+09)
        return float(val_str)
    except ValueError:
        return None


def parse_ncu_output(output, is_csv_file=False):
    """
    Parse Nsight Compute output to extract metrics.
    
    Supports both CSV file output and raw stdout output.
    """
    metrics = {
        "flop_count_fadd": 0,
        "flop_count_fmul": 0,
        "flop_count_ffma": 0,
        "flop_count_dadd": 0,
        "flop_count_dmul": 0,
        "flop_count_dfma": 0,
        "dram_bytes_read": 0,
        "dram_bytes_write": 0,
        "duration_ns": 0,
        "sm_throughput_pct": [],
        "dram_throughput_pct": [],
        "kernel_count": 0,
        "parse_success_count": 0,
    }
    
    if not output:
        print("    [NCU] No output received from ncu")
        return metrics
    
    lines = output.strip().split('\n')
    print(f"    [NCU] Parsing {len(lines)} lines of output...")
    
    # Detect if it's CSV format
    header_idx = -1
    for i, line in enumerate(lines):
        if '"ID"' in line or 'ID,' in line or 'Kernel Name' in line:
            header_idx = i
            break
    
    if header_idx >= 0:
        # CSV format - parse as structured data
        print(f"    [NCU] Detected CSV format, header at line {header_idx}")
        
        # Find column indices
        header = lines[header_idx].replace('"', '').split(',')
        col_indices = {}
        for i, col in enumerate(header):
            col_lower = col.lower().strip()
            if 'metric name' in col_lower or 'metric' in col_lower:
                col_indices['metric'] = i
            elif 'metric value' in col_lower or 'value' in col_lower:
                col_indices['value'] = i
        
        # Parse data rows
        for line in lines[header_idx + 1:]:
            if not line.strip() or line.startswith('#'):
                continue
            
            parts = line.replace('"', '').split(',')
            metrics["kernel_count"] += 1
            
            # Look for metric values in the line
            line_lower = line.lower()
            
            try:
                # FP32 FADD
                if 'op_fadd_pred_on' in line_lower:
                    val = parse_numeric_value(parts[-1] if parts else '')
                    if val is not None:
                    metrics["flop_count_fadd"] += int(val)
                        metrics["parse_success_count"] += 1
                
                # FP32 FMUL
                elif 'op_fmul_pred_on' in line_lower:
                    val = parse_numeric_value(parts[-1] if parts else '')
                    if val is not None:
                    metrics["flop_count_fmul"] += int(val)
                        metrics["parse_success_count"] += 1
                
                # FP32 FFMA (most important - fused multiply-add)
                elif 'op_ffma_pred_on' in line_lower:
                    val = parse_numeric_value(parts[-1] if parts else '')
                    if val is not None:
                    metrics["flop_count_ffma"] += int(val)
                        metrics["parse_success_count"] += 1
                
                # FP64 operations
                elif 'op_dadd_pred_on' in line_lower:
                    val = parse_numeric_value(parts[-1] if parts else '')
                    if val is not None:
                    metrics["flop_count_dadd"] += int(val)
                        metrics["parse_success_count"] += 1
                
                elif 'op_dmul_pred_on' in line_lower:
                    val = parse_numeric_value(parts[-1] if parts else '')
                    if val is not None:
                    metrics["flop_count_dmul"] += int(val)
                        metrics["parse_success_count"] += 1
                
                elif 'op_dfma_pred_on' in line_lower:
                    val = parse_numeric_value(parts[-1] if parts else '')
                    if val is not None:
                    metrics["flop_count_dfma"] += int(val)
                        metrics["parse_success_count"] += 1
                
                # DRAM bytes
                elif 'dram__bytes_read.sum' in line_lower:
                    val = parse_numeric_value(parts[-1] if parts else '')
                    if val is not None:
                    metrics["dram_bytes_read"] += int(val)
                        metrics["parse_success_count"] += 1
                
                elif 'dram__bytes_write.sum' in line_lower:
                    val = parse_numeric_value(parts[-1] if parts else '')
                    if val is not None:
                    metrics["dram_bytes_write"] += int(val)
                        metrics["parse_success_count"] += 1
                
                # Duration
                elif 'gpu__time_duration.sum' in line_lower:
                    val = parse_numeric_value(parts[-1] if parts else '')
                    if val is not None:
                    metrics["duration_ns"] += int(val)
                        metrics["parse_success_count"] += 1
                
                # Throughput percentages
                elif 'sm__throughput.avg' in line_lower:
                    val = parse_numeric_value(parts[-1] if parts else '')
                    if val is not None:
                        metrics["sm_throughput_pct"].append(val)
                        metrics["parse_success_count"] += 1
                
                elif 'dram__throughput.avg' in line_lower:
                    val = parse_numeric_value(parts[-1] if parts else '')
                    if val is not None:
                        metrics["dram_throughput_pct"].append(val)
                        metrics["parse_success_count"] += 1
            
            except Exception as e:
                continue
    
    else:
        # Raw text format - look for patterns in lines
        print("    [NCU] Parsing raw text format...")
        
        for line in lines:
            line_lower = line.lower()
            
            # Skip non-metric lines
            if '==' in line or not any(c.isdigit() for c in line):
                continue
            
            try:
                # Extract numeric values from line
                parts = line.split()
                
                if 'fadd' in line_lower and 'pred_on' in line_lower:
                    for part in reversed(parts):
                        val = parse_numeric_value(part)
                        if val is not None:
                            metrics["flop_count_fadd"] += int(val)
                            metrics["parse_success_count"] += 1
                            break
                
                elif 'ffma' in line_lower and 'pred_on' in line_lower:
                    for part in reversed(parts):
                        val = parse_numeric_value(part)
                        if val is not None:
                            metrics["flop_count_ffma"] += int(val)
                            metrics["parse_success_count"] += 1
                            break
                
                elif 'dram__bytes_read' in line_lower:
                    for part in reversed(parts):
                        val = parse_numeric_value(part)
                        if val is not None:
                            metrics["dram_bytes_read"] += int(val)
                            metrics["parse_success_count"] += 1
                            break
                
                elif 'dram__bytes_write' in line_lower:
                    for part in reversed(parts):
                        val = parse_numeric_value(part)
                        if val is not None:
                            metrics["dram_bytes_write"] += int(val)
                            metrics["parse_success_count"] += 1
                            break
                
                elif 'gpu__time_duration' in line_lower:
                    for part in reversed(parts):
                        val = parse_numeric_value(part)
                        if val is not None:
                            metrics["duration_ns"] += int(val)
                            metrics["parse_success_count"] += 1
                            break
            
            except Exception:
            continue
    
    # Debug summary
    print(f"    [NCU] Parsed {metrics['parse_success_count']} metric values from {metrics['kernel_count']} kernels")
    
    # Calculate totals
    total_fp32_flops = (metrics["flop_count_fadd"] + metrics["flop_count_fmul"] + 
                        2 * metrics["flop_count_ffma"])
    total_fp64_flops = (metrics["flop_count_dadd"] + metrics["flop_count_dmul"] + 
                        2 * metrics["flop_count_dfma"])
    total_bytes = metrics["dram_bytes_read"] + metrics["dram_bytes_write"]
    duration_sec = metrics["duration_ns"] / 1e9 if metrics["duration_ns"] > 0 else 0
    
    if total_fp32_flops > 0:
        gflops = total_fp32_flops / 1e9 / duration_sec if duration_sec > 0 else 0
        print(f"    [NCU] FP32 FLOPS: {total_fp32_flops:,} ({gflops:.1f} GFLOPS)")
    
    if total_fp64_flops > 0:
        gflops64 = total_fp64_flops / 1e9 / duration_sec if duration_sec > 0 else 0
        print(f"    [NCU] FP64 FLOPS: {total_fp64_flops:,} ({gflops64:.1f} GFLOPS)")
    
    if total_bytes > 0:
        bw_gbps = total_bytes / 1e9 / duration_sec if duration_sec > 0 else 0
        print(f"    [NCU] DRAM Bytes: {total_bytes:,} ({bw_gbps:.1f} GB/s)")
    
    if duration_sec > 0:
        print(f"    [NCU] Duration: {duration_sec:.3f} sec")
    
    return metrics


def monitor_gpu_continuous(gpu_id, stop_event, samples_list):
    """Continuously monitor GPU metrics in a separate thread."""
    while not stop_event.is_set():
        try:
            result = subprocess.run(
                f"nvidia-smi -i {gpu_id} --query-gpu=utilization.gpu,utilization.memory,power.draw,clocks.sm,clocks.mem,memory.used --format=csv,noheader,nounits",
                shell=True, capture_output=True, text=True, timeout=2
            )
            parts = [p.strip() for p in result.stdout.strip().split(',')]
            if len(parts) >= 6:
                sample = {
                    "gpu_util": float(parts[0]),
                    "mem_util": float(parts[1]),
                    "power_w": float(parts[2]),
                    "sm_clock_mhz": float(parts[3]),
                    "mem_clock_mhz": float(parts[4]),
                    "memory_used_mb": float(parts[5]),
                    "timestamp": time.time()
                }
                samples_list.append(sample)
        except:
            pass
        time.sleep(0.1)  # Sample every 100ms


def measure_with_continuous_monitoring(cuphy_path, vectors_yaml, gpu_id, target, iterations, delay, gpu_type):
    """
    Run test with continuous GPU monitoring for accurate utilization measurement.
    """
    samples = []
    stop_event = threading.Event()
    
    # Start monitoring thread
    monitor_thread = threading.Thread(
        target=monitor_gpu_continuous, 
        args=(gpu_id, stop_event, samples)
    )
    monitor_thread.start()
    
    # Give monitor thread time to start
    time.sleep(0.5)
    
    # Run test bench
    test_cmd = f"""
    CUDA_VISIBLE_DEVICES={gpu_id} \
    CUDA_MPS_PIPE_DIRECTORY=. \
    CUDA_LOG_DIRECTORY=. \
    {cuphy_path}/cubb_gpu_test_bench/cubb_gpu_test_bench \
    -i {vectors_yaml} \
    -r {iterations} -w {delay} -u 5 -d 0 -m 1 \
    --M {target[0]},{target[1]} \
    --U --D
    """
    
    start_time = time.time()
    result = subprocess.run(test_cmd, shell=True, capture_output=True, text=True)
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Stop monitoring
    stop_event.set()
    monitor_thread.join(timeout=2)
    
    # Filter samples to only those during test execution
    test_samples = [s for s in samples if start_time <= s["timestamp"] <= end_time]
    
    if not test_samples:
        test_samples = samples[-10:] if samples else []
    
    # Calculate metrics
    metrics = {
        "gpu_util_samples": [s["gpu_util"] for s in test_samples],
        "mem_util_samples": [s["mem_util"] for s in test_samples],
        "power_samples": [s["power_w"] for s in test_samples],
        "sm_clock_samples": [s["sm_clock_mhz"] for s in test_samples],
        "mem_clock_samples": [s["mem_clock_mhz"] for s in test_samples],
        "memory_used_samples": [s["memory_used_mb"] for s in test_samples],
        "elapsed_time_sec": elapsed_time,
        "sample_count": len(test_samples)
    }
    
    return metrics, result.stdout


def calculate_estimated_gflops(raw_metrics, gpu_type):
    """
    Calculate estimated GFLOPS based on GPU utilization and clock speed.
    
    Estimation method:
    - Achieved GFLOPS = Peak GFLOPS × (GPU Util / 100) × (Current Clock / Max Clock)
    """
    spec = GPU_SPECS.get(gpu_type, GPU_SPECS["H100_NVL"])
    
    # Get average values
    gpu_util = raw_metrics.get("gpu_util_samples", [])
    sm_clocks = raw_metrics.get("sm_clock_samples", [])
    
    if not gpu_util or not sm_clocks:
        return 0.0
    
    avg_gpu_util = sum(gpu_util) / len(gpu_util) if gpu_util else 0
    avg_sm_clock = sum(sm_clocks) / len(sm_clocks) if sm_clocks else spec["max_clock_mhz"]
    
    # Calculate peak GFLOPS at max clock
    peak_gflops = (spec["sm_count"] * spec["fp32_cores_per_sm"] * 2 * 
                   spec["max_clock_mhz"] / 1000)
    
    # Estimated achieved GFLOPS
    # Scale by utilization and clock ratio
    clock_ratio = avg_sm_clock / spec["max_clock_mhz"]
    estimated_gflops = peak_gflops * (avg_gpu_util / 100) * clock_ratio
    
    return estimated_gflops


def get_dcgm_bandwidth(gpu_id, duration_sec=5):
    """
    Use DCGM to get accurate memory bandwidth measurement.
    Returns (read_bw_gbps, write_bw_gbps) or (None, None) if DCGM not available.
    """
    try:
        # Check if dcgmi is available
        result = subprocess.run("which dcgmi", shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            return None, None
        
        # Run dcgmi dmon for bandwidth metrics
        # Field IDs: 1011=DRAM_ACTIVE (can derive bandwidth), 1005=SM_ACTIVE
        cmd = f"timeout {duration_sec} dcgmi dmon -i {gpu_id} -e 1011,1012 -d 100 2>/dev/null"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode != 0 or not result.stdout:
            return None, None
        
        # Parse DCGM output
        lines = result.stdout.strip().split('\n')
        read_bw_samples = []
        write_bw_samples = []
        
        for line in lines:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 4:
                try:
                    read_bw = float(parts[2])  # DRAM Read BW
                    write_bw = float(parts[3])  # DRAM Write BW
                    if read_bw > 0:
                        read_bw_samples.append(read_bw)
                    if write_bw > 0:
                        write_bw_samples.append(write_bw)
                except:
                    continue
        
        avg_read = sum(read_bw_samples) / len(read_bw_samples) if read_bw_samples else None
        avg_write = sum(write_bw_samples) / len(write_bw_samples) if write_bw_samples else None
        
        return avg_read, avg_write
    except Exception as e:
        return None, None


def calculate_estimated_bandwidth(raw_metrics, gpu_type):
    """
    Calculate estimated memory bandwidth based on memory utilization.
    Note: nvidia-smi memory utilization often reports 0% for bursty workloads.
    For compute-bound workloads like cuPHY, this is expected behavior.
    Use NCU (--use_ncu) for accurate bandwidth measurement, or DCGM if available.
    """
    spec = GPU_SPECS.get(gpu_type, GPU_SPECS["H100_NVL"])
    
    # Check if DCGM bandwidth was collected
    dcgm_bw = raw_metrics.get("dcgm_bandwidth_gbps")
    if dcgm_bw and dcgm_bw > 0:
        return dcgm_bw
    
    mem_util = raw_metrics.get("mem_util_samples", [])
    avg_mem_util = sum(mem_util) / len(mem_util) if mem_util else 0
    
    # If memory utilization is very low (< 1%) or no samples, estimate based on GPU utilization
    # nvidia-smi memory utilization is unreliable for bursty compute workloads
    # Typical compute-bound workloads use ~10-30% of peak bandwidth
    MIN_RELIABLE_MEM_UTIL = 1.0  # Below 1%, nvidia-smi mem_util is unreliable
    
    if avg_mem_util < MIN_RELIABLE_MEM_UTIL or not mem_util:
        gpu_util = raw_metrics.get("gpu_util_samples", [])
        avg_gpu_util = sum(gpu_util) / len(gpu_util) if gpu_util else 0
        # Rough estimate: assume 15% of peak bandwidth when GPU is active
        estimated_bandwidth = spec["memory_bandwidth_gbps"] * (avg_gpu_util / 100) * 0.15
        return estimated_bandwidth
    
    # Estimated bandwidth = Peak bandwidth × memory utilization
    estimated_bandwidth = spec["memory_bandwidth_gbps"] * (avg_mem_util / 100)
    
    return estimated_bandwidth


def calculate_all_metrics(raw_metrics, ncu_metrics, gpu_type):
    """
    Calculate all derived metrics from raw measurements.
    """
    calculated = {}
    spec = GPU_SPECS.get(gpu_type, GPU_SPECS["H100_NVL"])
    
    # From nvidia-smi samples
    gpu_util = raw_metrics.get("gpu_util_samples", [])
    mem_util = raw_metrics.get("mem_util_samples", [])
    power = raw_metrics.get("power_samples", [])
    sm_clocks = raw_metrics.get("sm_clock_samples", [])
    memory_used = raw_metrics.get("memory_used_samples", [])
    
    calculated["avg_gpu_util_pct"] = sum(gpu_util) / len(gpu_util) if gpu_util else 0
    calculated["max_gpu_util_pct"] = max(gpu_util) if gpu_util else 0
    calculated["avg_mem_util_pct"] = sum(mem_util) / len(mem_util) if mem_util else 0
    calculated["max_mem_util_pct"] = max(mem_util) if mem_util else 0
    calculated["avg_power_w"] = sum(power) / len(power) if power else 0
    calculated["max_power_w"] = max(power) if power else 0
    calculated["avg_sm_clock_mhz"] = sum(sm_clocks) / len(sm_clocks) if sm_clocks else 0
    calculated["max_sm_clock_mhz"] = max(sm_clocks) if sm_clocks else 0
    calculated["avg_memory_used_mb"] = sum(memory_used) / len(memory_used) if memory_used else 0
    calculated["sample_count"] = raw_metrics.get("sample_count", 0)
    
    # Estimated GFLOPS (based on utilization)
    calculated["estimated_gflops"] = calculate_estimated_gflops(raw_metrics, gpu_type)
    
    # Estimated Bandwidth (based on memory utilization)
    calculated["estimated_bandwidth_gbps"] = calculate_estimated_bandwidth(raw_metrics, gpu_type)
    
    # If ncu metrics available, calculate actual FLOPS
    if ncu_metrics:
        # FP32 FLOPS = FADD + FMUL + 2*FFMA
        fp32_flops = (ncu_metrics.get("flop_count_fadd", 0) + 
                      ncu_metrics.get("flop_count_fmul", 0) + 
                      2 * ncu_metrics.get("flop_count_ffma", 0))
        
        # FP64 FLOPS = DADD + DMUL + 2*DFMA
        fp64_flops = (ncu_metrics.get("flop_count_dadd", 0) + 
                      ncu_metrics.get("flop_count_dmul", 0) + 
                      2 * ncu_metrics.get("flop_count_dfma", 0))
        
        duration_sec = ncu_metrics.get("duration_ns", 1) / 1e9
        
        if duration_sec > 0:
            calculated["measured_fp32_gflops"] = fp32_flops / 1e9 / duration_sec
            calculated["measured_fp64_gflops"] = fp64_flops / 1e9 / duration_sec
            calculated["measured_total_gflops"] = (fp32_flops + fp64_flops) / 1e9 / duration_sec
        else:
            calculated["measured_fp32_gflops"] = 0
            calculated["measured_fp64_gflops"] = 0
            calculated["measured_total_gflops"] = 0
        
        # Bandwidth from ncu
        total_bytes = ncu_metrics.get("dram_bytes_read", 0) + ncu_metrics.get("dram_bytes_write", 0)
        calculated["measured_bandwidth_gbps"] = total_bytes / 1e9 / duration_sec if duration_sec > 0 else 0
        
        # SM and DRAM throughput from ncu
        sm_util = ncu_metrics.get("sm_throughput_pct", [])
        dram_util = ncu_metrics.get("dram_throughput_pct", [])
        calculated["ncu_avg_sm_throughput_pct"] = sum(sm_util) / len(sm_util) if sm_util else 0
        calculated["ncu_avg_dram_throughput_pct"] = sum(dram_util) / len(dram_util) if dram_util else 0
    
    # Theoretical peaks for reference
    peak = calculate_theoretical_peak(gpu_type)
    calculated["theoretical_peak_gflops"] = peak["peak_fp32_gflops"]
    calculated["theoretical_peak_bandwidth_gbps"] = peak["peak_bandwidth_gbps"]
    
    # Efficiency metrics
    if "measured_total_gflops" in calculated:
        calculated["compute_efficiency_pct"] = (calculated["measured_total_gflops"] / 
                                                 peak["peak_fp32_gflops"]) * 100
    else:
        calculated["compute_efficiency_pct"] = (calculated["estimated_gflops"] / 
                                                 peak["peak_fp32_gflops"]) * 100
    
    return calculated


def plot_results(results, output_prefix, gpu_type, measurement_mode="nvidia-smi"):
    """Generate visualization plots."""
    if not HAS_MATPLOTLIB:
        print("Plotting skipped (matplotlib not available)")
        return None
    
    cell_counts = sorted([int(c) for c in results.keys()])
    cell_counts_str = [str(c) for c in cell_counts]
    
    # Extract metrics
    gpu_util = [results[str(c)].get("avg_gpu_util_pct", 0) for c in cell_counts]
    max_gpu_util = [results[str(c)].get("max_gpu_util_pct", 0) for c in cell_counts]
    mem_util = [results[str(c)].get("avg_mem_util_pct", 0) for c in cell_counts]
    power = [results[str(c)].get("avg_power_w", 0) for c in cell_counts]
    sm_clock = [results[str(c)].get("avg_sm_clock_mhz", 0) for c in cell_counts]
    memory_used = [results[str(c)].get("avg_memory_used_mb", 0) for c in cell_counts]
    
    # GFLOPS - prefer measured, fallback to estimated
    gflops = []
    gflops_type = "estimated"
    for c in cell_counts:
        r = results[str(c)]
        measured = r.get("measured_total_gflops", 0)
        estimated = r.get("estimated_gflops", 0)
        if measured > 0:
            gflops.append(measured)
            gflops_type = "measured"
        else:
            gflops.append(estimated)
    gflops_label = f"{'Measured' if gflops_type == 'measured' else 'Estimated'} GFLOPS"
    
    # Bandwidth
    bandwidth = []
    bw_type = "estimated"
    for c in cell_counts:
        r = results[str(c)]
        measured = r.get("measured_bandwidth_gbps", 0)
        estimated = r.get("estimated_bandwidth_gbps", 0)
        if measured > 0:
            bandwidth.append(measured)
            bw_type = "measured"
        else:
            bandwidth.append(estimated)
    bw_label = f"{'Measured' if bw_type == 'measured' else 'Estimated'} Bandwidth"
    
    # Create figure with better styling
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'GPU Resource Usage vs Cell Count ({gpu_type})\nMeasurement: {measurement_mode}', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: GPU Utilization
    ax1 = axes[0, 0]
    ax1.fill_between(cell_counts, gpu_util, alpha=0.3, color='blue')
    ax1.plot(cell_counts, gpu_util, 'b-o', label='Avg GPU Util', linewidth=2, markersize=6)
    ax1.plot(cell_counts, max_gpu_util, 'b--^', label='Max GPU Util', linewidth=1, markersize=4, alpha=0.7)
    ax1.set_xlabel('Cell Count')
    ax1.set_ylabel('GPU Utilization (%)')
    ax1.set_title('GPU Utilization')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(cell_counts)
    ax1.set_ylim(0, 105)
    
    # Plot 2: GFLOPS
    ax2 = axes[0, 1]
    colors = ['green' if g > 0 else 'gray' for g in gflops]
    bars = ax2.bar(cell_counts, gflops, color=colors, alpha=0.7, edgecolor='darkgreen')
    ax2.set_xlabel('Cell Count')
    ax2.set_ylabel('GFLOPS')
    ax2.set_title(f'{gflops_label}')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xticks(cell_counts)
    
    # Add value labels on bars
    for bar, val in zip(bars, gflops):
        if val > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
                    f'{val:.0f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 3: Memory Bandwidth
    ax3 = axes[0, 2]
    ax3.bar(cell_counts, bandwidth, color='purple', alpha=0.7, edgecolor='darkviolet')
    ax3.set_xlabel('Cell Count')
    ax3.set_ylabel('Bandwidth (GB/s)')
    ax3.set_title(f'{bw_label}')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_xticks(cell_counts)
    
    # Plot 4: Power Consumption
    ax4 = axes[1, 0]
    ax4.fill_between(cell_counts, power, alpha=0.3, color='red')
    ax4.plot(cell_counts, power, 'r-s', linewidth=2, markersize=6)
    ax4.set_xlabel('Cell Count')
    ax4.set_ylabel('Power (W)')
    ax4.set_title('Power Consumption')
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(cell_counts)
    
    # Add TDP line
    spec = GPU_SPECS.get(gpu_type, GPU_SPECS["H100_NVL"])
    ax4.axhline(y=spec["tdp_w"], color='darkred', linestyle='--', label=f'TDP ({spec["tdp_w"]}W)')
    ax4.legend()
    
    # Plot 5: SM Clock
    ax5 = axes[1, 1]
    ax5.plot(cell_counts, sm_clock, 'orange', marker='D', linewidth=2, markersize=6)
    ax5.set_xlabel('Cell Count')
    ax5.set_ylabel('SM Clock (MHz)')
    ax5.set_title('SM Clock Frequency')
    ax5.grid(True, alpha=0.3)
    ax5.set_xticks(cell_counts)
    ax5.axhline(y=spec["max_clock_mhz"], color='darkorange', linestyle='--', 
                label=f'Max Boost ({spec["max_clock_mhz"]} MHz)')
    ax5.legend()
    
    # Plot 6: Memory Usage
    ax6 = axes[1, 2]
    ax6.bar(cell_counts, [m/1024 for m in memory_used], color='teal', alpha=0.7, edgecolor='darkcyan')
    ax6.set_xlabel('Cell Count')
    ax6.set_ylabel('Memory Used (GB)')
    ax6.set_title('GPU Memory Usage')
    ax6.grid(True, alpha=0.3, axis='y')
    ax6.set_xticks(cell_counts)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = f"{output_prefix}_plot.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {plot_file}")
    
    return plot_file


def main():
    args = parse_args()
    
    print("=" * 70)
    print("GPU FLOPS & Bandwidth Measurement")
    print("=" * 70)
    
    # Get GPU info
    gpu_info = get_gpu_info(args.gpu)
    print(f"GPU {args.gpu}: {gpu_info['name']}")
    print(f"GPU Type (for calculations): {args.gpu_type}")
    if args.freq:
        print(f"GPU Clock Frequency: {args.freq} MHz")
    if args.tdd_pattern:
        print(f"TDD Pattern: {args.tdd_pattern}")
    print(f"Cell range: {args.start} to {args.cap} (step {args.step})")
    print(f"SM allocation: DL={args.target[0]}, UL={args.target[1]}")
    print(f"Iterations: {args.iterations}, Delay: {args.delay} us")
    
    # Determine measurement mode
    use_ncu = args.use_ncu or args.ncu_only
    ncu_only = args.ncu_only
    use_mps = not (use_ncu or args.no_mps)
    
    if ncu_only:
        measurement_mode = "NCU-only (most accurate)"
    elif use_ncu:
        measurement_mode = "NCU hybrid (nvidia-smi + NCU)"
    else:
        measurement_mode = "nvidia-smi (estimated)"
    
    print(f"Measurement Mode: {measurement_mode}")
    print(f"MPS Enabled: {use_mps}")
    if use_ncu:
        print(f"NCU Iterations: {args.ncu_iterations}")
        print(f"NCU Timeout: {args.ncu_timeout}s per cell")
        estimated_time = (args.cap - args.start + 1) * args.ncu_iterations * 2  # ~2 min per iteration
        print(f"Estimated Total NCU Time: {estimated_time//60}h {estimated_time%60}min (worst case)")
    
    # Show theoretical peaks
    peak = calculate_theoretical_peak(args.gpu_type)
    print(f"\nTheoretical Peak ({args.gpu_type}):")
    print(f"  FP32: {peak['peak_fp32_tflops']:.1f} TFLOPS")
    print(f"  Memory BW: {peak['peak_bandwidth_gbps']} GB/s")
    print("=" * 70)
    
    # Parse and display test configuration from first available vectors file
    first_vectors_yaml = f"vectors-{str(args.start).zfill(2)}.yaml"
    if os.path.exists(first_vectors_yaml):
        test_config = parse_vectors_yaml(first_vectors_yaml)
        print_test_configuration(test_config, args.target, args.freq, args.tdd_pattern)
    else:
        print(f"\nWarning: Could not find {first_vectors_yaml} to display test configuration")
    
    results = {}
    
    # Save and set GPU clock frequency if specified
    gpu_clock_saved = None
    if args.freq:
        try:
            result = subprocess.run(
                f"nvidia-smi -i {args.gpu} --query-gpu=clocks.current.graphics --format=csv,noheader,nounits",
                shell=True, capture_output=True, text=True
            )
            gpu_clock_saved = int(result.stdout.strip())
            if gpu_clock_saved != args.freq:
                print(f"Setting GPU clock to {args.freq} MHz (was {gpu_clock_saved} MHz)...")
                os.system(f"nvidia-smi -i {args.gpu} -lgc {args.freq} 2>/dev/null")
        except:
            pass
    
    # Setup MPS only if not using NCU
    if use_mps:
        print("\nStarting MPS...")
    os.system(f"CUDA_VISIBLE_DEVICES={args.gpu} CUDA_MPS_PIPE_DIRECTORY=. CUDA_LOG_DIRECTORY=. nvidia-cuda-mps-control -d 2>/dev/null")
    else:
        print("\nMPS disabled for NCU profiling or by user request")
        # Ensure MPS is stopped if running
        stop_mps(args.gpu)
    
    try:
        total_cells = len(range(args.start, args.cap + 1, args.step))
        for idx, cell_count in enumerate(range(args.start, args.cap + 1, args.step)):
            print(f"\n[{cell_count:2d} cells] ({idx+1}/{total_cells}) Measuring...")
            
            # Check for vectors YAML
            vectors_yaml = f"vectors-{str(cell_count).zfill(2)}.yaml"
            
            if not os.path.exists(vectors_yaml):
                print(f"  ⚠️ Warning: {vectors_yaml} not found, skipping...")
                continue
            
            raw_metrics = None
            ncu_metrics = None
            
            if ncu_only:
                # NCU-only mode: skip nvidia-smi monitoring
                print(f"  [NCU-only mode] Running Nsight Compute profiling...")
                ncu_metrics = run_ncu_measurement(
                    args.cuphy, vectors_yaml, args.gpu, args.target, args.delay,
                    iterations=args.ncu_iterations, timeout_sec=args.ncu_timeout
                )
                
                # Create minimal raw_metrics structure for NCU-only mode
                raw_metrics = {
                    "gpu_util_samples": [],
                    "mem_util_samples": [],
                    "power_samples": [],
                    "sm_clock_samples": [],
                    "mem_clock_samples": [],
                    "memory_used_samples": [],
                    "elapsed_time_sec": 0,
                    "sample_count": 0
                }
                
                # Get a quick snapshot of GPU stats via nvidia-smi (for reference only)
                try:
                    result = subprocess.run(
                        f"nvidia-smi -i {args.gpu} --query-gpu=power.draw,clocks.sm,memory.used --format=csv,noheader,nounits",
                        shell=True, capture_output=True, text=True, timeout=5
                    )
                    parts = [p.strip() for p in result.stdout.strip().split(',')]
                    if len(parts) >= 3:
                        raw_metrics["power_samples"] = [float(parts[0])]
                        raw_metrics["sm_clock_samples"] = [float(parts[1])]
                        raw_metrics["memory_used_samples"] = [float(parts[2])]
                except:
                    pass
            else:
                # Standard mode: measure with continuous monitoring
            raw_metrics, test_output = measure_with_continuous_monitoring(
                args.cuphy, vectors_yaml, args.gpu, args.target, 
                args.iterations, args.delay, args.gpu_type
            )
            
                # Run ncu if requested (hybrid mode)
                if use_ncu:
                    print(f"  [Hybrid mode] Running Nsight Compute profiling...")
                ncu_metrics = run_ncu_measurement(
                        args.cuphy, vectors_yaml, args.gpu, args.target, args.delay,
                        iterations=args.ncu_iterations, timeout_sec=args.ncu_timeout
                )
            
            # Calculate all metrics
            calculated = calculate_all_metrics(raw_metrics, ncu_metrics, args.gpu_type)
            
            # Store results
            results[str(cell_count)] = calculated
            
            # Print summary
            print(f"\n  📊 Results Summary:")
            if not ncu_only:
            print(f"  Samples collected: {calculated['sample_count']}")
            print(f"  GPU Util: avg={calculated['avg_gpu_util_pct']:.1f}%, max={calculated['max_gpu_util_pct']:.1f}%")
            print(f"  SM Clock: avg={calculated['avg_sm_clock_mhz']:.0f} MHz")
            print(f"  Power: {calculated['avg_power_w']:.1f}W")
            print(f"  Memory Used: {calculated['avg_memory_used_mb']:.0f} MB")
            
            if "measured_total_gflops" in calculated and calculated["measured_total_gflops"] > 0:
                print(f"  ✅ GFLOPS (NCU measured): {calculated['measured_total_gflops']:.1f}")
                print(f"  ✅ Bandwidth (NCU measured): {calculated['measured_bandwidth_gbps']:.1f} GB/s")
                if "ncu_avg_sm_throughput_pct" in calculated:
                    print(f"     SM Throughput: {calculated['ncu_avg_sm_throughput_pct']:.1f}%")
                if "ncu_avg_dram_throughput_pct" in calculated:
                    print(f"     DRAM Throughput: {calculated['ncu_avg_dram_throughput_pct']:.1f}%")
            else:
                print(f"  ⚠️ GFLOPS (estimated): {calculated['estimated_gflops']:.1f}")
                # Check if bandwidth is rough estimate due to low/0% mem util
                mem_util = calculated.get("avg_mem_util_pct", 0)
                # nvidia-smi mem_util < 1% is unreliable, so we use GPU-based estimate
                if mem_util < 1.0 and calculated['estimated_bandwidth_gbps'] > 0:
                    print(f"  ⚠️ Bandwidth (rough est.): {calculated['estimated_bandwidth_gbps']:.1f} GB/s")
                    print(f"     (nvidia-smi mem_util={mem_util:.1f}%, using GPU-based estimate)")
                else:
                    print(f"  ⚠️ Bandwidth (estimated): {calculated['estimated_bandwidth_gbps']:.1f} GB/s")
            
    finally:
        # Cleanup MPS if it was started
        if use_mps:
            print("\nStopping MPS...")
        os.system(f"echo quit | CUDA_VISIBLE_DEVICES={args.gpu} CUDA_MPS_PIPE_DIRECTORY=. CUDA_LOG_DIRECTORY=. nvidia-cuda-mps-control 2>/dev/null")
        
        # Restore GPU clock frequency if it was changed
        if gpu_clock_saved and args.freq and gpu_clock_saved != args.freq:
            print(f"Restoring GPU clock to {gpu_clock_saved} MHz...")
            os.system(f"nvidia-smi -i {args.gpu} -lgc {gpu_clock_saved} 2>/dev/null")
        
        # Cleanup NCU log files
        for f in os.listdir('.'):
            if f.startswith('ncu_output_') and f.endswith('.csv'):
                try:
                    os.remove(f)
                except:
                    pass
    
    # Analyze capacity based on GPU utilization pattern
    # Capacity estimation: Find where GPU saturates (99%+) and stays there
    sorted_cells = sorted([int(c) for c in results.keys()])
    
    capacity_info = {
        "estimated_capacity": None,
        "saturation_point": None,  # First cell where GPU >= 95%
        "max_sustainable": None,   # Last cell before GPU drops to 0% or fails
        "status_per_cell": {},
        "scaling_analysis": {},    # Linear scaling analysis
        "bottleneck_analysis": {}  # Bottleneck detection
    }
    
    SATURATION_THRESHOLD = 95.0  # GPU% threshold for saturation
    CAPACITY_THRESHOLD = 99.0    # GPU% threshold for full capacity
    
    first_saturated = None
    last_sustainable = None
    capacity_cell = None
    
    # Collect data for scaling analysis
    gpu_utils = []
    gflops_list = []
    bw_list = []
    
    for i, cell in enumerate(sorted_cells):
        r = results[str(cell)]
        # For NCU-only mode, use NCU throughput as utilization proxy
        gpu_util = r.get("avg_gpu_util_pct", 0)
        if gpu_util == 0 and r.get("ncu_avg_sm_throughput_pct", 0) > 0:
            gpu_util = r.get("ncu_avg_sm_throughput_pct", 0)
        
        # Prefer measured values over estimated
        gflops = r.get("measured_total_gflops", 0) or r.get("estimated_gflops", 0)
        bw = r.get("measured_bandwidth_gbps", 0) or r.get("estimated_bandwidth_gbps", 0)
        
        gpu_utils.append(gpu_util)
        gflops_list.append(gflops)
        bw_list.append(bw)
        
        # Determine status for each cell
        if gpu_util < 10:
            status = "FAILED"
        elif gpu_util < SATURATION_THRESHOLD:
            status = "OK"
        elif gpu_util < CAPACITY_THRESHOLD:
            status = "NEAR_CAPACITY"
            if first_saturated is None:
                first_saturated = cell
        else:  # >= 99%
            status = "SATURATED"
            if first_saturated is None:
                first_saturated = cell
            if capacity_cell is None:
                capacity_cell = cell
        
        capacity_info["status_per_cell"][str(cell)] = status
        
        # Track last sustainable cell (before failure)
        if gpu_util >= SATURATION_THRESHOLD:
            last_sustainable = cell
    
    capacity_info["saturation_point"] = first_saturated
    capacity_info["max_sustainable"] = last_sustainable
    
    # Estimated capacity: the cell count where GPU first reaches 99%+ and stays there
    # This is comparable to measure.py's "100% on time" threshold
    if capacity_cell is not None:
        # Find the last cell before failure that maintained high utilization
        for cell in reversed(sorted_cells):
            if capacity_info["status_per_cell"][str(cell)] in ["SATURATED", "NEAR_CAPACITY"]:
                capacity_info["estimated_capacity"] = cell
                break
    
    # ========== LINEAR SCALING ANALYSIS ==========
    # Calculate scaling efficiency: how well does GPU util scale with cell count?
    # Ideal linear scaling: Cell 1 = X%, Cell 2 = 2X%, Cell 4 = 4X%, etc.
    scaling_analysis = {
        "is_linear_scaling": False,
        "scaling_efficiency_pct": 0,
        "gpu_util_per_cell": 0,
        "gflops_per_cell": 0,
        "linear_region_end": None,  # Last cell where scaling is still linear
        "plateau_start": None       # First cell where GFLOPS stops increasing
    }
    
    if len(sorted_cells) >= 2:
        # Calculate GPU util per cell in the linear region (before saturation)
        linear_cells = [c for c in sorted_cells if capacity_info["status_per_cell"][str(c)] == "OK"]
        
        if linear_cells:
            # Get utilization for first and last linear cell
            first_cell = linear_cells[0]
            last_linear_cell = linear_cells[-1]
            first_util = results[str(first_cell)].get("avg_gpu_util_pct", 0)
            last_util = results[str(last_linear_cell)].get("avg_gpu_util_pct", 0)
            
            # Calculate per-cell GPU utilization
            if first_cell > 0:
                scaling_analysis["gpu_util_per_cell"] = first_util / first_cell
            
            # Check linearity: is utilization roughly proportional to cell count?
            if len(linear_cells) >= 2 and first_util > 0:
                expected_last_util = first_util * (last_linear_cell / first_cell)
                actual_ratio = last_util / expected_last_util if expected_last_util > 0 else 0
                scaling_analysis["scaling_efficiency_pct"] = actual_ratio * 100
                scaling_analysis["is_linear_scaling"] = 0.8 <= actual_ratio <= 1.2  # Within 20%
                scaling_analysis["linear_region_end"] = last_linear_cell
        
        # Find plateau: where GFLOPS stops increasing significantly
        for i in range(1, len(sorted_cells)):
            prev_gflops = gflops_list[i-1]
            curr_gflops = gflops_list[i]
            if prev_gflops > 0:
                increase_pct = (curr_gflops - prev_gflops) / prev_gflops * 100
                # If GFLOPS increase is less than 5%, consider it a plateau
                if increase_pct < 5 and scaling_analysis["plateau_start"] is None:
                    scaling_analysis["plateau_start"] = sorted_cells[i]
        
        # Calculate average GFLOPS per cell in linear region
        if linear_cells:
            linear_gflops = [results[str(c)].get("estimated_gflops", 0) for c in linear_cells]
            if linear_cells[-1] > 0:
                scaling_analysis["gflops_per_cell"] = sum(linear_gflops) / len(linear_gflops) / (sum(linear_cells) / len(linear_cells))
    
    capacity_info["scaling_analysis"] = scaling_analysis
    
    # ========== BOTTLENECK ANALYSIS ==========
    # Determine if the system is GPU-bound or Memory-bound
    bottleneck_analysis = {
        "bottleneck_type": "UNKNOWN",
        "gpu_saturation_first": False,
        "memory_saturation_first": False,
        "peak_gflops": max(gflops_list) if gflops_list else 0,
        "peak_bandwidth_gbps": max(bw_list) if bw_list else 0,
        "peak_gpu_util": max(gpu_utils) if gpu_utils else 0,
        "theoretical_peak_gflops": calculate_theoretical_peak(args.gpu_type)["peak_fp32_gflops"],
        "theoretical_peak_bw_gbps": calculate_theoretical_peak(args.gpu_type)["peak_bandwidth_gbps"]
    }
    
    # Calculate utilization percentages
    if bottleneck_analysis["theoretical_peak_gflops"] > 0:
        bottleneck_analysis["compute_utilization_pct"] = (bottleneck_analysis["peak_gflops"] / 
                                                          bottleneck_analysis["theoretical_peak_gflops"]) * 100
    else:
        bottleneck_analysis["compute_utilization_pct"] = 0
    
    if bottleneck_analysis["theoretical_peak_bw_gbps"] > 0:
        bottleneck_analysis["memory_utilization_pct"] = (bottleneck_analysis["peak_bandwidth_gbps"] / 
                                                         bottleneck_analysis["theoretical_peak_bw_gbps"]) * 100
    else:
        bottleneck_analysis["memory_utilization_pct"] = 0
    
    # Determine bottleneck type
    # If GPU reaches high utilization before memory BW saturates -> GPU-bound (compute-bound)
    # If memory BW reaches high utilization before GPU saturates -> Memory-bound
    if bottleneck_analysis["peak_gpu_util"] >= 90:
        if bottleneck_analysis["memory_utilization_pct"] < 50:
            bottleneck_analysis["bottleneck_type"] = "GPU-BOUND (Compute Limited)"
            bottleneck_analysis["gpu_saturation_first"] = True
        else:
            bottleneck_analysis["bottleneck_type"] = "BALANCED"
    elif bottleneck_analysis["memory_utilization_pct"] >= 80:
        bottleneck_analysis["bottleneck_type"] = "MEMORY-BOUND (Bandwidth Limited)"
        bottleneck_analysis["memory_saturation_first"] = True
    else:
        bottleneck_analysis["bottleneck_type"] = "UNDER-UTILIZED"
    
    capacity_info["bottleneck_analysis"] = bottleneck_analysis
    
    # Save results
    # Get test configuration for JSON output
    first_vectors_yaml = f"vectors-{str(args.start).zfill(2)}.yaml"
    test_config_for_json = None
    if os.path.exists(first_vectors_yaml):
        test_config_for_json = parse_vectors_yaml(first_vectors_yaml)
    
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "measurement_mode": measurement_mode,
        "gpu_info": gpu_info,
        "gpu_type": args.gpu_type,
        "theoretical_peak": calculate_theoretical_peak(args.gpu_type),
        "test_configuration": test_config_for_json,
        "capacity_analysis": capacity_info,
        "config": {
            "cuphy": args.cuphy,
            "gpu": args.gpu,
            "gpu_clock_freq_mhz": args.freq,
            "tdd_pattern": args.tdd_pattern,
            "start": args.start,
            "cap": args.cap,
            "step": args.step,
            "target_sm_allocation": {"DL": args.target[0], "UL": args.target[1]},
            "delay": args.delay,
            "iterations": args.iterations,
            "use_ncu": use_ncu,
            "ncu_only": ncu_only,
            "ncu_iterations": args.ncu_iterations if use_ncu else None,
            "ncu_timeout": args.ncu_timeout if use_ncu else None
        },
        "results": results
    }
    
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {args.output}")
    
    # Generate plots
    if results:
        plot_file = plot_results(results, args.output.replace(".json", ""), args.gpu_type, 
                                  measurement_mode=measurement_mode)
        if args.show_plot and plot_file and HAS_MATPLOTLIB:
            print(f"Displaying plot: {plot_file}")
            plt.show()
    
    # Print summary table
    print("\n" + "=" * 70)
    print(f"Summary Table ({measurement_mode})")
    print("=" * 70)
    
    # Header varies based on mode
    if ncu_only:
        print(f"{'Cells':>6} | {'SM Tput%':>8} | {'GFLOPS':>12} | {'BW GB/s':>10} | {'Power W':>8} | {'Mem MB':>8} | {'Source':>10}")
    else:
        print(f"{'Cells':>6} | {'GPU%':>6} | {'GFLOPS':>12} | {'BW GB/s':>10} | {'Power W':>8} | {'Mem MB':>8} | {'Status':>12}")
    print("-" * 78)
    
    for cell in sorted_cells:
        r = results[str(cell)]
        # Use measured if available and > 0, otherwise use estimated
        measured_gflops = r.get("measured_total_gflops", 0)
        measured_bw = r.get("measured_bandwidth_gbps", 0)
        gflops = measured_gflops if measured_gflops > 0 else r.get("estimated_gflops", 0)
        bw = measured_bw if measured_bw > 0 else r.get("estimated_bandwidth_gbps", 0)
        
        # Source indicator
        if measured_gflops > 0:
            source = "✅ NCU"
            gflops_str = f"{gflops:>10.1f}*"
        else:
            source = "⚠️ Est."
            gflops_str = f"{gflops:>10.1f}~"
        
        if ncu_only:
            # For NCU-only mode, show SM throughput from NCU
            sm_tput = r.get("ncu_avg_sm_throughput_pct", 0)
            print(f"{cell:>6} | {sm_tput:>8.1f} | {gflops_str:>12} | {bw:>10.1f} | {r['avg_power_w']:>8.1f} | {r['avg_memory_used_mb']:>8.0f} | {source}")
        else:
            status = capacity_info["status_per_cell"].get(str(cell), "UNKNOWN")
            
            # Add visual indicators
            if status == "SATURATED":
                status_str = "█ SATURATED"
            elif status == "NEAR_CAPACITY":
                status_str = "▓ NEAR_CAP"
            elif status == "FAILED":
                status_str = "✗ FAILED"
            else:
                status_str = "○ OK"
            
            print(f"{cell:>6} | {r['avg_gpu_util_pct']:>6.1f} | {gflops_str:>12} | {bw:>10.1f} | {r['avg_power_w']:>8.1f} | {r['avg_memory_used_mb']:>8.0f} | {status_str}")
    
    # Legend
    print("-" * 78)
    print("  * = NCU measured (accurate),  ~ = nvidia-smi estimated")
    
    # Print capacity analysis
    print("-" * 70)
    print("\n" + "=" * 70)
    print("Capacity Analysis (based on GPU utilization)")
    print("=" * 70)
    if capacity_info["saturation_point"]:
        print(f"  GPU Saturation Point (≥95%): Cell {capacity_info['saturation_point']}")
    if capacity_info["estimated_capacity"]:
        print(f"  Estimated Max Capacity (99%): Cell {capacity_info['estimated_capacity']}")
        print(f"  ⚠️  Note: This is GPU utilization based. For accurate capacity,")
        print(f"      run measure.py which checks actual latency (100% on-time).")
    if capacity_info["max_sustainable"]:
        print(f"  Last Sustainable Cell:        Cell {capacity_info['max_sustainable']}")
    
    # Check for failures
    failed_cells = [c for c, s in capacity_info["status_per_cell"].items() if s == "FAILED"]
    if failed_cells:
        print(f"\n  ⚠️  Failed Cells: {', '.join(failed_cells)} (GPU util dropped to ~0%)")
    
    # Print Linear Scaling Analysis
    print("\n" + "=" * 70)
    print("Linear Scaling Analysis")
    print("=" * 70)
    scaling = capacity_info.get("scaling_analysis", {})
    if scaling.get("gpu_util_per_cell", 0) > 0:
        print(f"  GPU Util per Cell (linear region): ~{scaling['gpu_util_per_cell']:.1f}%")
        print(f"  GFLOPS per Cell (linear region):   ~{scaling['gflops_per_cell']:.0f}")
    if scaling.get("scaling_efficiency_pct", 0) > 0:
        efficiency = scaling["scaling_efficiency_pct"]
        if scaling.get("is_linear_scaling"):
            print(f"  Scaling Efficiency: {efficiency:.1f}% ✅ (Linear scaling confirmed)")
        else:
            print(f"  Scaling Efficiency: {efficiency:.1f}% ⚠️ (Non-linear scaling)")
    if scaling.get("linear_region_end"):
        print(f"  Linear Region: Cell 1 → Cell {scaling['linear_region_end']}")
    if scaling.get("plateau_start"):
        print(f"  GFLOPS Plateau Starts: Cell {scaling['plateau_start']} (bottleneck reached)")
    
    # Print Bottleneck Analysis
    print("\n" + "=" * 70)
    print("Bottleneck Analysis")
    print("=" * 70)
    bottleneck = capacity_info.get("bottleneck_analysis", {})
    print(f"  Peak GFLOPS Achieved:    {bottleneck.get('peak_gflops', 0):.0f} GFLOPS")
    print(f"  Peak Bandwidth Achieved: {bottleneck.get('peak_bandwidth_gbps', 0):.1f} GB/s")
    print(f"  Peak GPU Utilization:    {bottleneck.get('peak_gpu_util', 0):.1f}%")
    print(f"")
    print(f"  Theoretical Peak GFLOPS: {bottleneck.get('theoretical_peak_gflops', 0):.0f} GFLOPS")
    print(f"  Theoretical Peak BW:     {bottleneck.get('theoretical_peak_bw_gbps', 0):.0f} GB/s")
    print(f"")
    compute_util = bottleneck.get('compute_utilization_pct', 0)
    memory_util = bottleneck.get('memory_utilization_pct', 0)
    print(f"  Compute Utilization:     {compute_util:.1f}% of theoretical peak")
    print(f"  Memory BW Utilization:   {memory_util:.1f}% of theoretical peak")
    print(f"")
    
    bottleneck_type = bottleneck.get('bottleneck_type', 'UNKNOWN')
    if bottleneck_type == "GPU-BOUND (Compute Limited)":
        print(f"  🔥 Bottleneck: {bottleneck_type}")
        print(f"     → GPU saturates before memory bandwidth limit")
        print(f"     → System is compute-limited (typical for cuPHY)")
    elif bottleneck_type == "MEMORY-BOUND (Bandwidth Limited)":
        print(f"  🔥 Bottleneck: {bottleneck_type}")
        print(f"     → Memory bandwidth saturates before GPU")
        print(f"     → Consider optimizing memory access patterns")
    elif bottleneck_type == "BALANCED":
        print(f"  ✅ Bottleneck: {bottleneck_type}")
        print(f"     → Both GPU and Memory are well utilized")
    else:
        print(f"  ⚠️ Bottleneck: {bottleneck_type}")
        print(f"     → System resources are under-utilized")
    
    print("=" * 70)
    
    # Print expected graph interpretation guide
    print("\n" + "=" * 70)
    print("📊 Graph Interpretation Guide")
    print("=" * 70)
    print("  Expected Pattern for Normal Operation:")
    print("  ┌────────────────────────────────────────────────────────────┐")
    print("  │ GPU%  100│              ████████████████  ← Saturation    │")
    print("  │         │           ███                                   │")
    print("  │      75│        ███        Linear Scaling Region         │")
    print("  │         │     ███                                         │")
    print("  │      50│   ██                                             │")
    print("  │         │ ██                                               │")
    print("  │      25│█                                                  │")
    print("  │        └──┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬──→    │")
    print("  │           1   2   3   4   5   6   7   8   9  10  11  Cells│")
    print("  └────────────────────────────────────────────────────────────┘")
    print("")
    print("  ✅ Good Signs:")
    print("     - Linear increase in GPU% as cells increase (before saturation)")
    print("     - GFLOPS increases proportionally with cells")
    print("     - Clear saturation point (capacity limit)")
    print("")
    print("  ⚠️ Warning Signs:")
    print("     - Non-linear scaling (efficiency < 80%)")
    print("     - Early plateau in GFLOPS")
    print("     - Memory BW saturates before GPU (memory-bound)")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    main()
