#!/usr/bin/env python3
"""
PRB (Physical Resource Block) 자원 사용량 변화 측정 스크립트

Usage:
    python3 measure_prb_usage.py \
        --cuphy ../build \
        --vectors /workspace/aerial-cuda-accelerated-ran/testVectors \
        --config testcases_avg_F08.json \
        --gpu 1 \
        --output prb_usage_results.json
"""

import argparse
import subprocess
import json
import time
import os
import re

# PRB 설정 (testcases_avg_F08.json 기반)
PRB_CONFIGS = {
    "F08-PP-00": {"prb": 273, "mimo": "4x4", "qam": 256, "type": "peak"},
    "F08-AC-01": {"prb": 32, "mimo": "4x4", "qam": 256, "type": "average"},
    "F08-AC-02": {"prb": 64, "mimo": "4x4", "qam": 256, "type": "average"},
    "F08-AC-03": {"prb": 88, "mimo": "4x4", "qam": 256, "type": "average"},
    "F08-AC-04": {"prb": 120, "mimo": "4x4", "qam": 256, "type": "average"},
    "F08-AC-05": {"prb": 144, "mimo": "4x4", "qam": 256, "type": "average"},
    "F08-AC-06": {"prb": 176, "mimo": "4x4", "qam": 256, "type": "average"},
    "F08-AC-07": {"prb": 208, "mimo": "4x4", "qam": 256, "type": "average"},
    "F08-AM-01": {"prb": 40, "mimo": "2x4", "qam": 64, "type": "average-mid"},
    "F08-AM-02": {"prb": 72, "mimo": "2x4", "qam": 64, "type": "average-mid"},
    "F08-AM-03": {"prb": 104, "mimo": "2x4", "qam": 64, "type": "average-mid"},
    "F08-AM-04": {"prb": 136, "mimo": "2x4", "qam": 64, "type": "average-mid"},
    "F08-AM-05": {"prb": 168, "mimo": "2x4", "qam": 64, "type": "average-mid"},
    "F08-AM-06": {"prb": 200, "mimo": "2x4", "qam": 64, "type": "average-mid"},
    "F08-AM-07": {"prb": 232, "mimo": "2x4", "qam": 64, "type": "average-mid"},
    "F08-AE-01": {"prb": 24, "mimo": "1x4", "qam": 16, "type": "average-edge"},
    "F08-AE-02": {"prb": 48, "mimo": "1x4", "qam": 16, "type": "average-edge"},
    "F08-AE-03": {"prb": 64, "mimo": "1x4", "qam": 16, "type": "average-edge"},
    "F08-AE-04": {"prb": 88, "mimo": "1x4", "qam": 16, "type": "average-edge"},
    "F08-AE-05": {"prb": 104, "mimo": "1x4", "qam": 16, "type": "average-edge"},
    "F08-AE-06": {"prb": 128, "mimo": "1x4", "qam": 16, "type": "average-edge"},
    "F08-AE-07": {"prb": 144, "mimo": "1x4", "qam": 16, "type": "average-edge"},
}


def create_prb_use_case(base_uc, prb_config, cell_count, output_path):
    """특정 PRB 설정으로 use case 파일 생성"""
    with open(base_uc, 'r') as f:
        uc_data = json.load(f)
    
    # Peak 설정을 선택된 PRB 설정으로 교체
    new_uc = {}
    peak_key = f"Peak: {cell_count}"
    
    if peak_key in uc_data:
        new_uc[peak_key] = {
            "Average: 0": {
                "F08 - PDSCH": [prb_config] * cell_count,
                "F08 - PUSCH": [prb_config] * cell_count,
                "F08 - SSB": ["F08-PP-00"] * cell_count,  # SSB는 Peak 유지
                "F08 - PRACH": ["F08-PP-00"] * cell_count,
                "F08 - PDCCH": ["F08-PP-00"] * cell_count,
                "F08 - CSIRS": ["F08-PP-00"] * cell_count,
                "F08 - PUCCH": ["F08-PP-00"] * cell_count,
                "F08 - MAC": ["F08-PP-00"],
                "F08 - MAC2": ["F08-PP-00"]
            }
        }
    
    with open(output_path, 'w') as f:
        json.dump(new_uc, f, indent=4)
    
    return output_path


def run_test_with_prb(args, prb_config_name, cell_count, gpu_id, target):
    """특정 PRB 설정으로 테스트 실행 및 GPU 메트릭 수집"""
    
    prb_info = PRB_CONFIGS.get(prb_config_name, PRB_CONFIGS["F08-PP-00"])
    
    # 임시 use case 파일 생성
    temp_uc = f"/tmp/uc_prb_{prb_info['prb']}_{cell_count}.json"
    create_prb_use_case(args.base_uc, prb_config_name, cell_count, temp_uc)
    
    # vectors.yaml 파일 경로
    vectors_yaml = os.path.join(args.vectors, f"vectors-{str(cell_count).zfill(2)}.yaml")
    
    if not os.path.exists(vectors_yaml):
        print(f"Warning: {vectors_yaml} not found, skipping...")
        return None
    
    # cubb_gpu_test_bench 실행
    test_cmd = f"""
    CUDA_VISIBLE_DEVICES={gpu_id} \
    CUDA_MPS_PIPE_DIRECTORY=. \
    CUDA_LOG_DIRECTORY=. \
    {args.cuphy}/cubb_gpu_test_bench/cubb_gpu_test_bench \
    -i {vectors_yaml} \
    -r 50 -w 10 -u 5 -d 0 -m 1 \
    --M {target[0]},{target[1]} \
    --U --D
    """
    
    print(f"\nTesting PRB={prb_info['prb']}, Cells={cell_count}, MIMO={prb_info['mimo']}")
    
    # 백그라운드에서 테스트 실행
    test_process = subprocess.Popen(
        test_cmd, shell=True, 
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    
    # GPU 메트릭 수집
    time.sleep(0.5)  # 테스트 시작 대기
    
    metrics = {
        "gpu_util": [],
        "mem_util": [],
        "power": [],
        "memory_used": []
    }
    
    sample_count = 0
    while test_process.poll() is None and sample_count < 30:
        try:
            result = subprocess.run(
                f"nvidia-smi -i {gpu_id} --query-gpu=utilization.gpu,utilization.memory,power.draw,memory.used --format=csv,noheader,nounits",
                shell=True, capture_output=True, text=True
            )
            parts = [p.strip() for p in result.stdout.strip().split(',')]
            if len(parts) >= 4:
                metrics["gpu_util"].append(float(parts[0]))
                metrics["mem_util"].append(float(parts[1]))
                metrics["power"].append(float(parts[2]))
                metrics["memory_used"].append(float(parts[3]))
        except Exception as e:
            pass
        time.sleep(0.2)
        sample_count += 1
    
    # 테스트 완료 대기
    stdout, stderr = test_process.communicate(timeout=60)
    
    # 결과에서 처리 시간 파싱
    output = stdout.decode('utf-8', errors='ignore')
    
    # latency 파싱 (예: "DL latency: 123.45 us")
    dl_latency = None
    ul_latency = None
    
    dl_match = re.search(r'DL.*?latency.*?(\d+\.?\d*)\s*us', output, re.IGNORECASE)
    ul_match = re.search(r'UL.*?latency.*?(\d+\.?\d*)\s*us', output, re.IGNORECASE)
    
    if dl_match:
        dl_latency = float(dl_match.group(1))
    if ul_match:
        ul_latency = float(ul_match.group(1))
    
    # PRB 기반 throughput 계산 (이론값)
    # Throughput = PRB * 12 subcarriers * symbols * bits/symbol * coding_rate
    prb = prb_info['prb']
    symbols_per_slot = 14
    subcarriers_per_prb = 12
    
    # QAM bits
    qam_bits = {16: 4, 64: 6, 256: 8}
    bits_per_symbol = qam_bits.get(prb_info['qam'], 8)
    
    # MIMO layers
    mimo_layers = int(prb_info['mimo'].split('x')[0])
    
    # 이론적 throughput (Mbps) - simplified calculation
    # 실제로는 coding rate, overhead 등 고려 필요
    slot_duration_ms = 0.5  # 30kHz SCS
    bits_per_slot = prb * subcarriers_per_prb * (symbols_per_slot - 2) * bits_per_symbol * mimo_layers * 0.9  # ~90% coding
    throughput_mbps = bits_per_slot / slot_duration_ms / 1000
    
    # 메트릭 평균 계산
    result = {
        "prb_config": prb_config_name,
        "prb": prb,
        "mimo": prb_info['mimo'],
        "qam": prb_info['qam'],
        "cell_count": cell_count,
        "avg_gpu_util": sum(metrics["gpu_util"]) / len(metrics["gpu_util"]) if metrics["gpu_util"] else 0,
        "avg_mem_util": sum(metrics["mem_util"]) / len(metrics["mem_util"]) if metrics["mem_util"] else 0,
        "avg_power_w": sum(metrics["power"]) / len(metrics["power"]) if metrics["power"] else 0,
        "avg_memory_mb": sum(metrics["memory_used"]) / len(metrics["memory_used"]) if metrics["memory_used"] else 0,
        "dl_latency_us": dl_latency,
        "ul_latency_us": ul_latency,
        "theoretical_throughput_mbps": throughput_mbps,
        "prb_utilization_pct": (prb / 273) * 100  # PRB 사용률 (273 = 100MHz max)
    }
    
    print(f"  PRB Utilization: {result['prb_utilization_pct']:.1f}%")
    print(f"  GPU Util: {result['avg_gpu_util']:.1f}%, Memory: {result['avg_memory_mb']:.0f} MB")
    print(f"  Theoretical Throughput: {throughput_mbps:.1f} Mbps")
    
    return result


def plot_prb_results(results, output_prefix):
    """PRB 별 결과 그래프 생성"""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib/numpy not installed, skipping plot")
        return
    
    # PRB 별로 그룹화
    prb_groups = {}
    for r in results:
        prb = r['prb']
        if prb not in prb_groups:
            prb_groups[prb] = []
        prb_groups[prb].append(r)
    
    prbs = sorted(prb_groups.keys())
    
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('PRB Resource Usage vs GPU Metrics', fontsize=14)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(prbs)))
    
    # 1. PRB vs GPU Utilization
    for idx, prb in enumerate(prbs):
        cells = [r['cell_count'] for r in prb_groups[prb]]
        gpu_util = [r['avg_gpu_util'] for r in prb_groups[prb]]
        axs[0, 0].plot(cells, gpu_util, 'o-', color=colors[idx], label=f'PRB={prb}')
    axs[0, 0].set_xlabel('Cell Count')
    axs[0, 0].set_ylabel('GPU Utilization (%)')
    axs[0, 0].set_title('GPU Utilization by PRB')
    axs[0, 0].legend(fontsize=8, ncol=2)
    axs[0, 0].grid(True, alpha=0.3)
    
    # 2. PRB vs Memory Usage
    for idx, prb in enumerate(prbs):
        cells = [r['cell_count'] for r in prb_groups[prb]]
        memory = [r['avg_memory_mb'] for r in prb_groups[prb]]
        axs[0, 1].plot(cells, memory, 'o-', color=colors[idx], label=f'PRB={prb}')
    axs[0, 1].set_xlabel('Cell Count')
    axs[0, 1].set_ylabel('Memory Usage (MB)')
    axs[0, 1].set_title('Memory Usage by PRB')
    axs[0, 1].legend(fontsize=8, ncol=2)
    axs[0, 1].grid(True, alpha=0.3)
    
    # 3. PRB vs Power
    for idx, prb in enumerate(prbs):
        cells = [r['cell_count'] for r in prb_groups[prb]]
        power = [r['avg_power_w'] for r in prb_groups[prb]]
        axs[1, 0].plot(cells, power, 'o-', color=colors[idx], label=f'PRB={prb}')
    axs[1, 0].set_xlabel('Cell Count')
    axs[1, 0].set_ylabel('Power Draw (W)')
    axs[1, 0].set_title('Power Consumption by PRB')
    axs[1, 0].legend(fontsize=8, ncol=2)
    axs[1, 0].grid(True, alpha=0.3)
    
    # 4. PRB Utilization vs Throughput
    prb_util = [r['prb_utilization_pct'] for r in results]
    throughput = [r['theoretical_throughput_mbps'] for r in results]
    cell_counts = [r['cell_count'] for r in results]
    
    scatter = axs[1, 1].scatter(prb_util, throughput, c=cell_counts, cmap='plasma', s=100)
    axs[1, 1].set_xlabel('PRB Utilization (%)')
    axs[1, 1].set_ylabel('Theoretical Throughput (Mbps)')
    axs[1, 1].set_title('PRB Utilization vs Throughput')
    plt.colorbar(scatter, ax=axs[1, 1], label='Cell Count')
    axs[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_prb_plot.png", dpi=150)
    print(f"\nPlot saved: {output_prefix}_prb_plot.png")


def main():
    parser = argparse.ArgumentParser(description="PRB Resource Usage Measurement")
    parser.add_argument("--cuphy", type=str, required=True, help="cuPHY build directory")
    parser.add_argument("--vectors", type=str, required=True, help="testVectors directory")
    parser.add_argument("--config", type=str, required=True, help="Test cases config file")
    parser.add_argument("--base_uc", type=str, default="uc_avg_F08_TDD.json", help="Base use case file")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument("--start", type=int, default=4, help="Start cell count")
    parser.add_argument("--cap", type=int, default=12, help="Max cell count")
    parser.add_argument("--step", type=int, default=2, help="Cell count step")
    parser.add_argument("--target", nargs=2, type=int, default=[60, 60], help="SM target [DL UL]")
    parser.add_argument("--prb_configs", nargs='+', default=["F08-PP-00", "F08-AC-03", "F08-AC-05", "F08-AC-07"],
                       help="PRB configs to test")
    parser.add_argument("--output", type=str, default="prb_usage_results.json", help="Output file")
    
    args = parser.parse_args()
    
    print("="*60)
    print("PRB Resource Usage Measurement")
    print("="*60)
    print(f"Configs to test: {args.prb_configs}")
    print(f"Cell range: {args.start} - {args.cap} (step {args.step})")
    print(f"SM target: DL={args.target[0]}, UL={args.target[1]}")
    
    all_results = []
    
    for prb_config in args.prb_configs:
        if prb_config not in PRB_CONFIGS:
            print(f"Warning: Unknown PRB config '{prb_config}', skipping")
            continue
            
        print(f"\n{'='*40}")
        print(f"Testing PRB Config: {prb_config}")
        print(f"PRB={PRB_CONFIGS[prb_config]['prb']}, "
              f"MIMO={PRB_CONFIGS[prb_config]['mimo']}, "
              f"QAM={PRB_CONFIGS[prb_config]['qam']}")
        print(f"{'='*40}")
        
        for cell_count in range(args.start, args.cap + 1, args.step):
            result = run_test_with_prb(args, prb_config, cell_count, args.gpu, args.target)
            if result:
                all_results.append(result)
            time.sleep(1)
    
    # 결과 저장
    output_data = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "prb_configs": args.prb_configs,
            "cell_range": [args.start, args.cap, args.step],
            "sm_target": args.target
        },
        "results": all_results
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=4)
    print(f"\nResults saved: {args.output}")
    
    # 그래프 생성
    if all_results:
        plot_prb_results(all_results, args.output.replace('.json', ''))
    
    # 요약 출력
    print("\n" + "="*60)
    print("Summary by PRB Configuration")
    print("="*60)
    
    for prb_config in args.prb_configs:
        config_results = [r for r in all_results if r['prb_config'] == prb_config]
        if config_results:
            avg_gpu = sum(r['avg_gpu_util'] for r in config_results) / len(config_results)
            avg_mem = sum(r['avg_memory_mb'] for r in config_results) / len(config_results)
            prb_info = PRB_CONFIGS[prb_config]
            print(f"\n{prb_config} (PRB={prb_info['prb']}):")
            print(f"  Avg GPU Util: {avg_gpu:.1f}%")
            print(f"  Avg Memory: {avg_mem:.0f} MB")
            print(f"  PRB Utilization: {prb_info['prb']/273*100:.1f}%")


if __name__ == "__main__":
    main()
