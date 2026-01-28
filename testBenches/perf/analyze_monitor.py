#!/usr/bin/env python3
"""
Resource Monitor Analysis Script (Unified)
==========================================
GPU/CPU/Memory 모니터링 로그를 분석하고 그래프를 생성합니다.

지원하는 입력 형식:
1. start_monitor.sh 출력 (새 형식)
   - gpu_monitor.csv, cpu_monitor.csv, memory_monitor.csv
   
2. nvidia-smi 직접 출력 (기존 형식)
   - {prefix}_gpu.csv (nvidia-smi --query-gpu)
   - {prefix}_gpu_dmon.txt (nvidia-smi dmon)
   - {prefix}_vmstat.txt (vmstat)

Usage:
    # 새 형식 (디렉토리 지정)
    python3 analyze_monitor.py ./monitor_logs
    
    # 기존 형식 (prefix 지정)
    python3 analyze_monitor.py ./monitoring_results/monitor_20260127_120000 --legacy
    
    # 출력 파일 지정
    python3 analyze_monitor.py ./monitor_logs --output ./results/analysis
"""

import os
import sys
import argparse
import glob
from datetime import datetime

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# =============================================================================
# 새 형식 파서 (start_monitor.sh 출력)
# =============================================================================

def load_new_gpu_data(filepath):
    """Load GPU monitoring CSV (new format from start_monitor.sh)."""
    if not os.path.exists(filepath):
        return None
    
    if HAS_PANDAS:
        try:
            df = pd.read_csv(filepath)
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d_%H:%M:%S.%f', errors='coerce')
            return df
        except Exception as e:
            print(f"⚠️ Failed to load GPU data: {e}")
            return None
    return None


def load_new_cpu_data(filepath):
    """Load CPU monitoring CSV (new format)."""
    if not os.path.exists(filepath):
        return None
    
    if HAS_PANDAS:
        try:
            df = pd.read_csv(filepath)
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d_%H:%M:%S.%f', errors='coerce')
            return df
        except Exception as e:
            print(f"⚠️ Failed to load CPU data: {e}")
            return None
    return None


def load_new_memory_data(filepath):
    """Load memory monitoring CSV (new format)."""
    if not os.path.exists(filepath):
        return None
    
    if HAS_PANDAS:
        try:
            df = pd.read_csv(filepath)
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d_%H:%M:%S.%f', errors='coerce')
            return df
        except Exception as e:
            print(f"⚠️ Failed to load memory data: {e}")
            return None
    return None


# =============================================================================
# 기존 형식 파서 (nvidia-smi, vmstat 직접 출력)
# =============================================================================

def parse_legacy_gpu_csv(filepath):
    """Parse nvidia-smi --query-gpu CSV output (legacy format)."""
    if not os.path.exists(filepath):
        return None
    
    try:
        df = pd.read_csv(filepath, skipinitialspace=True)
        df.columns = df.columns.str.strip()
        
        # 새 형식으로 변환
        result = pd.DataFrame()
        result['timestamp'] = pd.to_datetime('now')  # 타임스탬프 없으면 현재 시간
        
        # GPU 사용률
        util_col = [c for c in df.columns if 'utilization.gpu' in c.lower()]
        if util_col:
            result['gpu_util_pct'] = df[util_col[0]].astype(str).str.replace(' %', '').str.replace('%', '').astype(float)
        
        # 메모리 사용률
        mem_util_col = [c for c in df.columns if 'utilization.memory' in c.lower()]
        if mem_util_col:
            result['mem_util_pct'] = df[mem_util_col[0]].astype(str).str.replace(' %', '').str.replace('%', '').astype(float)
        
        # 메모리 사용량
        mem_used_col = [c for c in df.columns if 'memory.used' in c.lower()]
        if mem_used_col:
            result['mem_used_mb'] = df[mem_used_col[0]].astype(str).str.replace(' MiB', '').astype(float)
        
        # 전력
        power_col = [c for c in df.columns if 'power.draw' in c.lower()]
        if power_col:
            result['power_w'] = df[power_col[0]].astype(str).str.replace(' W', '').astype(float)
        
        # 온도
        temp_col = [c for c in df.columns if 'temperature' in c.lower()]
        if temp_col:
            result['temp_c'] = df[temp_col[0]].astype(float)
        
        return result
    except Exception as e:
        print(f"⚠️ Legacy GPU CSV parsing error: {e}")
        return None


def parse_gpu_dmon(filepath):
    """Parse nvidia-smi dmon output."""
    if not os.path.exists(filepath):
        return None
    
    data = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                parts = line.split()
                if len(parts) >= 10 and parts[0].isdigit():
                    data.append({
                        'gpu_index': int(parts[0]),
                        'power_w': float(parts[1]) if parts[1] != '-' else 0,
                        'temp_c': int(parts[2]) if parts[2] != '-' else 0,
                        'gpu_util_pct': int(parts[3]) if parts[3] != '-' else 0,
                        'mem_util_pct': int(parts[4]) if parts[4] != '-' else 0,
                        'enc_util': int(parts[5]) if parts[5] != '-' else 0,
                        'dec_util': int(parts[6]) if parts[6] != '-' else 0,
                        'mem_clock_mhz': int(parts[7]) if parts[7] != '-' else 0,
                        'sm_clock_mhz': int(parts[8]) if parts[8] != '-' else 0,
                    })
        if data:
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime('now')
            return df
        return None
    except Exception as e:
        print(f"⚠️ GPU dmon parsing error: {e}")
        return None


def parse_vmstat(filepath):
    """Parse vmstat output."""
    if not os.path.exists(filepath):
        return None
    
    data = []
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            for line in lines[2:]:  # 헤더 스킵
                parts = line.split()
                if len(parts) >= 17 and parts[0].isdigit():
                    data.append({
                        'r': int(parts[0]),       # 실행 대기 프로세스
                        'b': int(parts[1]),       # 블록된 프로세스
                        'swpd': int(parts[2]),
                        'free': int(parts[3]),
                        'buff': int(parts[4]),
                        'cache': int(parts[5]),
                        'si': int(parts[6]),
                        'so': int(parts[7]),
                        'bi': int(parts[8]),      # 블록 입력
                        'bo': int(parts[9]),      # 블록 출력
                        'interrupts': int(parts[10]),
                        'context_switches': int(parts[11]),
                        'user_pct': int(parts[12]),
                        'system_pct': int(parts[13]),
                        'idle_pct': int(parts[14]),
                        'iowait_pct': int(parts[15]),
                        'stolen_pct': int(parts[16]),
                    })
        if data:
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime('now')
            return df
        return None
    except Exception as e:
        print(f"⚠️ vmstat parsing error: {e}")
        return None


# =============================================================================
# 통계 분석
# =============================================================================

def analyze_gpu(df):
    """Analyze GPU data and return statistics."""
    if df is None or len(df) == 0:
        return None
    
    stats = {}
    
    if 'gpu_util_pct' in df.columns:
        stats['gpu_util'] = {
            'min': float(df['gpu_util_pct'].min()),
            'max': float(df['gpu_util_pct'].max()),
            'avg': float(df['gpu_util_pct'].mean()),
            'std': float(df['gpu_util_pct'].std()) if len(df) > 1 else 0
        }
    
    if 'mem_util_pct' in df.columns:
        stats['mem_util'] = {
            'min': float(df['mem_util_pct'].min()),
            'max': float(df['mem_util_pct'].max()),
            'avg': float(df['mem_util_pct'].mean()),
            'std': float(df['mem_util_pct'].std()) if len(df) > 1 else 0
        }
    
    if 'mem_used_mb' in df.columns:
        stats['mem_used'] = {
            'min': float(df['mem_used_mb'].min()),
            'max': float(df['mem_used_mb'].max()),
            'avg': float(df['mem_used_mb'].mean())
        }
    
    if 'power_w' in df.columns:
        stats['power'] = {
            'min': float(df['power_w'].min()),
            'max': float(df['power_w'].max()),
            'avg': float(df['power_w'].mean())
        }
    
    if 'temp_c' in df.columns:
        stats['temp'] = {
            'min': float(df['temp_c'].min()),
            'max': float(df['temp_c'].max()),
            'avg': float(df['temp_c'].mean())
        }
    
    if 'sm_clock_mhz' in df.columns:
        stats['sm_clock'] = {
            'min': float(df['sm_clock_mhz'].min()),
            'max': float(df['sm_clock_mhz'].max()),
            'avg': float(df['sm_clock_mhz'].mean())
        }
    
    return stats


def analyze_cpu(df):
    """Analyze CPU data and return statistics."""
    if df is None or len(df) == 0:
        return None
    
    # 'all' CPU 또는 전체 데이터 사용
    if 'cpu_id' in df.columns:
        df_all = df[df['cpu_id'] == 'all'] if 'all' in df['cpu_id'].values else df
    else:
        df_all = df
    
    stats = {}
    
    if 'user_pct' in df_all.columns:
        stats['user'] = {
            'min': float(df_all['user_pct'].min()),
            'max': float(df_all['user_pct'].max()),
            'avg': float(df_all['user_pct'].mean())
        }
    
    if 'system_pct' in df_all.columns:
        stats['system'] = {
            'min': float(df_all['system_pct'].min()),
            'max': float(df_all['system_pct'].max()),
            'avg': float(df_all['system_pct'].mean())
        }
    
    if 'idle_pct' in df_all.columns:
        stats['idle'] = {
            'min': float(df_all['idle_pct'].min()),
            'max': float(df_all['idle_pct'].max()),
            'avg': float(df_all['idle_pct'].mean())
        }
    
    if 'iowait_pct' in df_all.columns:
        stats['iowait'] = {
            'min': float(df_all['iowait_pct'].min()),
            'max': float(df_all['iowait_pct'].max()),
            'avg': float(df_all['iowait_pct'].mean())
        }
    
    # vmstat 특수 필드
    if 'context_switches' in df_all.columns:
        stats['context_switches'] = {
            'min': float(df_all['context_switches'].min()),
            'max': float(df_all['context_switches'].max()),
            'avg': float(df_all['context_switches'].mean())
        }
    
    if 'interrupts' in df_all.columns:
        stats['interrupts'] = {
            'min': float(df_all['interrupts'].min()),
            'max': float(df_all['interrupts'].max()),
            'avg': float(df_all['interrupts'].mean())
        }
    
    return stats


def analyze_memory(df):
    """Analyze memory data and return statistics."""
    if df is None or len(df) == 0:
        return None
    
    stats = {}
    
    if 'used_pct' in df.columns:
        stats['used_pct'] = {
            'min': float(df['used_pct'].min()),
            'max': float(df['used_pct'].max()),
            'avg': float(df['used_pct'].mean())
        }
    
    if 'used_mb' in df.columns:
        stats['used_mb'] = {
            'min': float(df['used_mb'].min()),
            'max': float(df['used_mb'].max()),
            'avg': float(df['used_mb'].mean())
        }
    
    if 'available_mb' in df.columns:
        stats['available_mb'] = {
            'min': float(df['available_mb'].min()),
            'max': float(df['available_mb'].max()),
            'avg': float(df['available_mb'].mean())
        }
    
    return stats


# =============================================================================
# 그래프 생성
# =============================================================================

def generate_graphs(gpu_df, cpu_df, mem_df, output_prefix, dmon_df=None):
    """Generate resource usage graphs."""
    if not HAS_MATPLOTLIB:
        print("⚠️ matplotlib not available. Skipping graph generation.")
        return None
    
    # 사용 가능한 데이터 확인
    available_data = []
    if gpu_df is not None and len(gpu_df) > 0:
        available_data.append(('gpu', gpu_df))
    if dmon_df is not None and len(dmon_df) > 0:
        available_data.append(('dmon', dmon_df))
    if cpu_df is not None and len(cpu_df) > 0:
        available_data.append(('cpu', cpu_df))
    if mem_df is not None and len(mem_df) > 0:
        available_data.append(('mem', mem_df))
    
    if not available_data:
        print("⚠️ No data available for graphing.")
        return None
    
    n_plots = len(available_data)
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 4*n_plots), sharex=False)
    if n_plots == 1:
        axes = [axes]
    
    colors = {
        'primary': '#2E86AB',
        'secondary': '#A23B72',
        'accent': '#F18F01',
        'good': '#28965A',
        'warning': '#E9C46A',
        'danger': '#C73E1D'
    }
    
    plot_idx = 0
    
    for data_type, df in available_data:
        ax = axes[plot_idx]
        
        if data_type == 'gpu':
            # GPU Utilization Plot
            if 'timestamp' in df.columns and df['timestamp'].notna().any():
                x_data = df['timestamp']
                x_label = 'Time'
            else:
                x_data = range(len(df))
                x_label = 'Sample'
            
            if 'gpu_util_pct' in df.columns:
                ax.plot(x_data, df['gpu_util_pct'], label='GPU Util (%)', 
                        color=colors['primary'], linewidth=1.5)
            
            if 'mem_util_pct' in df.columns:
                ax.plot(x_data, df['mem_util_pct'], label='Memory Util (%)', 
                        color=colors['secondary'], linewidth=1.5)
            
            # Power on secondary axis
            if 'power_w' in df.columns:
                ax2 = ax.twinx()
                ax2.plot(x_data, df['power_w'], label='Power (W)', 
                         color=colors['accent'], linewidth=1.5, linestyle='--')
                ax2.set_ylabel('Power (W)', color=colors['accent'])
                ax2.tick_params(axis='y', labelcolor=colors['accent'])
                ax2.legend(loc='upper right')
            
            ax.set_ylabel('Utilization (%)')
            ax.set_title('GPU Resource Usage', fontweight='bold')
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 105)
            ax.set_xlabel(x_label)
            
            # Add stats annotation
            gpu_stats = analyze_gpu(df)
            if gpu_stats and 'gpu_util' in gpu_stats:
                stats_text = f"GPU: Avg={gpu_stats['gpu_util']['avg']:.1f}%, Max={gpu_stats['gpu_util']['max']:.1f}%"
                ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        elif data_type == 'dmon':
            # dmon detailed plot
            x_data = range(len(df))
            
            if 'gpu_util_pct' in df.columns:
                ax.plot(x_data, df['gpu_util_pct'], label='SM Util (%)', 
                        color=colors['primary'], linewidth=1.5)
            
            if 'mem_util_pct' in df.columns:
                ax.plot(x_data, df['mem_util_pct'], label='Mem Util (%)', 
                        color=colors['secondary'], linewidth=1.5)
            
            ax.set_ylabel('Utilization (%)')
            ax.set_title('GPU Detailed Metrics (dmon)', fontweight='bold')
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 105)
            ax.set_xlabel('Sample')
        
        elif data_type == 'cpu':
            # CPU Plot
            if 'timestamp' in df.columns and df['timestamp'].notna().any():
                # Filter for 'all' CPU
                if 'cpu_id' in df.columns and 'all' in df['cpu_id'].values:
                    df_plot = df[df['cpu_id'] == 'all'].copy()
                else:
                    df_plot = df.copy()
                
                if len(df_plot) > 0:
                    x_data = df_plot['timestamp']
                    x_label = 'Time'
                else:
                    x_data = range(len(df))
                    x_label = 'Sample'
                    df_plot = df
            else:
                x_data = range(len(df))
                x_label = 'Sample'
                df_plot = df
            
            if 'user_pct' in df_plot.columns:
                ax.fill_between(x_data, 0, df_plot['user_pct'], 
                               label='User', color=colors['primary'], alpha=0.7)
            
            if 'system_pct' in df_plot.columns and 'user_pct' in df_plot.columns:
                ax.fill_between(x_data, df_plot['user_pct'], 
                               df_plot['user_pct'] + df_plot['system_pct'],
                               label='System', color=colors['secondary'], alpha=0.7)
            
            ax.set_ylabel('CPU Usage (%)')
            ax.set_title('CPU Resource Usage', fontweight='bold')
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 105)
            ax.set_xlabel(x_label)
            
            # Add stats
            cpu_stats = analyze_cpu(df)
            if cpu_stats:
                total_avg = cpu_stats.get('user', {}).get('avg', 0) + cpu_stats.get('system', {}).get('avg', 0)
                stats_text = f"CPU Total: Avg={total_avg:.1f}%"
                ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        elif data_type == 'mem':
            # Memory Plot
            if 'timestamp' in df.columns and df['timestamp'].notna().any():
                x_data = df['timestamp']
                x_label = 'Time'
            else:
                x_data = range(len(df))
                x_label = 'Sample'
            
            if 'used_pct' in df.columns:
                ax.fill_between(x_data, 0, df['used_pct'], 
                               label='Used (%)', color=colors['good'], alpha=0.7)
            
            ax.set_ylabel('Memory Usage (%)')
            ax.set_title('System Memory Usage', fontweight='bold')
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 105)
            ax.set_xlabel(x_label)
            
            # Add stats
            mem_stats = analyze_memory(df)
            if mem_stats and 'used_pct' in mem_stats:
                stats_text = f"Memory: Avg={mem_stats['used_pct']['avg']:.1f}%, Max={mem_stats['used_pct']['max']:.1f}%"
                ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plot_idx += 1
    
    plt.suptitle('Resource Monitoring Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    graph_path = f"{output_prefix}_resource_monitor.png"
    plt.savefig(graph_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Graph saved: {graph_path}")
    return graph_path


# =============================================================================
# 리포트 생성
# =============================================================================

def print_console_summary(gpu_stats, cpu_stats, mem_stats, dmon_stats=None):
    """Print summary to console."""
    print("\n" + "=" * 70)
    print("              RESOURCE MONITORING SUMMARY")
    print("=" * 70)
    
    if gpu_stats:
        print("\n📊 GPU:")
        if 'gpu_util' in gpu_stats:
            print(f"   Utilization : Avg={gpu_stats['gpu_util']['avg']:.1f}%, "
                  f"Max={gpu_stats['gpu_util']['max']:.1f}%, "
                  f"Min={gpu_stats['gpu_util']['min']:.1f}%")
        if 'mem_util' in gpu_stats:
            print(f"   Memory Util : Avg={gpu_stats['mem_util']['avg']:.1f}%, "
                  f"Max={gpu_stats['mem_util']['max']:.1f}%")
        if 'mem_used' in gpu_stats:
            print(f"   VRAM Used   : Avg={gpu_stats['mem_used']['avg']:.0f} MB, "
                  f"Max={gpu_stats['mem_used']['max']:.0f} MB")
        if 'power' in gpu_stats:
            print(f"   Power       : Avg={gpu_stats['power']['avg']:.1f} W, "
                  f"Max={gpu_stats['power']['max']:.1f} W")
        if 'temp' in gpu_stats:
            print(f"   Temperature : Avg={gpu_stats['temp']['avg']:.1f}°C, "
                  f"Max={gpu_stats['temp']['max']:.1f}°C")
        if 'sm_clock' in gpu_stats:
            print(f"   SM Clock    : Avg={gpu_stats['sm_clock']['avg']:.0f} MHz, "
                  f"Max={gpu_stats['sm_clock']['max']:.0f} MHz")
    
    if dmon_stats:
        print("\n📈 GPU (dmon detailed):")
        if 'gpu_util' in dmon_stats:
            print(f"   SM Util     : Avg={dmon_stats['gpu_util']['avg']:.1f}%, "
                  f"Max={dmon_stats['gpu_util']['max']:.1f}%")
    
    if cpu_stats:
        print("\n💻 CPU:")
        total_avg = cpu_stats.get('user', {}).get('avg', 0) + cpu_stats.get('system', {}).get('avg', 0)
        total_max = cpu_stats.get('user', {}).get('max', 0) + cpu_stats.get('system', {}).get('max', 0)
        print(f"   Total Usage : Avg={total_avg:.1f}%, Max={total_max:.1f}%")
        if 'user' in cpu_stats:
            print(f"   User        : Avg={cpu_stats['user']['avg']:.1f}%")
        if 'system' in cpu_stats:
            print(f"   System      : Avg={cpu_stats['system']['avg']:.1f}%")
        if 'iowait' in cpu_stats:
            print(f"   I/O Wait    : Avg={cpu_stats['iowait']['avg']:.1f}%")
        if 'context_switches' in cpu_stats:
            print(f"   Ctx Switch  : Avg={cpu_stats['context_switches']['avg']:.0f}/s, "
                  f"Max={cpu_stats['context_switches']['max']:.0f}/s")
        if 'interrupts' in cpu_stats:
            print(f"   Interrupts  : Avg={cpu_stats['interrupts']['avg']:.0f}/s")
    
    if mem_stats:
        print("\n🧠 Memory:")
        if 'used_pct' in mem_stats:
            print(f"   Usage       : Avg={mem_stats['used_pct']['avg']:.1f}%, "
                  f"Max={mem_stats['used_pct']['max']:.1f}%")
        if 'used_mb' in mem_stats:
            print(f"   Used        : Avg={mem_stats['used_mb']['avg']:.0f} MB, "
                  f"Max={mem_stats['used_mb']['max']:.0f} MB")
    
    print("\n" + "=" * 70)


def generate_text_report(gpu_stats, cpu_stats, mem_stats, output_prefix, dmon_stats=None):
    """Generate text report file."""
    report_path = f"{output_prefix}_resource_summary.txt"
    
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("           RESOURCE MONITORING SUMMARY REPORT\n")
        f.write("=" * 70 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        if gpu_stats:
            f.write("-" * 50 + "\n")
            f.write(" GPU Statistics\n")
            f.write("-" * 50 + "\n")
            if 'gpu_util' in gpu_stats:
                f.write(f"  GPU Utilization:\n")
                f.write(f"    Average : {gpu_stats['gpu_util']['avg']:.2f}%\n")
                f.write(f"    Maximum : {gpu_stats['gpu_util']['max']:.2f}%\n")
                f.write(f"    Minimum : {gpu_stats['gpu_util']['min']:.2f}%\n")
                f.write(f"    Std Dev : {gpu_stats['gpu_util']['std']:.2f}%\n")
            if 'mem_util' in gpu_stats:
                f.write(f"\n  GPU Memory Utilization:\n")
                f.write(f"    Average : {gpu_stats['mem_util']['avg']:.2f}%\n")
                f.write(f"    Maximum : {gpu_stats['mem_util']['max']:.2f}%\n")
            if 'mem_used' in gpu_stats:
                f.write(f"\n  GPU VRAM Usage:\n")
                f.write(f"    Average : {gpu_stats['mem_used']['avg']:.0f} MB\n")
                f.write(f"    Maximum : {gpu_stats['mem_used']['max']:.0f} MB\n")
            if 'power' in gpu_stats:
                f.write(f"\n  Power Consumption:\n")
                f.write(f"    Average : {gpu_stats['power']['avg']:.2f} W\n")
                f.write(f"    Maximum : {gpu_stats['power']['max']:.2f} W\n")
            if 'temp' in gpu_stats:
                f.write(f"\n  Temperature:\n")
                f.write(f"    Average : {gpu_stats['temp']['avg']:.1f}°C\n")
                f.write(f"    Maximum : {gpu_stats['temp']['max']:.1f}°C\n")
            f.write("\n")
        
        if dmon_stats:
            f.write("-" * 50 + "\n")
            f.write(" GPU Detailed Statistics (dmon)\n")
            f.write("-" * 50 + "\n")
            if 'gpu_util' in dmon_stats:
                f.write(f"  SM Utilization:\n")
                f.write(f"    Average : {dmon_stats['gpu_util']['avg']:.2f}%\n")
                f.write(f"    Maximum : {dmon_stats['gpu_util']['max']:.2f}%\n")
            f.write("\n")
        
        if cpu_stats:
            f.write("-" * 50 + "\n")
            f.write(" CPU Statistics\n")
            f.write("-" * 50 + "\n")
            total_avg = cpu_stats.get('user', {}).get('avg', 0) + cpu_stats.get('system', {}).get('avg', 0)
            f.write(f"  Total CPU Usage:\n")
            f.write(f"    Average : {total_avg:.2f}%\n")
            if 'user' in cpu_stats:
                f.write(f"\n  User Space:\n")
                f.write(f"    Average : {cpu_stats['user']['avg']:.2f}%\n")
                f.write(f"    Maximum : {cpu_stats['user']['max']:.2f}%\n")
            if 'system' in cpu_stats:
                f.write(f"\n  Kernel Space:\n")
                f.write(f"    Average : {cpu_stats['system']['avg']:.2f}%\n")
                f.write(f"    Maximum : {cpu_stats['system']['max']:.2f}%\n")
            if 'iowait' in cpu_stats:
                f.write(f"\n  I/O Wait:\n")
                f.write(f"    Average : {cpu_stats['iowait']['avg']:.2f}%\n")
            if 'context_switches' in cpu_stats:
                f.write(f"\n  Context Switches:\n")
                f.write(f"    Average : {cpu_stats['context_switches']['avg']:.0f}/s\n")
                f.write(f"    Maximum : {cpu_stats['context_switches']['max']:.0f}/s\n")
            f.write("\n")
        
        if mem_stats:
            f.write("-" * 50 + "\n")
            f.write(" Memory Statistics\n")
            f.write("-" * 50 + "\n")
            if 'used_pct' in mem_stats:
                f.write(f"  Usage Percentage:\n")
                f.write(f"    Average : {mem_stats['used_pct']['avg']:.2f}%\n")
                f.write(f"    Maximum : {mem_stats['used_pct']['max']:.2f}%\n")
            if 'used_mb' in mem_stats:
                f.write(f"\n  Usage (MB):\n")
                f.write(f"    Average : {mem_stats['used_mb']['avg']:.0f} MB\n")
                f.write(f"    Maximum : {mem_stats['used_mb']['max']:.0f} MB\n")
            f.write("\n")
        
        f.write("=" * 70 + "\n")
    
    print(f"✅ Text report saved: {report_path}")
    return report_path


def generate_markdown_report(gpu_stats, cpu_stats, mem_stats, output_prefix, dmon_stats=None):
    """Generate markdown report file."""
    report_path = f"{output_prefix}_resource_report.md"
    
    with open(report_path, 'w') as f:
        f.write("# Resource Monitoring Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Summary\n\n")
        f.write("| Resource | Metric | Average | Maximum | Minimum |\n")
        f.write("|----------|--------|---------|---------|--------|\n")
        
        if gpu_stats and 'gpu_util' in gpu_stats:
            f.write(f"| GPU | Utilization | {gpu_stats['gpu_util']['avg']:.1f}% | {gpu_stats['gpu_util']['max']:.1f}% | {gpu_stats['gpu_util']['min']:.1f}% |\n")
        if gpu_stats and 'mem_util' in gpu_stats:
            f.write(f"| GPU | Memory Util | {gpu_stats['mem_util']['avg']:.1f}% | {gpu_stats['mem_util']['max']:.1f}% | {gpu_stats['mem_util']['min']:.1f}% |\n")
        if gpu_stats and 'power' in gpu_stats:
            f.write(f"| GPU | Power | {gpu_stats['power']['avg']:.1f} W | {gpu_stats['power']['max']:.1f} W | {gpu_stats['power']['min']:.1f} W |\n")
        if gpu_stats and 'temp' in gpu_stats:
            f.write(f"| GPU | Temperature | {gpu_stats['temp']['avg']:.1f}°C | {gpu_stats['temp']['max']:.1f}°C | {gpu_stats['temp']['min']:.1f}°C |\n")
        
        if cpu_stats:
            total_avg = cpu_stats.get('user', {}).get('avg', 0) + cpu_stats.get('system', {}).get('avg', 0)
            total_max = cpu_stats.get('user', {}).get('max', 0) + cpu_stats.get('system', {}).get('max', 0)
            f.write(f"| CPU | Total Usage | {total_avg:.1f}% | {total_max:.1f}% | - |\n")
        
        if mem_stats and 'used_pct' in mem_stats:
            f.write(f"| Memory | Usage | {mem_stats['used_pct']['avg']:.1f}% | {mem_stats['used_pct']['max']:.1f}% | {mem_stats['used_pct']['min']:.1f}% |\n")
        
        f.write("\n")
        
        if gpu_stats:
            f.write("## GPU Details\n\n")
            if 'gpu_util' in gpu_stats:
                f.write(f"- **GPU Utilization:** Avg={gpu_stats['gpu_util']['avg']:.2f}%, Std={gpu_stats['gpu_util']['std']:.2f}%\n")
            if 'mem_used' in gpu_stats:
                f.write(f"- **VRAM Usage:** Avg={gpu_stats['mem_used']['avg']:.0f} MB, Max={gpu_stats['mem_used']['max']:.0f} MB\n")
            f.write("\n")
        
        if cpu_stats:
            f.write("## CPU Details\n\n")
            if 'user' in cpu_stats:
                f.write(f"- **User Space:** Avg={cpu_stats['user']['avg']:.2f}%\n")
            if 'system' in cpu_stats:
                f.write(f"- **Kernel Space:** Avg={cpu_stats['system']['avg']:.2f}%\n")
            if 'context_switches' in cpu_stats:
                f.write(f"- **Context Switches:** Avg={cpu_stats['context_switches']['avg']:.0f}/s\n")
            f.write("\n")
        
        f.write("---\n")
        f.write("*Generated by analyze_monitor.py*\n")
    
    print(f"✅ Markdown report saved: {report_path}")
    return report_path


# =============================================================================
# 메인
# =============================================================================

def detect_format(path):
    """Detect input format (new or legacy)."""
    if os.path.isdir(path):
        # Check for new format files
        if os.path.exists(os.path.join(path, 'gpu_monitor.csv')):
            return 'new'
        # Check for legacy format files
        csv_files = glob.glob(os.path.join(path, '*_gpu.csv'))
        if csv_files:
            return 'legacy'
    else:
        # Path is a prefix
        if os.path.exists(f"{path}_gpu.csv"):
            return 'legacy'
    
    return 'unknown'


def load_data(args):
    """Load data based on format."""
    gpu_df = None
    cpu_df = None
    mem_df = None
    dmon_df = None
    
    if args.legacy:
        # Legacy format: prefix_gpu.csv, prefix_vmstat.txt, etc.
        prefix = args.path
        
        gpu_file = f"{prefix}_gpu.csv"
        if os.path.exists(gpu_file):
            print(f"Loading: {gpu_file}")
            gpu_df = parse_legacy_gpu_csv(gpu_file)
        
        dmon_file = f"{prefix}_gpu_dmon.txt"
        if os.path.exists(dmon_file):
            print(f"Loading: {dmon_file}")
            dmon_df = parse_gpu_dmon(dmon_file)
        
        vmstat_file = f"{prefix}_vmstat.txt"
        if os.path.exists(vmstat_file):
            print(f"Loading: {vmstat_file}")
            cpu_df = parse_vmstat(vmstat_file)
    else:
        # New format: directory with gpu_monitor.csv, etc.
        monitor_dir = args.path
        
        gpu_file = os.path.join(monitor_dir, 'gpu_monitor.csv')
        if os.path.exists(gpu_file):
            print(f"Loading: {gpu_file}")
            gpu_df = load_new_gpu_data(gpu_file)
        
        cpu_file = os.path.join(monitor_dir, 'cpu_monitor.csv')
        if os.path.exists(cpu_file):
            print(f"Loading: {cpu_file}")
            cpu_df = load_new_cpu_data(cpu_file)
        
        mem_file = os.path.join(monitor_dir, 'memory_monitor.csv')
        if os.path.exists(mem_file):
            print(f"Loading: {mem_file}")
            mem_df = load_new_memory_data(mem_file)
    
    return gpu_df, cpu_df, mem_df, dmon_df


def main():
    parser = argparse.ArgumentParser(
        description='Analyze resource monitoring logs (unified script)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # New format (start_monitor.sh output)
  python3 analyze_monitor.py ./monitor_logs
  
  # Legacy format (nvidia-smi, vmstat output)
  python3 analyze_monitor.py ./monitoring_results/monitor_20260127 --legacy
  
  # Specify output prefix
  python3 analyze_monitor.py ./monitor_logs --output ./results/analysis
        """
    )
    parser.add_argument('path', help='Monitor directory (new) or log prefix (legacy)')
    parser.add_argument('--output', '-o', help='Output prefix for generated files')
    parser.add_argument('--legacy', action='store_true', 
                        help='Use legacy format (nvidia-smi CSV, vmstat)')
    parser.add_argument('--no-graph', action='store_true', help='Skip graph generation')
    parser.add_argument('--markdown', action='store_true', help='Generate markdown report')
    args = parser.parse_args()
    
    # Auto-detect format if not specified
    if not args.legacy:
        detected = detect_format(args.path)
        if detected == 'legacy':
            print("Auto-detected legacy format")
            args.legacy = True
        elif detected == 'unknown':
            # Try to find latest log
            if os.path.isdir(args.path):
                legacy_files = glob.glob(os.path.join(args.path, '*_gpu.csv'))
                if legacy_files:
                    latest = max(legacy_files, key=os.path.getctime)
                    args.path = latest.replace('_gpu.csv', '')
                    args.legacy = True
                    print(f"Using latest legacy log: {args.path}")
    
    # Validate path
    if args.legacy:
        if not os.path.exists(f"{args.path}_gpu.csv") and not os.path.exists(f"{args.path}_vmstat.txt"):
            print(f"❌ Error: No valid log files found with prefix: {args.path}")
            sys.exit(1)
    else:
        if not os.path.isdir(args.path):
            print(f"❌ Error: Directory not found: {args.path}")
            sys.exit(1)
    
    # Set output prefix
    if args.output:
        output_prefix = args.output
    elif args.legacy:
        output_prefix = args.path
    else:
        output_prefix = os.path.join(args.path, 'analysis')
    
    # Load data
    print("\n📂 Loading monitoring data...")
    gpu_df, cpu_df, mem_df, dmon_df = load_data(args)
    
    if gpu_df is None and cpu_df is None and mem_df is None and dmon_df is None:
        print("❌ No monitoring data found!")
        sys.exit(1)
    
    # Analyze
    print("\n📊 Analyzing data...")
    gpu_stats = analyze_gpu(gpu_df) if gpu_df is not None else None
    cpu_stats = analyze_cpu(cpu_df) if cpu_df is not None else None
    mem_stats = analyze_memory(mem_df) if mem_df is not None else None
    dmon_stats = analyze_gpu(dmon_df) if dmon_df is not None else None
    
    # Print console summary
    print_console_summary(gpu_stats, cpu_stats, mem_stats, dmon_stats)
    
    # Generate outputs
    if not args.no_graph and HAS_MATPLOTLIB and HAS_PANDAS:
        print("\n📈 Generating graphs...")
        generate_graphs(gpu_df, cpu_df, mem_df, output_prefix, dmon_df)
    
    print("\n📝 Generating reports...")
    generate_text_report(gpu_stats, cpu_stats, mem_stats, output_prefix, dmon_stats)
    
    if args.markdown:
        generate_markdown_report(gpu_stats, cpu_stats, mem_stats, output_prefix, dmon_stats)
    
    print("\n✅ Analysis complete!")


if __name__ == '__main__':
    main()
