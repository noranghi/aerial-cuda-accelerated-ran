# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
measure_k8s.py - Kubernetes/Container 환경용 measure.py
nvidia-smi 실패 시에도 테스트가 진행되도록 수정됨
"""

import os
import subprocess

def get_gpu_status(gpu_id):
    """
    nvidia-smi로 GPU 상태를 가져옴.
    실패 시 기본값 반환 (K8s/컨테이너 환경 대응)
    """
    gpuStatSave = {
        'clockFreq': 1980,      # 기본값
        'powerLimit': 700.0,    # 기본값
        'persistMode': 'Enabled' # 기본값 (컨테이너에서는 보통 enabled)
    }
    
    try:
        command = f"nvidia-smi -i {gpu_id} --query-gpu=clocks.current.graphics,power.limit,persistence_mode --format=csv,noheader"
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                shell=True, encoding="utf-8", timeout=10)
        
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split()
            if len(parts) >= 5:
                gpuStatSave['clockFreq'] = int(parts[0])
                gpuStatSave['powerLimit'] = float(parts[2])
                gpuStatSave['persistMode'] = parts[4]
                print(f"[K8s] GPU 상태 확인 성공: freq={gpuStatSave['clockFreq']}, power={gpuStatSave['powerLimit']}, persist={gpuStatSave['persistMode']}")
            else:
                print(f"[K8s] nvidia-smi 출력 파싱 실패, 기본값 사용: {result.stdout}")
        else:
            print(f"[K8s] nvidia-smi 실행 실패 (returncode={result.returncode}), 기본값 사용")
            if result.stderr:
                print(f"[K8s] stderr: {result.stderr[:200]}")
    except subprocess.TimeoutExpired:
        print("[K8s] nvidia-smi 타임아웃, 기본값 사용")
    except Exception as e:
        print(f"[K8s] nvidia-smi 에러: {e}, 기본값 사용")
    
    return gpuStatSave


def set_gpu_config(gpu_id, freq, power, is_GH200, sudo=""):
    """
    GPU 설정 변경 시도. 실패해도 계속 진행.
    """
    try:
        if is_GH200:
            os.system(f"{sudo}nvidia-smi -i {gpu_id} -lgc {freq} --mode=1 2>/dev/null")
        else:
            os.system(f"{sudo}nvidia-smi -i {gpu_id} -lgc {freq} 2>/dev/null")
        print(f"[K8s] GPU clock 설정 시도: {freq} MHz")
    except Exception as e:
        print(f"[K8s] GPU clock 설정 실패 (무시됨): {e}")

    if power is not None:
        try:
            os.system(f"{sudo}nvidia-smi -i {gpu_id} -pl {power} 2>/dev/null")
            print(f"[K8s] GPU power limit 설정 시도: {power} W")
        except Exception as e:
            print(f"[K8s] GPU power 설정 실패 (무시됨): {e}")


def restore_gpu_config(gpu_id, gpuStatSave, args, sudo=""):
    """
    GPU 설정 복원 시도. 실패해도 무시.
    """
    try:
        if gpuStatSave['clockFreq'] != args.freq:
            if args.is_GH200:
                os.system(f"{sudo}nvidia-smi -i {gpu_id} -lgc {gpuStatSave['clockFreq']} --mode=1 2>/dev/null")
            else:
                os.system(f"{sudo}nvidia-smi -i {gpu_id} -lgc {gpuStatSave['clockFreq']} 2>/dev/null")
        
        if (args.power is not None) and (gpuStatSave['powerLimit'] != args.power):
            os.system(f"{sudo}nvidia-smi -i {gpu_id} -pl {gpuStatSave['powerLimit']} 2>/dev/null")
                
        if gpuStatSave['persistMode'] == 'Disabled':
            os.system(f"{sudo}nvidia-smi -i {gpu_id} -pm 0 2>/dev/null")
        
        print("[K8s] GPU 설정 복원 시도 완료")
    except Exception as e:
        print(f"[K8s] GPU 설정 복원 실패 (무시됨): {e}")


if __name__ == "__main__":

    from measure.cli import arguments

    print("=" * 60)
    print("[K8s] measure_k8s.py - Kubernetes/Container 환경용")
    print("=" * 60)

    if os.geteuid() == 0:
        sudo = ""
    else:
        sudo = "sudo "

    base, args = arguments()

    # GPU 상태 가져오기 (실패 시 기본값 사용)
    gpuStatSave = get_gpu_status(args.gpu)
    
    # GPU 설정 변경 시도 (실패해도 계속 진행)
    if gpuStatSave['clockFreq'] != args.freq:
        set_gpu_config(args.gpu, args.freq, args.power, args.is_GH200, sudo)
    
    # persistence mode 설정 시도
    if gpuStatSave['persistMode'] == 'Disabled':
        try:
            os.system(f"{sudo}nvidia-smi -i {args.gpu} -pm 1 2>/dev/null")
        except:
            pass
    
    # start testing
    if args.mig is not None:

        # Clean up any existing MIG instances first
        result = os.system(f"{sudo}nvidia-smi mig -i {args.gpu} -dci > buffer.txt 2>&1")
        if result != 0:
            if os.path.exists("buffer.txt"):
                os.remove("buffer.txt")
            base.error("Failed to destroy MIG compute instances. Make sure you have proper permissions.")
            
        result = os.system(f"{sudo}nvidia-smi mig -i {args.gpu} -dgi > buffer.txt 2>&1")
        if result != 0:
            if os.path.exists("buffer.txt"):
                os.remove("buffer.txt")
            base.error("Failed to destroy MIG GPU instances. Make sure you have proper permissions.")
        
        # Disable MIG mode first (to clean state)
        result = os.system(f"{sudo}nvidia-smi -i {args.gpu} -mig 0 > buffer.txt 2>&1")
        if result != 0:
            if os.path.exists("buffer.txt"):
                os.remove("buffer.txt")
            base.error("Failed to disable MIG mode. Make sure you have proper permissions and the GPU supports MIG.")
        
        # Enable MIG mode
        result = os.system(f"{sudo}nvidia-smi -i {args.gpu} -mig 1 > buffer.txt 2>&1")
        if result != 0:
            if os.path.exists("buffer.txt"):
                os.remove("buffer.txt")
            base.error("Failed to enable MIG mode. Make sure you have proper permissions and MIG-capable GPU.")

        ifile = open("buffer.txt", "r")
        lines = ifile.readlines()
        ifile.close()
        os.remove("buffer.txt")

        if lines[0].split()[0] == "Enabled" and lines[0].split()[1] == "MIG":
            import measure.mig

            measure.mig.measure(base, args)
            
        else:

            base.error("encountered issues in enabling MIG on the selected GPU")

    else:

        import measure.nomig

        measure.nomig.measure(base, args)

    # GPU 설정 복원 (실패해도 무시)
    restore_gpu_config(args.gpu, gpuStatSave, args, sudo)
    
    print("=" * 60)
    print("[K8s] 테스트 완료")
    print("=" * 60)
