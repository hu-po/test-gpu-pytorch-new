""" Mock training script for testing GPU support."""

import argparse
import os
import pprint
import time

import platform
import wandb
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, NVMLError

parser = argparse.ArgumentParser(description='Mock training script for testing GPU support.')
parser.add_argument("--project", type=str, default="summerhack", help="W&B project")
parser.add_argument('--gpu', type=str, default="0", help='Comma seperated list of GPU ids to use')

# How long should this fake training script run for?
parser.add_argument("--train_time", type=int, default=30)

# TODO: args?
TARGET_GPU_UTILIZATION = 0.9
MIN_LOG_INTERVAL = 0.1
GPU_MEM_GROWTH_RATE = 1.07
INITIAL_TENSOR_SIZE = 8

if __name__ == "__main__":
    args = parser.parse_args()
    wandb.login(key=os.getenv('WANDB_API_KEY'))
    run = wandb.init(project=args.project)
    run.log_code(".")

    # Print out wandb information
    print('\n\n---\tWandB Information\t---\n')
    print(f'\tentity: {run._entity or os.environ.get("WANDB_ENTITY")}')
    print(f'\tproject: {run._project or os.environ.get("WANDB_PROJECT")}')
    print(f'\tconfig: {pprint.pformat(wandb.config)}\n')

    # Print out some system information
    print('\n---\tSystem Information\t---\n')
    print(f'\tsystem: {platform.system()}')
    print(f'\tarchitecture: {platform.architecture()}')
    print(f'\tprocessor: {platform.processor()}')
    print(f'\tmachine: {platform.machine()}')
    print(f'\tpython_version: {platform.python_version()}')

    try:
        import torch
    except Exception as e:
        raise ImportError(f'Error importing PyTorch: {e}')

    # Tell framework to use specific GPUs
    print('\n---\tGPU Information\t---\n')
    print(f'\tPyTorch was able to find CUDA: {torch.cuda.is_available()}')
    print(f'\tPyTorch was able to find {torch.cuda.device_count()} GPUs')
    assert torch.cuda.is_available(), 'Torch was unable to find CUDA'

    # Use NVIDIA Python bindings to get GPU information
    nvmlInit()
    nvidia_devices = {}
    for id in args.gpu.split(','):
        try:
            _id = int(id)
            _nvidia_device = nvmlDeviceGetHandleByIndex(_id)
        except (NVMLError, TypeError) as e:
            raise AssertionError(f"Error getting device handle to GPU {_id}: {e}")
        print(f'\tGPU {id} has {nvmlDeviceGetMemoryInfo(_nvidia_device).total} total memory')
        print(f'\tGPU {id} has {nvmlDeviceGetMemoryInfo(_nvidia_device).free} free memory')
        print(f'\tGPU {id} has {nvmlDeviceGetMemoryInfo(_nvidia_device).used} used memory')
        nvidia_devices[_id] = _nvidia_device
    assert len(nvidia_devices) > 0, 'No NVIDIA GPUs found'

    # Register devices with PyTorch
    devices = {}
    for id in nvidia_devices.keys():
        assert 0 <= id < torch.cuda.device_count(), f'Invalid GPU index {id}'
        devices[id] = torch.device(f'cuda:{id}')
        print(f'\tPytorch: created device using GPU {id}')

    # Increase size of tensor over time until max GPU memory utilization
    tensor_size = {id: INITIAL_TENSOR_SIZE for id in devices.keys()}

    # Do some fake taining
    assert args.train_time > 0, "Fake training must last longer than 0 seconds."
    print('\n---\tFake Training\t---\n')
    start_time = time.time()
    old_time_remaining = args.train_time
    while time.time() - start_time < args.train_time:
        time_remaining = int(args.train_time - (time.time() - start_time))
        # Don't spam too much
        if old_time_remaining - time_remaining < MIN_LOG_INTERVAL:
            continue
        old_time_remaining = time_remaining
        print(f'\tTraining, {time_remaining} seconds remaining.')
        wandb.log({"time.remaining": time_remaining})
        wandb.log({"time.now": time.time()})

        for id, device in devices.items():
            _tensor_size = int(tensor_size[id])
            a = torch.randn(_tensor_size, _tensor_size).to(device)
            b = torch.randn(_tensor_size, _tensor_size).to(device)
            c = torch.mm(a, b).to(device)
            _used = nvmlDeviceGetMemoryInfo(_nvidia_device).used
            _total = nvmlDeviceGetMemoryInfo(_nvidia_device).total
            utilization = _used / _total
            wandb.log({f"gpu.mem.utilization.{id}": utilization})
            print(f'\t\tGPU {id} Memory Utilization at {utilization}')
            if utilization < TARGET_GPU_UTILIZATION:
                tensor_size[id] *= GPU_MEM_GROWTH_RATE
            print(f'\t\t A ({_tensor_size},{_tensor_size}) x B ({_tensor_size},{_tensor_size})')

    wandb.finish()
