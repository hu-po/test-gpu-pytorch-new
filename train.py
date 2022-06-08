""" Mock training script for testing GPU support."""

import argparse
import os
import pprint
import time

import platform
import wandb
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

parser = argparse.ArgumentParser(description='Mock training script for testing GPU support.')
parser.add_argument("--project", type=str, default="launch-examples", help="W&B project")
parser.add_argument('--gpu', nargs="+", type=int, default=0, help='Specify which GPUs to use')

# How long should this fake training script run for?
parser.add_argument("--train_time", type=int, default=5)

# Target steady state gpu utilization
parser.add_argument("--target_gpu_utilization", type=float, default=0.9)

if __name__ == "__main__":
    args = parser.parse_args()
    wandb.login(key=os.getenv('WANDB_API_KEY'))
    run = wandb.init(project=args.project)

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

    devices = {}
    if len(args.gpu) > 1:
        for _gpu in args.gpu:
            assert 0 <= _gpu < torch.cuda.device_count(), f'Invalid GPU index {_gpu}'
            print(f'\tUsing GPU {args.gpu}')
            devices[_gpu] = torch.device(f'cuda:{args.gpu}')

    # Use NVIDIA Python bindings to get GPU information
    nvmlInit()
    nvidia_devices = {}
    for id, device in devices:
        _nvidia_device = nvmlDeviceGetHandleByIndex(id)
        print(f'\tGPU {id} has {nvmlDeviceGetMemoryInfo(_nvidia_device).total} total memory')
        print(f'\tGPU {id} has {nvmlDeviceGetMemoryInfo(_nvidia_device).free} free memory')
        print(f'\tGPU {id} has {nvmlDeviceGetMemoryInfo(_nvidia_device).used} used memory')
        nvidia_devices[id] = _nvidia_device
    
    # Increase size of tensor over time until max GPU memory utilization
    tensor_size = {id: 2 for id in devices.keys()}

    # Do some fake taining
    assert args.train_time > 0, "Fake training must last longer than 0 seconds."
    print('\n---\tFake Training\t---\n')
    start_time = time.time()
    while time.time() - start_time < args.train_time:
        time_remaining = int(args.train_time - (time.time() - start_time))
        print(f'\tTraining, {time_remaining} seconds remaining.')
        wandb.log({"time.remaining": time_remaining})
        wandb.log({"time.now": time.time()})

        for id, device in devices:
            _tensor_size = int(tensor_size[id])
            a = torch.randn(_tensor_size, _tensor_size).to(device)
            b = torch.randn(_tensor_size, _tensor_size).to(device)
            c = torch.mm(a, b).to(device)
            _used = nvmlDeviceGetMemoryInfo(_nvidia_device).used
            _total = nvmlDeviceGetMemoryInfo(_nvidia_device).total
            utilization = _used / _total
            wandb.log({f"gpu.utilization.{id}": utilization})
            print(f'\t\tGPU {id} Utilization at {utilization}')
            if utilization < args.target_gpu_utilization:
                tensor_size[id] *= 1.03
            print(f'\t\t A ({_tensor_size},{_tensor_size}) x B ({_tensor_size},{_tensor_size})')

    wandb.finish()
