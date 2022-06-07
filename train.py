""" Mock training script for testing GPU support."""

import argparse
import os
import pprint
import time

import platform
import wandb

parser = argparse.ArgumentParser(description='Mock training script for testing GPU support.')
parser.add_argument("--project", type=str, default="launch-examples", help="W&B project")
parser.add_argument('--gpu', type=int, default=-1, help='Specify which GPU to use (-1 for all gpus)')

# How long should this fake training script run for?
parser.add_argument("--train_time", type=int, default=5)

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
    print(f'\tPyTorch was able to find GPUs: {torch.cuda.device_count()}')
    assert torch.cuda.is_available(), 'Torch was unable to find CUDA'

    if args.gpu >= 0:
        assert args.gpu < torch.cuda.device_count(), f'Invalid GPU {args.gpu} specified: found {torch.cuda.device_count()}'
        print(f'\tUsing GPU {args.gpu}')
        _device = torch.device(f'cuda:{args.gpu}')
    else:
        print(f'\tUsing all {torch.cuda.device_count()} available GPUs')
        _device = torch.device("cuda")

    # Do some fake taining
    assert args.train_time > 0, "Fake training must last longer than 0 seconds."
    print('\n---\tFake Training\t---\n')
    start_time = time.time()
    while time.time() - start_time < args.train_time:
        a = torch.randn(100, 100).to(_device)
        b = torch.randn(100, 100).to(_device)
        c = torch.mm(a, b).to(_device)
        time_remaining = int(args.train_time - (time.time() - start_time))
        print(f'\tTraining, {time_remaining} seconds remaining.')
        wandb.log({"alive_time": time_remaining})
        time.sleep(1)

    wandb.finish()
