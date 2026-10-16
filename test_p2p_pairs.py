#!/usr/bin/env python3
"""Test GPU P2P connectivity for all pairs on BI-V100."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

import torch

num_gpus = torch.cuda.device_count()
print(f"Visible GPUs: {num_gpus}")

ok_pairs = []
fail_pairs = []

for i in range(num_gpus):
    for j in range(i+1, num_gpus):
        can = torch.cuda.can_device_access_peer(i, j)
        if can:
            ok_pairs.append(f"{i}-{j}")
        else:
            fail_pairs.append(f"{i}-{j}")

print(f"P2P OK: {','.join(ok_pairs)}")
print(f"P2P FAIL: {','.join(fail_pairs) or 'none'}")

# Functional: copy data between each OK pair
for pair in ok_pairs:
    i, j = map(int, pair.split('-'))
    src = torch.randn(4096, device=f'cuda:{i}')
    dst = torch.empty(4096, device=f'cuda:{j}')
    dst.copy_(src)
    torch.cuda.synchronize()
    diff = (dst.cpu() - src.cpu()).abs().max().item()
    assert diff == 0.0, f"Copy {i}->{j} diff={diff}"
    print(f"  {i}<->{j}: copy OK")

print("P2P test PASSED")
