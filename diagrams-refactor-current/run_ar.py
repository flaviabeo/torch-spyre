"""Two-rank driver: AllReduceWithComputeModule, iteration 0 (warm-up) and 1."""

import os
import sys

import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d

sys.path.insert(0, os.path.join(os.environ["TS_DIR"], "tests"))
sys.path.insert(0, os.path.join(os.environ["TS_DIR"], "tests/distributed"))
from test_all_reduce_compiled import AllReduceWithComputeModule  # noqa: E402

torch.spyre._impl._lazy_init()
dist.init_process_group("cpu:gloo,spyre:spyreccl")
c10d._register_process_group("default", dist.group.WORLD)
rank = dist.get_rank()
dev = torch.device(f"spyre:{rank}")

m = torch.compile(AllReduceWithComputeModule())
for it in range(2):
    x = torch.ones((128,), dtype=torch.float16, device=dev)
    y = torch.ones((128,), dtype=torch.float16, device=dev)
    torch.spyre.synchronize() if hasattr(torch.spyre, "synchronize") else None
    print(f"### ITER {it} BEGIN", flush=True)
    out = m(x, y).to("cpu")
    print(
        f"### ITER {it} END value={out[0].item()} x_after={x.to('cpu')[0].item()}",
        flush=True,
    )
dist.destroy_process_group()
