# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import torch

def get_machine_local_and_dist_rank():
    """
    Get the distributed and local rank of the current gpu.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        distributed_rank = torch.distributed.get_rank()
    else:
        try:
            distributed_rank = int(os.environ.get("RANK", "0"))
        except (TypeError, ValueError):
            distributed_rank = 0

    try:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    except (TypeError, ValueError):
        local_rank = 0
    return local_rank, distributed_rank
