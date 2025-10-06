import json
from typing import Any, Dict, List, Union
import argparse, os, torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
import torch.distributed as dist

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import time

from stage_with_mutiple_ranks import PipelineStage_with_mutiple_ranks
from schedule_runtime import PipelineScheduleRuntimeWithDirection
from pipelining_source_code.schedules import _Action, _ComputationType

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
from simple_1F1B_Action import generate_1f1b_pipeline_actions, generate_1f1b_pipeline_actions_pro

class PartMiddle(nn.Module):
    """公共基类：仅负责若干 transformer layer。"""
    def __init__(self, model, start, end):
        super().__init__()
        self.layers = nn.ModuleList(model.model.layers[start:end])
        self.rotary_emb = model.model.rotary_emb

    def forward(self, hidden, attn_mask):
        bsz, seqlen = hidden.shape[:2]
        device = hidden.device
        position_ids = torch.arange(seqlen, device=device).unsqueeze(0).expand(bsz, -1)
        pos_emb = self.rotary_emb(hidden, position_ids)

        if attn_mask.dim() == 2:                       # 兼容单矩阵传递
            attn_mask = torch.triu(
                torch.full((seqlen, seqlen), float('-inf'), device=device), 1
            ).unsqueeze(0).unsqueeze(0).expand(bsz, 1, -1, -1).contiguous()
        elif not attn_mask.is_contiguous():
            attn_mask = attn_mask.contiguous()

        for layer in self.layers:
            hidden = layer(hidden_states=hidden,
                           attention_mask=attn_mask,
                           position_ids=position_ids,
                           position_embeddings=pos_emb,
                           output_attentions=False,
                           use_cache=False)[0]
        return hidden.contiguous(), attn_mask          

def synth_hidden(model, B, L, seed=1234, device="cpu", dtype=None, std=0.02):
    """造 [B, L, H] 的 hidden，确定性高斯分布。"""
    H = model.config.hidden_size
    if dtype is None:
        try:
            dtype = next(model.parameters()).dtype
        except StopIteration:
            dtype = torch.float32

    g = torch.Generator(device=device).manual_seed(seed)
    # 正态分布，方差小，数值稳定
    hidden = torch.randn((B, L, H), generator=g, device=device, dtype=dtype) * std
    return hidden.contiguous()

_mask_cache = {}

def synth_attn_mask(B, L, mode="causal", window=None, device="cpu"):
    """
    生成 attention_mask：
    - 'causal': 传 2D (B, L) 的占位，触发你 PartMiddle 里“内部构造因果掩码”的路径（统一且重现）。
    - 'sliding': 生成 4D (B, 1, L, L) 的带状 -inf 掩码，近似 SWA 的计算图复杂度。
    - 'dense':   生成 4D 全 0 掩码（无屏蔽，测算子上界）。
    """
    key = (B, L, mode, window)
    if key in _mask_cache:
        return _mask_cache[key]

    if mode == "causal":
        # 传 2D 占位（例如全 1），你的 PartMiddle 会内部构造三角 causal mask
        attn_mask = torch.ones((B, L), dtype=torch.bool, device=device)
    elif mode == "sliding":
        assert window is not None and window > 0, "sliding 模式需要 window"
        base = torch.full((L, L), float("-inf"), device=device)
        # 仅允许 [i-window+1, i] 之间注意
        for i in range(L):
            s = max(0, i - window + 1)
            base[i, s:i+1] = 0.0
        attn_mask = base.unsqueeze(0).unsqueeze(0).expand(B, 1, L, L).contiguous()
    elif mode == "dense":
        attn_mask = torch.zeros((B, 1, L, L), device=device)
    else:
        raise ValueError(f"Unknown attn mode: {mode}")

    _mask_cache[key] = attn_mask
    return attn_mask

def synth_grad_like(output, seed=4321, std=0.01):
    """给 backward 用的外部梯度，确定性随机，避免全 1 带来的模式化分支。"""
    dev = output.device
    g = torch.Generator(device="cpu" if dev.type == "cpu" else dev).manual_seed(seed)
    grad = torch.randn(output.shape, generator=g, device=dev, dtype=output.dtype) * std
    return grad.contiguous()


# parser = argparse.ArgumentParser()
# parser.add_argument("--layers", type=int, default=1,
#                     help="number of transformer layers")
# parser.add_argument("--batch_size", type=int,
#                     default=int(os.getenv("BATCH_SIZE", 20)),
#                     help="The batch size of each rank (the environment variable BATCH_SIZE can be overridden)")
# parser.add_argument("--microbatch_num", type=int,
#                     default=int(os.getenv("MICROBATCH_NUM", 5)),
#                     help="Micro-batch number (the environment variable MICROBATCH_NUM can be overridden)")
# parser.add_argument("--sudo_pass", default=os.getenv("SUDO_PASS"),
#                     help='Write the password of root')
# parser.add_argument("--upstream", default=os.getenv("upstream"),
#                     help='Write the upstream in mbps')
# parser.add_argument("--plan_loc", type=str, required=True,
#                     help='the json file that stores the sharding plans...')
# args = parser.parse_args()


def main():
    device = torch.device("cpu")     

    name = "Qwen/Qwen3-0.6B"
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    full = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True)

    layers, B, L =8, 8, 256
    
    stage_mod = PartMiddle(full, 0, layers)
    stage_mod.to(device)
    
    hidden = synth_hidden(full, B, L, seed=42, device="cpu")   # H 自动从 config 取
    attn_mask = synth_attn_mask(B, L, mode="causal", device="cpu")

    del full                        
    import gc; gc.collect()

    stage_mod.train(False)   # 关闭 dropout 等随机性（Qwen3 通常无 dropout，但稳妥）
    out, attn_mask_used = stage_mod(hidden.requires_grad_(True), attn_mask)
    print("forward算完了")
    
    # 反传（若你需要计入 backward 的算力）
    grad_out = synth_grad_like(out, seed=43)
    out.backward(grad_out)
    print("backward算完了")
    
    




if __name__ == "__main__":
    main()