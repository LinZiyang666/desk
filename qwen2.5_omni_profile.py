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

from transformers.models.qwen2_5_omni import Qwen2_5OmniThinkerForConditionalGeneration
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
from simple_1F1B_Action import generate_1f1b_pipeline_actions, generate_1f1b_pipeline_actions_pro

torch.set_num_threads(16)

class PartMiddle(nn.Module):
    def __init__(self, text_model, L1, L2):
        super().__init__()
        self.layers = nn.ModuleList(text_model.layers[L1:L2])
        self.rotary_emb = text_model.rotary_emb
        
    def forward(self, *args, **kwargs):
        # Handle flexible arguments
        if len(args) >= 3:
            hidden, attn_mask, position_ids = args[:3]
        else:
            hidden = args[0] if len(args) > 0 else kwargs['hidden']
            attn_mask = args[1] if len(args) > 1 else kwargs['attn_mask']
            position_ids = args[2] if len(args) > 2 else kwargs['position_ids']
        
        if position_ids.dim() == 2:
            position_ids = position_ids.unsqueeze(0).repeat(3, 1, 1)
        
        pos_emb = self.rotary_emb(hidden, position_ids)
        for blk in self.layers:
            hidden = blk(
                hidden_states=hidden,
                attention_mask=attn_mask,
                position_ids=position_ids,
                position_embeddings=pos_emb,
                output_attentions=False,
                use_cache=False
            )[0]
        return hidden.contiguous(), attn_mask.contiguous(), position_ids.contiguous()  


# ---------------------------
# 1) 造 hidden: [B, L, H]
# ---------------------------
def synth_hidden(text_model, B, L, *, seed=1234, std=0.02, device="cpu", dtype=None):
    """
    生成确定性高斯 hidden，数值稳定（小方差），形状 [B, L, H]。
    H 自动从 text_model.config.hidden_size 读取。
    """
    H = getattr(getattr(text_model, "config", text_model), "hidden_size")
    if dtype is None:
        try:
            dtype = next(text_model.parameters()).dtype
        except StopIteration:
            dtype = torch.float32

    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((B, L, H), generator=g, device=device, dtype=dtype) * std
    return x.contiguous()


# ---------------------------
# 2) 造 attention mask: 4D
#    支持 'causal' / 'sliding' / 'dense'
# ---------------------------
_mask_cache = {}

def synth_attn_mask(B, L, *, mode="causal", window=None, device="cpu", dtype=torch.float32):
    """
    产出 4D inverted 掩码 (B, 1, L, L)，与 Omni 文本层期望一致：
      - 允许位置处为 0
      - 禁止位置处为 极小负值 (≈ -inf)
    若 mode='sliding'，需提供 window（窗口大小）。
    """
    key = (B, L, mode, window, device, dtype)
    if key in _mask_cache:
        return _mask_cache[key]

    min_val = torch.finfo(dtype).min
    if mode == "dense":
        # 全允许：全 0
        mask = torch.zeros((B, 1, L, L), device=device, dtype=dtype)

    elif mode == "causal":
        # 上三角禁止，含自身以下允许
        base = torch.full((L, L), fill_value=min_val, device=device, dtype=dtype)
        base = torch.triu(base, diagonal=1)  # 上三角为 min_val，其他为 0
        mask = base.unsqueeze(0).unsqueeze(0).expand(B, 1, L, L).contiguous()

    elif mode == "sliding":
        assert isinstance(window, int) and window > 0, "sliding 模式需要正整数 window"
        base = torch.full((L, L), fill_value=min_val, device=device, dtype=dtype)
        # 仅允许 [i-window+1, i] 范围
        for i in range(L):
            s = max(0, i - window + 1)
            base[i, s:i+1] = 0.0
        mask = base.unsqueeze(0).unsqueeze(0).expand(B, 1, L, L).contiguous()

    else:
        raise ValueError(f"Unknown attn mode: {mode}")

    _mask_cache[key] = mask
    return mask


# ---------------------------
# 2.5)（可选）造 position_ids: [3, B, L]
# Omni 文本使用 3 维 mRoPE；纯文本时三个分量可同一 0..L-1
# ---------------------------
def synth_position_ids(B, L, *, device="cpu", base=0):
    """
    返回 [3, B, L] 的整型位置索引（纯文本：三路一致）。
    你可按需求把三路改成 (t, h, w) 的不同步进。
    """
    pos = torch.arange(base, base + L, device=device, dtype=torch.long)
    pos3 = pos.view(1, 1, L).expand(3, B, L).contiguous()
    return pos3


# ---------------------------
# 3) 造外部梯度：与输出同形
# ---------------------------
def synth_grad_like(output, *, seed=4321, std=0.01):
    """
    给 backward 用的外部梯度，确定性随机；避免使用 randn_like(generator=...)
    以兼容较老的 PyTorch。
    """
    g = torch.Generator(device=output.device).manual_seed(seed)
    grad = torch.randn(output.shape, generator=g, device=output.device, dtype=output.dtype) * std
    return grad.contiguous()


import shutil, glob

# ===== 工具函数：设置所有 policy 的目标频率（kHz），尽量夹紧到硬件范围 =====
def _set_all_policies_fixed_freq_khz(target_khz: int, verbose: bool = False):
    base = "/sys/devices/system/cpu/cpufreq"
    pols = sorted(glob.glob(os.path.join(base, "policy*")))
    if not pols:
        if verbose:
            print("[WARN] 未发现 cpufreq policy 目录，可能系统未暴露调频接口，跳过设频。")
        return
    for p in pols:
        name = os.path.basename(p)
        try:
            # 读取硬件上下限，做夹紧
            hwmin = int(open(os.path.join(p, "cpuinfo_min_freq")).read().strip())
            hwmax = int(open(os.path.join(p, "cpuinfo_max_freq")).read().strip())
            t = max(hwmin, min(hwmax, int(target_khz)))
            gov_path = os.path.join(p, "scaling_governor")
            av_gov = open(os.path.join(p, "scaling_available_governors")).read().strip().split()
            # 尝试用 performance 或 userspace（有些平台固定频点需要 userspace）
            prefer = "userspace" if "userspace" in av_gov else ("performance" if "performance" in av_gov else None)
            if prefer is not None:
                with open(gov_path, "w") as f: f.write(prefer)
            # 写 min/max 为同一频点（即“尽量固定”）
            with open(os.path.join(p, "scaling_min_freq"), "w") as f: f.write(str(t))
            with open(os.path.join(p, "scaling_max_freq"), "w") as f: f.write(str(t))
            if verbose:
                print(f"[{name}] governor={prefer} set {t/1_000_000:.3f} GHz")
        except Exception as e:
            if verbose:
                print(f"[WARN] 设置 {name} 失败：{e}（继续其它 policy）")

# ===== 工具函数：把本地生成的两个文件移动到目标目录 =====
def _move_outputs_to_dir(target_dir: str, verbose: bool = False):
    os.makedirs(target_dir, exist_ok=True)
    moved_any = False
    for fname in ["timeline_rank0.json", "cpu_mem_rank0.jsonl"]:
        if os.path.exists(fname):
            shutil.move(fname, os.path.join(target_dir, fname))
            moved_any = True
            if verbose:
                print(f"[MOVE] {fname} -> {target_dir}/")
    if not moved_any and verbose:
        print(f"[WARN] 未找到需要移动的输出文件（可能本轮未产生或命名不同）。")

def main():
    device = torch.device("cpu")

    name = "Qwen/Qwen2.5-Omni-3B"
    full_base = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(name, trust_remote_code=True)
    full_base = full_base.model
    L = 256  # 序列长度保持与你原脚本一致
    # 三重循环参数
    freqs = [round(1.2 + 0.2*i, 1) for i in range(int((3.2 - 1.2) / 0.2) + 1)] + [3.2]
    freqs = sorted(set(freqs))  # 防止浮点累加误差
    layers_list = list(range(1, 11))
    B_list = list(range(1, 11))

    total = len(freqs) * len(layers_list) * len(B_list)
    pbar = tqdm(total=total, desc="Grid Running", ncols=100)

    try:
        for f_ghz in freqs:
            # 设 CPU 频率（kHz）
            _set_all_policies_fixed_freq_khz(int(f_ghz * 1_000_000), verbose=False)

            for layers in layers_list:
                # 为当前 layers 重新构建 stage 与输入
                # 复制一个轻量的“中间层切片”模型
                # 注意：这里基于 full_base 的结构创建 PartMiddle，不重复下载权重
                full = full_base  # 仅为可读性，实际仍用同一对象
                stage_mod = PartMiddle(full, 0, layers).to(device)
                hidden_cache = {}   # 缓存不同 B 的合成输入，减少重复构造
                attn_cache = {}
                posid_cache = {}

                for B in B_list:
                    # 更新进度条的任务题头
                    pbar.set_description(f"freq={f_ghz:.1f}GHz layers={layers} B={B}")
                    # 合成输入
                    if B not in hidden_cache:
                        hidden_cache[B] = synth_hidden(full, B, L, seed=42, device=str(device))
                    if B not in attn_cache:
                        attn_cache[B] = synth_attn_mask(B, L, mode="causal", device=str(device))
                    if B not in posid_cache:
                        posid_cache[B] = synth_position_ids(B, L, device=str(device))
                    hidden = hidden_cache[B]
                    attn_mask = attn_cache[B]
                    position_ids = posid_cache[B]

                    # 回收器
                    from recorder import Recorder
                    rec = Recorder(0, mark_actions=False)

                    # 1F1B: 5 warmup F，6 轮 (F+B)，再 5 个冷却 B —— 与你之前脚本一致
                    stage_mod.train(False)

                    outs = {}
                    aid = 0

                    # Warmup F (5)
                    for mb in range(5):
                        with rec.record(batch_id=0, action_id=aid, action="FORWARD", stage_idx=0, mb_idx=mb):
                            out, _, _ = stage_mod(hidden.clone().requires_grad_(True),
                                                  attn_mask,
                                                  position_ids)
                        outs[mb] = out
                        aid += 1

                    # Ordinary 6 轮
                    for k in range(6):
                        f_mb = 5 + k
                        b_mb = k
                        with rec.record(batch_id=0, action_id=aid, action="FORWARD", stage_idx=0, mb_idx=f_mb):
                            out, _, _ = stage_mod(hidden.clone().requires_grad_(True),
                                                  attn_mask,
                                                  position_ids)
                        outs[f_mb] = out
                        aid += 1

                        grad_out = synth_grad_like(outs[b_mb], seed=43 + b_mb)
                        with rec.record(batch_id=0, action_id=aid, action="FULL_BACKWARD", stage_idx=0, mb_idx=b_mb):
                            outs[b_mb].backward(grad_out)
                        del outs[b_mb]
                        aid += 1

                    # Cooldown B (5)
                    for b_mb in range(6, 11):
                        grad_out = synth_grad_like(outs[b_mb], seed=43 + b_mb)
                        with rec.record(batch_id=0, action_id=aid, action="FULL_BACKWARD", stage_idx=0, mb_idx=b_mb):
                            outs[b_mb].backward(grad_out)
                        del outs[b_mb]
                        aid += 1

                    # dump 到当前目录
                    rec.dump()

                    # 移动到 ./qwen3_0.6/{cpu_frequence}/{layers}/{mb_size}/
                    target_dir = os.path.join("qwen3_1.7B", f"{f_ghz:.1f}", str(layers), str(B))
                    _move_outputs_to_dir(target_dir, verbose=False)

                    # 明确显示当前完成项
                    tqdm.write(f"[DONE] freq={f_ghz:.1f}GHz layers={layers} B={B} -> {target_dir}")

                    # 更新进度
                    pbar.update(1)

                    # 释放显存/内存（CPU 上主要是 Python 对象）
                    del rec, outs
                    import gc; gc.collect()

                # 为不同 layers 释放 stage_mod
                del stage_mod
                import gc; gc.collect()

    finally:
        pbar.close()

    
    




if __name__ == "__main__":
    main()