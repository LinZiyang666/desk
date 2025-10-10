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

def _apply_cpu_cap(percent: float, period_us: int = 100000, cgname: str = "prof_cap", verbose: bool = False):
    """
    将当前进程的 CPU 聚合使用率限制为全机总算力的 percent%。
    v2 方案：在“当前进程所在 cgroup 的父目录”下创建子 cgroup，启用父目录 +cpu，再对子级写 cpu.max，并把进程放入子级。
    v1 方案：创建 cpu controller 的 cgroup，写 cfs_quota/period，并把进程加入 tasks。
    """
    import errno
    percent = float(max(1.0, min(100.0, percent)))
    ncpu = os.cpu_count() or 1
    quota = int(period_us * (percent / 100.0) * ncpu)
    quota = max(quota, 1000)  # 避免过小导致调度抖动
    pid = os.getpid()
    cg_root = "/sys/fs/cgroup"

    # ---------- cgroups v2 ----------
    controllers_path = os.path.join(cg_root, "cgroup.controllers")
    if os.path.exists(controllers_path):
        # 解析当前进程的 cgroup 路径，如 "0::/user.slice/xxx.scope"
        v2_rel = "/"
        try:
            with open("/proc/self/cgroup", "r") as f:
                for line in f:
                    parts = line.strip().split(":")
                    if len(parts) == 3 and parts[0] == "0":
                        v2_rel = parts[2] or "/"
                        break
        except Exception:
            v2_rel = "/"

        parent_dir = os.path.normpath(os.path.join(cg_root, v2_rel))
        if not os.path.isdir(parent_dir):
            parent_dir = cg_root  # 兜底

        # 1) 确保父目录启用了 +cpu 控制器
        ctrls = ""
        try:
            with open(os.path.join(parent_dir, "cgroup.controllers"), "r") as f:
                ctrls = f.read().strip()
        except Exception:
            pass
        subtree_ctrl = os.path.join(parent_dir, "cgroup.subtree_control")
        if "cpu" in ctrls:
            try:
                with open(subtree_ctrl, "r+") as f:
                    cur = f.read()
                if "+cpu" not in cur.split():
                    with open(subtree_ctrl, "w") as f:
                        f.write("+cpu")
            except OSError as e:
                if e.errno not in (errno.EPERM, errno.EACCES):
                    raise
        else:
            # 父目录没有 cpu 控制器，可能无权限启用；继续尝试，若失败再报错
            try:
                with open(subtree_ctrl, "w") as f:
                    f.write("+cpu")
            except OSError as e:
                if e.errno in (errno.EPERM, errno.EACCES):
                    raise PermissionError(
                        f"无法在 {parent_dir} 启用 +cpu（需要管理员权限）。"
                    ) from e

        # 2) 在父目录下创建/使用子 cgroup
        cgdir = os.path.join(parent_dir, cgname)
        os.makedirs(cgdir, exist_ok=True)

        # 3) 写 cpu.max
        try:
            with open(os.path.join(cgdir, "cpu.max"), "w") as f:
                f.write(f"{quota} {period_us}")
        except OSError as e:
            if e.errno in (errno.EPERM, errno.EACCES):
                raise PermissionError(
                    f"写入 {cgdir}/cpu.max 被拒绝。很可能父目录未启用 +cpu。"
                    f"请执行： echo +cpu > {subtree_ctrl}"
                ) from e
            raise

        # 4) 把当前进程移入子 cgroup
        with open(os.path.join(cgdir, "cgroup.procs"), "w") as f:
            f.write(str(pid))

        if verbose:
            print(f"[CGv2] parent={parent_dir}, cpu.max={quota} {period_us}, percent={percent:.2f}%, cores={ncpu}")
        return

    # ---------- cgroups v1 回退 ----------
    cg_cpu = os.path.join(cg_root, "cpu")
    if os.path.isdir(cg_cpu):
        cgdir = os.path.join(cg_cpu, cgname)
        os.makedirs(cgdir, exist_ok=True)
        with open(os.path.join(cgdir, "cpu.cfs_period_us"), "w") as f:
            f.write(str(period_us))
        with open(os.path.join(cgdir, "cpu.cfs_quota_us"), "w") as f:
            f.write(str(quota))
        with open(os.path.join(cgdir, "tasks"), "w") as f:
            f.write(str(pid))
        if verbose:
            print(f"[CGv1] cpu.cfs_quota_us={quota} cpu.cfs_period_us={period_us}, percent={percent:.2f}%, cores={ncpu}")
        return

    raise RuntimeError("未检测到 cgroups v2 或 v1，无法应用 CPU cap。")

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



parser = argparse.ArgumentParser()
parser.add_argument(
    "--cpu-cap-range", type=float, nargs=2, metavar=("START%", "END%"),
    default=[30.0, 100.0],
    help="CPU max utilization sweep range in percent of TOTAL machine CPU, e.g. --cpu-cap-range 30 100"
)
parser.add_argument(
    "--cpu-cap-step", type=float, default=10.0,
    help="Sweep step (percent), e.g. --cpu-cap-step 10"
)
args = parser.parse_args()
def main():
    device = torch.device("cpu")

    name = "Qwen/Qwen2.5-Omni-3B"
    full_base = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(name, trust_remote_code=True)
    full_base = full_base.model
    L = 256  # 序列长度保持与你原脚本一致
    from decimal import Decimal, getcontext
    getcontext().prec = 6   # 足够处理一位小数

    def build_caps(start: float, end: float, step: float):
        if step <= 0:
            raise ValueError("cpu-cap-step must be > 0")
        if end < start:
            # 允许用户反写，自动交换
            start, end = end, start

        s = Decimal(str(start))
        e = Decimal(str(end))
        st = Decimal(str(step))

        # floor((end-start)/step) + 1 个点
        n = int(((e - s) / st).to_integral_value(rounding="ROUND_FLOOR")) + 1
        vals = [float(s + st * i) for i in range(n)]

        # 若因舍入遗漏端点，补上 end
        if vals[-1] < float(e) - 1e-12:
            vals.append(float(e))

        # 统一到一位小数，去重并排序（防浮点毛刺）
        vals = sorted({round(v, 1) for v in vals})
        return vals

    # 在 main() 或你需要的地方使用：
    start, end = args.cpu_cap_range
    step = args.cpu_cap_step
    caps = build_caps(start, end, step)
    # layers_list = list(range(1, 11))
    # B_list = list(range(1, 11))
    layers_list = list(range(1, 8))
    B_list = list(range(1, 8))


    total = len(caps) * len(layers_list) * len(B_list)
    pbar = tqdm(total=total, desc="Grid Running", ncols=100)

    try:
        for cap in caps:
            # 应用 CPU cap（cgroups 限流）
            _apply_cpu_cap(cap, verbose=False)

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
                    pbar.set_description(f"cap={cap:.1f}% layers={layers} B={B}")
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
                    target_dir = os.path.join("qwen2.5_omni", f"{cap:.1f}", str(layers), str(B))
                    _move_outputs_to_dir(target_dir, verbose=False)

                    # 明确显示当前完成项
                    tqdm.write(f"[DONE] cap={cap:.1f}% layers={layers} B={B} -> {target_dir}")

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