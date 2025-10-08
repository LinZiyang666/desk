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

torch.set_num_threads(16)

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

import shutil, glob

# # ===== 工具函数：设置所有 policy 的目标频率（kHz），尽量夹紧到硬件范围 =====
# def _set_all_policies_fixed_freq_khz(target_khz: int, verbose: bool = False):
#     base = "/sys/devices/system/cpu/cpufreq"
#     pols = sorted(glob.glob(os.path.join(base, "policy*")))
#     if not pols:
#         if verbose:
#             print("[WARN] 未发现 cpufreq policy 目录，可能系统未暴露调频接口，跳过设频。")
#         return
#     for p in pols:
#         name = os.path.basename(p)
#         try:
#             # 读取硬件上下限，做夹紧
#             hwmin = int(open(os.path.join(p, "cpuinfo_min_freq")).read().strip())
#             hwmax = int(open(os.path.join(p, "cpuinfo_max_freq")).read().strip())
#             t = max(hwmin, min(hwmax, int(target_khz)))
#             gov_path = os.path.join(p, "scaling_governor")
#             av_gov = open(os.path.join(p, "scaling_available_governors")).read().strip().split()
#             # 尝试用 performance 或 userspace（有些平台固定频点需要 userspace）
#             prefer = "userspace" if "userspace" in av_gov else ("performance" if "performance" in av_gov else None)
#             if prefer is not None:
#                 with open(gov_path, "w") as f: f.write(prefer)
#             # 写 min/max 为同一频点（即“尽量固定”）
#             with open(os.path.join(p, "scaling_min_freq"), "w") as f: f.write(str(t))
#             with open(os.path.join(p, "scaling_max_freq"), "w") as f: f.write(str(t))
#             if verbose:
#                 print(f"[{name}] governor={prefer} set {t/1_000_000:.3f} GHz")
#         except Exception as e:
#             if verbose:
#                 print(f"[WARN] 设置 {name} 失败：{e}（继续其它 policy）")

# ===== 新增：精确限流（cgroups CPU quota，v2 首选，v1 回退）=====
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
                    # v2 行形如：0::/user.slice/user-0.slice/session-1.scope
                    parts = line.strip().split(":")
                    if len(parts) == 3 and parts[0] == "0":
                        v2_rel = parts[2] or "/"
                        break
        except Exception:
            v2_rel = "/"

        parent_dir = os.path.normpath(os.path.join(cg_root, v2_rel))
        # 确保父目录存在
        if not os.path.isdir(parent_dir):
            parent_dir = cg_root  # 兜底

        # 1) 确保父目录启用了 +cpu 控制器
        ctrls = ""
        try:
            with open(os.path.join(parent_dir, "cgroup.controllers"), "r") as f:
                ctrls = f.read().strip()
        except Exception:
            pass
        if "cpu" not in ctrls.split():
            # 父目录没有 cpu 控制器可用（极少见），降级到根去尝试
            parent_dir = cg_root
            try:
                with open(os.path.join(parent_dir, "cgroup.controllers"), "r") as f:
                    ctrls = f.read().strip()
            except Exception:
                ctrls = ""

        # 尝试在 parent_dir 开启 +cpu
        subtree_ctrl = os.path.join(parent_dir, "cgroup.subtree_control")
        try:
            cur = ""
            try:
                with open(subtree_ctrl, "r") as f:
                    cur = f.read().strip()
            except Exception:
                cur = ""
            if "+cpu" not in cur and "cpu" not in cur.split():
                with open(subtree_ctrl, "w") as f:
                    f.write("+cpu")
        except OSError as e:
            if e.errno in (errno.EPERM, errno.EACCES):
                raise PermissionError(
                    f"无法在 {parent_dir} 启用 +cpu（需要在父 cgroup 打开 controller）。"
                    f"请以 root 执行： echo +cpu > {subtree_ctrl}"
                ) from e
            # 其它错误：继续尝试，可能已经启用

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
            print(f"[CGv1] quota={quota}, period={period_us}, percent={percent:.2f}%, cores={ncpu}")
        return

    raise RuntimeError("未检测到可用的 cgroups CPU 控制器（既无 v2 也无 v1）。")

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



# ----------------- argparse 新增 -----------------
parser = argparse.ArgumentParser()
# parser.add_argument(
#     "--freq-range", type=float, nargs=2, metavar=("START", "END"),
#     default=[1.2, 3.2],
#     help="CPU frequency sweep range in GHz, e.g. --freq-range 1.2 3.2"
# )
# parser.add_argument(
#     "--freq-step", type=float, default=0.2,
#     help="CPU frequency sweep step in GHz, e.g. --freq-step 0.2"
# )
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

    name = "Qwen/Qwen3-1.7B"
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    full_base = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True)

    L = 256 
    from decimal import Decimal, getcontext
    getcontext().prec = 8   # 足够处理百分比步进

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
    cap_start, cap_end = args.cpu_cap_range
    cap_step = args.cpu_cap_step
    caps = build_caps(cap_start, cap_end, cap_step)
    layers_list = list(range(1, 11))
    B_list = list(range(1, 11))

    total = len(caps) * len(layers_list) * len(B_list)
    pbar = tqdm(total=total, desc="Grid Running", ncols=100)

    try:
        for cap in caps:
            # 设 CPU 频率（kHz）
            #_set_all_policies_fixed_freq_khz(int(f_ghz * 1_000_000), verbose=False)
            _apply_cpu_cap(cap, period_us=100000, cgname="prof_cap", verbose=False)

            for layers in layers_list:
                # 为当前 layers 重新构建 stage 与输入
                # 复制一个轻量的“中间层切片”模型
                # 注意：这里基于 full_base 的结构创建 PartMiddle，不重复下载权重
                full = full_base  # 仅为可读性，实际仍用同一对象
                stage_mod = PartMiddle(full, 0, layers).to(device)
                hidden_cache = {}  # 缓存不同 B 的合成输入，减少重复构造
                attn_cache = {}

                for B in B_list:
                    # 更新进度条的任务题头
                    #pbar.set_description(f"freq={f_ghz:.1f}GHz layers={layers} B={B}")
                    pbar.set_description(f"cap={cap:.1f}% layers={layers} B={B}")
                    # 合成输入
                    if B not in hidden_cache:
                        hidden_cache[B] = synth_hidden(full, B, L, seed=42, device=str(device))
                    if B not in attn_cache:
                        attn_cache[B] = synth_attn_mask(B, L, mode="causal", device=str(device))
                    hidden = hidden_cache[B]
                    attn_mask = attn_cache[B]

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
                            out, attn_mask_used = stage_mod(hidden.clone().requires_grad_(True), attn_mask)
                        outs[mb] = out
                        aid += 1

                    # Ordinary 6 轮
                    for k in range(6):
                        f_mb = 5 + k
                        b_mb = k
                        with rec.record(batch_id=0, action_id=aid, action="FORWARD", stage_idx=0, mb_idx=f_mb):
                            out, attn_mask_used = stage_mod(hidden.clone().requires_grad_(True), attn_mask)
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

                    # 移动到 ./qwen3_1.7B/capXX/{layers}/{mb_size}/
                    target_dir = os.path.join("qwen3_1.7B", f"cap{int(round(cap)):02d}", str(layers), str(B))
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