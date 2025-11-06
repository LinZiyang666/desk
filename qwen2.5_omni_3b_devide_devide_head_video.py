import argparse, os, torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
import torchaudio
import torchaudio.functional as AF
import torch.distributed as dist

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

import time, math

from stage_with_mutiple_ranks import PipelineStage_with_mutiple_ranks, PipelineStage_Multimodality
from schedule_runtime import PipelineScheduleRuntimeWithDirection
from pipelining_source_code.schedules import _Action, _ComputationType

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
from simple_1F1B_Action import generate_1f1b_pipeline_actions
from transformers.models.qwen2_5_omni import Qwen2_5OmniThinkerForConditionalGeneration

# ==== 加载 Thinker（Omni-3B，只要文本主干 + 多模态编码器） ====
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen2_5_omni import Qwen2_5OmniThinkerForConditionalGeneration
from transformers import AutoProcessor
from typing import Dict, Any, Optional, Tuple, List
import torch, torch.nn as nn, torch.nn.functional as F

# Helper to load local ./video.mp4 once and convert to processor-ready video pack
def _load_video_pack(proc, path, vision_module, max_tubelets=16):
    print(f"[video_loader] cwd={os.getcwd()} trying processor on path={path}")
    # Prefer the dedicated video processor if available; otherwise fall back to the generic AutoProcessor.
    video_proc = getattr(proc, "video_processor", None) or getattr(proc, "vision_processor", None)
    use_auto = video_proc is None
    if use_auto:
        video_proc = proc

    def _run_processor(video_input):
        kwargs = {"videos": [video_input], "return_tensors": "pt"}
        if use_auto:
            # AutoProcessor main API expects a companion text input; supply empty prompt.
            kwargs.setdefault("text", [""])
        return video_proc(**kwargs)

    try:
        batch = _run_processor(path)
        pv = batch.get("pixel_values_videos", None)
        vthw = batch.get("video_grid_thw", None)
        print(f"[video_loader] processor returned pv={None if pv is None else tuple(pv.shape)} grid={None if vthw is None else tuple(vthw.shape)}")
        if pv is not None and vthw is not None and pv.shape[0] >= 1:
            # Optionally clip tubelets to reduce memory
            if max_tubelets is not None and pv.shape[1] > max_tubelets:
                batch["pixel_values_videos"] = pv[:, :max_tubelets]
                batch["video_grid_thw"] = vthw.clone()
                batch["video_grid_thw"][:, 0] = max_tubelets
            return {
                "pixel_values_videos": batch["pixel_values_videos"],
                "video_grid_thw": batch["video_grid_thw"],
            }
    except Exception as e:
        print(f"[video_loader] processor path load failed with {type(e).__name__}: {e}")
    try:
        # Fallback with torchvision if direct path processing is unsupported
        from torchvision.io import read_video
        print("[video_loader] trying torchvision.read_video fallback")
        vframes, _, _ = read_video(path, pts_unit="sec")  # [T,H,W,3] uint8
        print(f"[video_loader] read_video frames shape={tuple(vframes.shape)}")
        frame_list = [f.cpu().numpy() for f in vframes]   # list of HWC uint8
        batch = _run_processor(frame_list)
        pv = batch.get("pixel_values_videos", None)
        vthw = batch.get("video_grid_thw", None)
        print(f"[video_loader] fallback processor returned pv={None if pv is None else tuple(pv.shape)} grid={None if vthw is None else tuple(vthw.shape)}")
        if pv is not None and vthw is not None:
            if max_tubelets is not None and pv.shape[1] > max_tubelets:
                batch["pixel_values_videos"] = pv[:, :max_tubelets]
                batch["video_grid_thw"] = vthw.clone()
                batch["video_grid_thw"][:, 0] = max_tubelets
            return {
                "pixel_values_videos": batch["pixel_values_videos"],
                "video_grid_thw": batch["video_grid_thw"],
            }
    except Exception as e2:
        print(f"[video_loader] torchvision fallback failed with {type(e2).__name__}: {e2}")
    print("[video_loader] returning empty video pack")
    return {"pixel_values_videos": None, "video_grid_thw": None}


def _replicate_video_pack_for_microbatches(video_pack: Optional[Dict[str, torch.Tensor]], num_microbatches: int) -> Optional[Dict[str, torch.Tensor]]:
    if video_pack is None or num_microbatches <= 1:
        return video_pack
    replicated: Dict[str, torch.Tensor] = {}
    for key, value in video_pack.items():
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                replicated[key] = value
                continue
            reps = [1] * (value.dim() + 1)
            reps[0] = num_microbatches
            replicated[key] = value.unsqueeze(0).repeat(*reps).contiguous()
        else:
            replicated[key] = value
    return replicated


def build_causal(mask_len, device):
    return torch.triu(torch.full((mask_len, mask_len), float("-inf"), device=device), diagonal=1)\
                .unsqueeze(0).unsqueeze(0)  # [1,1,T,T]

def pack_modalities(text_embeds, audio_seq=None, vision_seq=None):

    def _ensure_3d(x):
        if x is None:
            return None
        return x if x.dim() == 3 else x.unsqueeze(0)  # [T,D] -> [1,T,D]
    seqs = [_ensure_3d(x) for x in [text_embeds, audio_seq, vision_seq] if x is not None]
    return torch.cat(seqs, dim=1) if len(seqs) > 1 else seqs[0]


DEFAULT_QWEN_OMNI_SYS = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)
USE_TTS_SYS = True


class AudioFrontAndTwoLayers(nn.Module):
    """
    Stage-A:
      - 输入: audio_inputs = {"input_features": [B, n_mels, T], "feature_attention_mask": [B, T] (bool/long)}
      - 逻辑: 复用官方分块与pad → conv1/conv2+GELU → 加正弦位置 → 选择有效帧 → 过前2层 encoder layer
      - 输出: (hidden_states_A: [sum_aftercnn, D], cu_seqlens: [B+1], aftercnn_lens: [B])
    """
    def __init__(self, enc: nn.Module, use_checkpoint: bool = False):
        super().__init__()
        # 引用原 AudioEncoder 的子模块与属性
        self.enc = enc  # Qwen2_5OmniAudioEncoder
        self.conv1 = enc.conv1
        self.conv2 = enc.conv2
        self.positional_embedding = enc.positional_embedding  # SinusoidsPositionEmbedding with .positional_embedding buffer
        self.n_window = enc.n_window

        # 前两层
        self.layer0 = enc.layers[0]
        self.layer1 = enc.layers[1]

        # 运行选项
        self.use_checkpoint = use_checkpoint

    @staticmethod
    def _as_bool(x: torch.Tensor) -> torch.Tensor:
        return x if x.dtype == torch.bool else x.to(dtype=torch.bool)

    def _maybe_ckpt(self, layer: nn.Module, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            # 仅将 layer 输出的 hidden_states 作为有效输出（与官方层 forward(outputs[0]) 对齐）
            return torch.utils.checkpoint.checkpoint(lambda hs, cs: layer(hs, cs)[0], hidden_states, cu_seqlens, use_reentrant=False)
        return layer(hidden_states, cu_seqlens)[0]

    def forward(self, audio_inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if "input_features" not in audio_inputs or "feature_attention_mask" not in audio_inputs:
            raise KeyError("audio_inputs must contain 'input_features'([B,n_mels,T]) and 'feature_attention_mask'([B,T]).")

        # === 0) 取输入 + 参考 dtype/device ===
        x: torch.Tensor = audio_inputs["input_features"]           # [B, n_mels, T]
        attn_mask_2d: torch.Tensor = audio_inputs["feature_attention_mask"]  # [B, T]
        conv1_w = self.conv1.weight
        ref_dtype, ref_device = conv1_w.dtype, conv1_w.device

        if x.dim() != 3:
            raise AssertionError(f"input_features must be 3D [B,n_mels,T], got {tuple(x.shape)}")
        if attn_mask_2d.dim() != 2:
            raise AssertionError(f"feature_attention_mask must be 2D [B,T], got {tuple(attn_mask_2d.shape)}")

        B, n_mels, T = x.shape
        attn_mask_2d = self._as_bool(attn_mask_2d)

        # === 1) 计算长度: feature_lens / aftercnn_lens ===
        feature_lens = attn_mask_2d.sum(dim=1).to(device=ref_device, dtype=torch.long)  # [B]
        # 根据官方实现接口获取卷积后的长度（与后续 split/还原保持一致）
        out = self.enc._get_feat_extract_output_lengths(feature_lens)
        aftercnn_lens = (out[0] if isinstance(out, (tuple, list)) else out).to(device=ref_device, dtype=torch.long)  # [B]

        # === 2) [B,n_mels,T] → [n_mels, sumT]（与官方 forward 一致的拼接方式）===
        # 先转 [B,T,n_mels]，再用 mask 选择有效帧 -> [sumT, n_mels]，最后转 [n_mels, sumT]
        val = x.permute(0, 2, 1)  # [B, T, n_mels]
        flat = val[attn_mask_2d.to(device=val.device)]  # [sumT, n_mels]
        feats_cat = flat.permute(1, 0).contiguous().to(dtype=ref_dtype, device=ref_device)  # [n_mels, sumT]

        # === 3) 依据 n_window 做分块长度（与官方 forward 一致）===
        # 每条样本切成 ceil(L / (2*n_window)) 个chunk；tail修正；0替换为整窗
        chunk_num = torch.ceil(feature_lens / (self.n_window * 2)).long()  # [B]
        chunk_lengths = torch.tensor(
            [self.n_window * 2] * int(chunk_num.sum().item()),
            dtype=torch.long,
            device=feature_lens.device,
        )
        # 尾块长度 = L % (2*n_window)（若整除则仍用整窗）
        tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
        chunk_lengths[tail_chunk_index] = feature_lens % (self.n_window * 2)
        chunk_lengths = torch.where(chunk_lengths == 0, torch.tensor(self.n_window * 2, device=chunk_lengths.device), chunk_lengths)

        # === 4) 调用官方的 padded_and_mask_function 做pad与mask（保持完全一致的语义）===
        chunk_list = feats_cat.split(chunk_lengths.tolist(), dim=1)  # list of [n_mels, chunk_len]
        padded_feature, padded_mask, padded_mask_after_cnn = self.enc.padded_and_mask_function(
            chunk_list, chunk_lengths, padding_value=0, padding_side="right"
        )
        # shapes:
        #   padded_feature: [num_chunks, n_mels, Lmax]
        #   padded_mask:    [num_chunks, 1, Lmax] (long)
        #   padded_mask_after_cnn: [num_chunks, Lmax_cnn] (bool)

        # === 5) CNN 前端 + GELU，转 [B', T', D]，加正弦位置嵌入 ===
        padded_feature = padded_feature.to(dtype=ref_dtype, device=ref_device)
        padded_mask = padded_mask.to(device=ref_device)
        padded_embed = F.gelu(self.conv1(padded_feature)) * padded_mask  # [B', D, L']
        padded_embed = F.gelu(self.conv2(padded_embed)).transpose(1, 2).contiguous()  # [B', T', D]

        # 位置嵌入（与官方相同：直接切 buffer 并 broadcast 到 batch 维）
        pos = self.positional_embedding.positional_embedding[: padded_embed.shape[1], :].unsqueeze(0)
        pos = pos.to(dtype=padded_embed.dtype, device=padded_embed.device)
        padded_embed = padded_embed + pos  # [B', T', D]

        # 依据 after-cnn 的 mask 选出有效时间步，拼成变长扁平序列
        padded_mask_after_cnn = padded_mask_after_cnn.to(device=padded_embed.device)
        hidden_states = padded_embed[padded_mask_after_cnn]  # [sum_aftercnn, D]（严格与官方保持一致语义）

        # cu_seqlens: 变长序列的前缀和，用于 varlen attention 的块对角遮罩
        cu_seqlens = torch.cat(
            (torch.zeros(1, device=padded_mask_after_cnn.device, dtype=torch.int32),
             padded_mask_after_cnn.sum(1).cumsum(0).to(torch.int32))
        )  # [B'+1]，这里 B' 等价于“所有 chunk 聚合后的样本维”，与 aftercnn_lens 总和一致

        # === 6) 过前2层 Transformer encoder layer ===
        hidden_states = self._maybe_ckpt(self.layer0, hidden_states, cu_seqlens)
        hidden_states = self._maybe_ckpt(self.layer1, hidden_states, cu_seqlens)

        return hidden_states.contiguous(), cu_seqlens.contiguous(), aftercnn_lens.contiguous()


class AudioEncoderMidFive(nn.Module):
    """
    Stage-B:
      - 输入: (hidden_states: [sum_aftercnn, D], cu_seqlens: [B+1], aftercnn_lens: [B])
      - 逻辑: 过中间5层 encoder layer（索引 2..6）
      - 输出: (hidden_states, cu_seqlens, aftercnn_lens)  # 后二者透传
    """
    def __init__(self, enc: nn.Module, use_checkpoint: bool = False):
        super().__init__()
        self.layers = nn.ModuleList([enc.layers[i] for i in range(2, 7)])  # 2,3,4,5,6 共5层
        self.use_checkpoint = use_checkpoint

    def _maybe_ckpt(self, layer: nn.Module, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(lambda hs, cs: layer(hs, cs)[0], hidden_states, cu_seqlens, use_reentrant=False)
        return layer(hidden_states, cu_seqlens)[0]

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor, aftercnn_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            hidden_states = self._maybe_ckpt(layer, hidden_states, cu_seqlens)
        return hidden_states.contiguous(), cu_seqlens.contiguous(), aftercnn_lens.contiguous()


def _compute_vision_pad_lengths(enc: nn.Module) -> tuple[int, int, int]:
    cfg = getattr(enc, "config", None)
    window_size = getattr(cfg, "window_size", getattr(enc, "window_size", 112))
    spatial_merge_size = getattr(cfg, "spatial_merge_size", getattr(enc, "spatial_merge_size", 1))
    temporal_hint = getattr(cfg, "max_temporal_length", None)
    if temporal_hint is None:
        temporal_hint = getattr(cfg, "temporal_patch_size", getattr(enc, "temporal_patch_size", 1))
    try:
        temporal_hint = max(1, int(temporal_hint))
    except Exception:
        temporal_hint = 1
    try:
        grid_side = max(1, int(window_size) // int(spatial_merge_size))
    except Exception:
        grid_side = 56
    pad_seq_len = temporal_hint * grid_side * grid_side
    pad_seq_len = max(pad_seq_len, 4096)
    spatial_merge_unit = getattr(enc, "spatial_merge_unit", 1)
    pad_window_len = max(4, (pad_seq_len // max(1, spatial_merge_unit)) + 4)
    pad_merged_len = max(1, pad_seq_len // max(1, spatial_merge_unit))
    return pad_seq_len, pad_window_len, pad_merged_len

class VisionFrontAndTwoLayers(nn.Module):
    """
    Stage-A (Vision):
      - 输入: vision_inputs = {
            "pixel_values": FloatTensor,   # 原始视觉输入（图像/视频堆栈），形状与官方一致
            "grid_thw":   LongTensor       # [num_imgs_or_vids, 3]，LLM 尺度下 (T,H,W)
        }
      - 逻辑: PatchEmbed → 计算 RoPE → 计算并应用窗口重排（只在本段做一次）→ 过前2层 VisionBlock
      - 输出: (hidden_states_A, cu_window_seqlens, grid_thw)  # 后两项供 Stage-B 复用/透传
    """
    def __init__(self, enc: nn.Module, use_checkpoint: bool = False):
        super().__init__()
        # 引用原 VisionEncoder 的关键成员（权重共享）
        self.enc = enc  # Qwen2_5OmniVisionEncoder
        self.patch_embed = enc.patch_embed
        self.spatial_merge_size = enc.spatial_merge_size
        self.spatial_merge_unit = enc.spatial_merge_unit
        self.fullatt_block_indexes = set(int(i) for i in enc.fullatt_block_indexes)
        # 仅取前两层
        take = min(2, len(enc.blocks))
        self.blocks = nn.ModuleList([enc.blocks[i] for i in range(take)])
        self.offset = take
        self.use_checkpoint = use_checkpoint
        pad_seq_len, pad_window_len, pad_merged_len = _compute_vision_pad_lengths(enc)
        self.pad_seq_len = pad_seq_len
        self.pad_window_len = pad_window_len
        self.pad_merged_len = pad_merged_len
        setattr(enc, "_vision_pad_seq_len", pad_seq_len)
        setattr(enc, "_vision_pad_window_len", pad_window_len)
        setattr(enc, "_vision_pad_merged_len", pad_merged_len)

    @staticmethod
    def _cat_pixel_values(vision_inputs):
        if vision_inputs is None or not isinstance(vision_inputs, dict):
            return None, None
        if "pixel_values" in vision_inputs and vision_inputs["pixel_values"] is not None:
            return vision_inputs["pixel_values"], vision_inputs.get("grid_thw", None)
        if "pixel_values_list" in vision_inputs:
            pv_list = vision_inputs["pixel_values_list"]
            if pv_list is None or len(pv_list) == 0:
                return None, vision_inputs.get("grid_thw", None)
            pixel_values = torch.cat(pv_list, dim=0)
            return pixel_values, vision_inputs.get("grid_thw", None)
        return None, vision_inputs.get("grid_thw", None)

    def _maybe_ckpt(self, layer: nn.Module, hidden_states: torch.Tensor,
                    cu_seqlens: torch.Tensor, rotary_pos_emb: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            fn = lambda hs, cs, rp: layer(hs, cu_seqlens=cs, rotary_pos_emb=rp)
            return torch.utils.checkpoint.checkpoint(fn, hidden_states, cu_seqlens, rotary_pos_emb, use_reentrant=False)
        return layer(hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)

    def forward(self, vision_inputs):
        if vision_inputs is None:
            return None, None, None

        pixel_values, grid_thw = self._cat_pixel_values(vision_inputs)
        if grid_thw is None:
            return None, None, None
        if pixel_values is None:
            grid_thw = grid_thw if torch.is_tensor(grid_thw) else torch.as_tensor(grid_thw, dtype=torch.long)
            return None, None, grid_thw

        # 对齐 dtype / device，兼容旧版 VisionStage 的行为
        if hasattr(self.enc, "get_dtype"):
            target_dtype = self.enc.get_dtype()
            if pixel_values.dtype != target_dtype:
                pixel_values = pixel_values.to(dtype=target_dtype)
        first_param = next(self.enc.parameters(), None)
        if first_param is not None:
            pixel_values = pixel_values.to(first_param.device)
        pixel_values = pixel_values.contiguous()
        if not torch.is_tensor(grid_thw):
            grid_thw = torch.as_tensor(grid_thw, dtype=torch.long, device=pixel_values.device)
        else:
            grid_thw = grid_thw.to(device=pixel_values.device)
        grid_thw = grid_thw.contiguous()

        if grid_thw.numel() == 0 or pixel_values.numel() == 0:
            try:
                rid = dist.get_rank() if dist.is_initialized() else -1
                print(f"[rank{rid}] VisionFrontAndTwoLayers.forward: empty tensors detected pixel_values_shape={tuple(pixel_values.shape)} grid_thw_shape={tuple(grid_thw.shape)}; returning None")
            except Exception:
                pass
            return None, None, None

        debug_cnt = getattr(self, "_debug_cnt", 0)
        if debug_cnt < 4:
            try:
                rid = dist.get_rank() if dist.is_initialized() else -1
                grid_info = grid_thw.tolist() if grid_thw.numel() <= 12 else grid_thw[:min(4, grid_thw.size(0))].tolist()
                print(f"[rank{rid}] VisionFrontAndTwoLayers.forward (debug): pixel_values_shape={tuple(pixel_values.shape)} grid_thw_shape={tuple(grid_thw.shape)} grid_sample={grid_info}")
            except Exception:
                pass
            self._debug_cnt = debug_cnt + 1

        # 1) Patchify
        hidden_states = self.patch_embed(pixel_values)  # [seq_len, hidden_size]

        # 2) RoPE（按 LLM 尺度的 (T,H,W)）
        rotary_pos_emb = self.enc.rot_pos_emb(grid_thw)

        # 3) 窗口索引与 varlen seqlens（窗口注意力）
        window_index, cu_window_seqlens = self.enc.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=hidden_states.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        # 将 token 与 pos_emb 都重排到“窗口顺序”
        seq_len, _ = hidden_states.size()
        s2 = self.spatial_merge_unit
        hidden_states = hidden_states.reshape(seq_len // s2, s2, -1)[window_index, :, :].reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // s2, s2, -1)[window_index, :, :].reshape(seq_len, -1)

        # 4) 全局 seqlens（供全局注意力层使用）
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0, dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        # 5) 前两层 Transformer（按配置在窗口/全局二者间切换）
        for layer_idx, blk in enumerate(self.blocks):
            global_attn = (layer_idx in self.fullatt_block_indexes)
            cu_now = cu_seqlens if global_attn else cu_window_seqlens
            hidden_states = self._maybe_ckpt(blk, hidden_states, cu_now, rotary_pos_emb)

        seq_len_cur = hidden_states.size(0)
        if seq_len_cur > self.pad_seq_len:
            hidden_states = hidden_states[: self.pad_seq_len]
            seq_len_cur = self.pad_seq_len
            try:
                rid = dist.get_rank() if dist.is_initialized() else -1
                print(f"[rank{rid}] VisionFrontAndTwoLayers.forward: seq_len {seq_len} exceeds pad_seq_len {self.pad_seq_len}, truncating.")
            except Exception:
                pass
        elif seq_len_cur < self.pad_seq_len:
            pad_rows = self.pad_seq_len - seq_len_cur
            pad_tensor = hidden_states.new_zeros(pad_rows, hidden_states.size(1))
            hidden_states = torch.cat([hidden_states, pad_tensor], dim=0)

        cu_window_tensor = cu_window_seqlens.to(hidden_states.device)
        if cu_window_tensor.numel() == 0:
            cu_window_tensor = torch.zeros(1, dtype=torch.int32, device=hidden_states.device)
        if cu_window_tensor.numel() < self.pad_window_len:
            last_val = cu_window_tensor[-1]
            pad_vals = last_val.expand(self.pad_window_len - cu_window_tensor.numel())
            cu_window_tensor = torch.cat([cu_window_tensor, pad_vals.to(cu_window_tensor.dtype)], dim=0)
        elif cu_window_tensor.numel() > self.pad_window_len:
            cu_window_tensor = cu_window_tensor[: self.pad_window_len]

        try:
            rid = dist.get_rank() if dist.is_initialized() else -1
            print(f"[rank{rid}] VisionFrontAndTwoLayers.forward: returning hidden_states_shape={tuple(hidden_states.shape)} cu_window_len={tuple(cu_window_tensor.shape)} grid_thw_shape={tuple(grid_thw.shape)}")
        except Exception:
            pass

        return hidden_states.contiguous(), cu_window_tensor.contiguous(), grid_thw.contiguous()


class VisionEncoderMidRest(nn.Module):
    """
    Stage-B (Vision):
      - 输入: (hidden_states: [seq_len, hidden], cu_window_seqlens: [W+1], grid_thw: [N,3])
      - 逻辑: 重新计算 RoPE 与 window_index（仅用于对齐 pos_emb；hidden 不再重排）→
             过剩余层（按层决定窗口/全局注意力）→ merger → 逆序恢复到原始顺序
      - 输出: (hidden_states, cu_window_seqlens, grid_thw)  # 后二者透传，便于与你既有流水线对齐
    """
    def __init__(self, enc: nn.Module, use_checkpoint: bool = False):
        super().__init__()
        self.enc = enc
        self.spatial_merge_unit = enc.spatial_merge_unit
        self.fullatt_block_indexes = set(int(i) for i in enc.fullatt_block_indexes)
        # 取剩余层
        take = min(2, len(enc.blocks))
        self.offset = take
        self.blocks = nn.ModuleList([enc.blocks[i] for i in range(self.offset, len(enc.blocks))])
        self.merger = enc.merger
        self.use_checkpoint = use_checkpoint
        self.pad_seq_len = int(getattr(enc, "_vision_pad_seq_len", _compute_vision_pad_lengths(enc)[0]))
        self.pad_window_len = int(getattr(enc, "_vision_pad_window_len", _compute_vision_pad_lengths(enc)[1]))
        self.pad_merged_len = int(getattr(enc, "_vision_pad_merged_len", _compute_vision_pad_lengths(enc)[2]))

    def _maybe_ckpt(self, layer: nn.Module, hidden_states: torch.Tensor,
                    cu_seqlens: torch.Tensor, rotary_pos_emb: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            fn = lambda hs, cs, rp: layer(hs, cu_seqlens=cs, rotary_pos_emb=rp)
            return torch.utils.checkpoint.checkpoint(fn, hidden_states, cu_seqlens, rotary_pos_emb, use_reentrant=False)
        return layer(hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)

    def forward(self, hidden_states, cu_window_seqlens, grid_thw):
        if hidden_states is None or cu_window_seqlens is None or grid_thw is None:
            return hidden_states, cu_window_seqlens, grid_thw

        if torch.is_tensor(grid_thw) and torch.all(grid_thw == 0):
            cfg = getattr(self.enc, "config", None)
            out_dim = getattr(self.merger.mlp[-1], "out_features", getattr(cfg, "out_hidden_size", getattr(self.enc, "out_hidden_size", hidden_states.size(-1) if torch.is_tensor(hidden_states) else 0)))
            device = hidden_states.device if torch.is_tensor(hidden_states) else next(self.merger.parameters()).device
            hidden_states = torch.zeros(self.pad_merged_len, out_dim, device=device)
            cu_window_tensor = torch.zeros(self.pad_window_len, dtype=torch.int32, device=device)
            return hidden_states.contiguous(), cu_window_tensor, grid_thw

        if not torch.is_tensor(hidden_states):
            raise RuntimeError("VisionEncoderMidRest expects tensor hidden_states")

        grid_thw = grid_thw.to(hidden_states.device, dtype=torch.long)
        actual_tokens = int((grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).sum().item())
        if actual_tokens <= 0:
            out_dim = getattr(self.merger.mlp[-1], "out_features", hidden_states.size(-1))
            device = hidden_states.device
            hidden_states = torch.zeros(self.pad_merged_len, out_dim, device=device)
            cu_window_tensor = torch.zeros(self.pad_window_len, dtype=torch.int32, device=device)
            return hidden_states.contiguous(), cu_window_tensor, grid_thw

        debug_cnt = getattr(self, "_debug_cnt", 0)
        if debug_cnt < 4:
            try:
                rid = dist.get_rank() if dist.is_initialized() else -1
                grid_info = grid_thw.tolist() if grid_thw.numel() <= 12 else grid_thw[:min(4, grid_thw.size(0))].tolist()
                print(f"[rank{rid}] VisionEncoderMidRest.forward (debug): hidden_states_shape={tuple(hidden_states.shape)} actual_tokens={actual_tokens} cu_window_len={tuple(cu_window_seqlens.shape)} grid_thw_shape={tuple(grid_thw.shape)} grid_sample={grid_info}")
            except Exception:
                pass
            self._debug_cnt = debug_cnt + 1

        hidden_states = hidden_states[:actual_tokens].contiguous()

        rotary_pos_emb = self.enc.rot_pos_emb(grid_thw)
        window_index, cu_window_list = self.enc.get_window_index(grid_thw)
        seq_len, _ = hidden_states.size()
        s2 = self.spatial_merge_unit
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // s2, s2, -1)[window_index, :, :].reshape(seq_len, -1)

        cu_window_tensor = torch.tensor(
            cu_window_list,
            device=hidden_states.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_window_tensor = torch.unique_consecutive(cu_window_tensor)

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0, dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for local_idx, blk in enumerate(self.blocks):
            global_attn = ((local_idx + self.offset) in self.fullatt_block_indexes)
            cu_now = cu_seqlens if global_attn else cu_window_tensor
            hidden_states = self._maybe_ckpt(blk, hidden_states, cu_now, rotary_pos_emb)

        hidden_states = self.merger(hidden_states)
        try:
            rid = dist.get_rank() if dist.is_initialized() else -1
            print(f"[rank{rid}] VisionEncoderMidRest.forward: after merger shape={tuple(hidden_states.shape)}")
        except Exception:
            pass
        reverse_indices = torch.argsort(window_index)
        hidden_states = hidden_states[reverse_indices, :]

        merged_len = hidden_states.size(0)
        out_dim = hidden_states.size(1)
        if merged_len > self.pad_merged_len:
            hidden_states = hidden_states[: self.pad_merged_len]
        elif merged_len < self.pad_merged_len:
            pad_rows = self.pad_merged_len - merged_len
            pad_tensor = hidden_states.new_zeros(pad_rows, out_dim)
            hidden_states = torch.cat([hidden_states, pad_tensor], dim=0)

        if cu_window_tensor.numel() < self.pad_window_len:
            last_val = cu_window_tensor[-1] if cu_window_tensor.numel() > 0 else cu_window_tensor.new_tensor(0)
            pad_vals = last_val.expand(self.pad_window_len - cu_window_tensor.numel())
            cu_window_tensor = torch.cat([cu_window_tensor, pad_vals.to(cu_window_tensor.dtype)], dim=0)
        elif cu_window_tensor.numel() > self.pad_window_len:
            cu_window_tensor = cu_window_tensor[: self.pad_window_len]

        return hidden_states.contiguous(), cu_window_tensor.contiguous(), grid_thw.contiguous()



# class TextStage(nn.Module):
#     def __init__(self, text_model):
#         super().__init__()
#         self.text_model = text_model
#         self.embed_tokens = text_model.embed_tokens

#     def forward(self, input_ids, attention_mask=None):
#         """
#         返回:
#             hidden: [B, T, H]
#             attn_4d: [B, 1, T, T]（含 pad 与因果遮罩）
#             position_ids: 这里返回 None，交由 Stage1 统一计算（含多模态感知索引）
#         """
#         import time
#         _t0 = time.perf_counter()
#         device = input_ids.device
#         B, T = input_ids.shape

#         # 文本嵌入
#         _t_emb0 = time.perf_counter()
#         hidden = self.embed_tokens(input_ids)
#         _t_emb1 = time.perf_counter()

#         # 4D attention mask（因果+pad）
#         if attention_mask is None:
#             attention_mask = torch.ones(B, T, dtype=torch.long, device=device)
#         _t_attn0 = time.perf_counter()
#         attn_4d = build_causal(T, device=device).expand(B, -1, -1, -1).clone()
#         pad = (attention_mask == 0).view(B, 1, 1, T)
#         attn_4d = attn_4d.masked_fill(pad, float("-inf")).contiguous()
#         _t_attn1 = time.perf_counter()

#         # 在头部阶段直接给出基础 position_ids（三路堆叠），避免下游元信息校验因 None 失败
#         _t_pos0 = time.perf_counter()
#         base_pos = torch.arange(T, device=device).unsqueeze(0).repeat(B, 1)
#         position_ids = torch.stack([base_pos, base_pos, base_pos], dim=0).contiguous()
#         _t_pos1 = time.perf_counter()
#         # 额外返回 input_ids 和 attention_mask，供 packing 阶段做多模态替换与位置计算
#         out = (hidden.contiguous(), attn_4d.contiguous(), position_ids, input_ids.contiguous(), attention_mask.contiguous())
#         try:
            
#             rid = dist.get_rank() if dist.is_initialized() else -1
#             _ms = (time.perf_counter() - _t0) * 1000.0
#             # 简要统计，避免过重代价
#             attn_sum = int(attention_mask.sum().item()) if isinstance(attention_mask, torch.Tensor) else -1
#             uid_min = int(input_ids.min().item()) if isinstance(input_ids, torch.Tensor) else -1
#             uid_max = int(input_ids.max().item()) if isinstance(input_ids, torch.Tensor) else -1
#             emb_ms = (_t_emb1 - _t_emb0) * 1000.0
#             attn_ms = (_t_attn1 - _t_attn0) * 1000.0
#             pos_ms = (_t_pos1 - _t_pos0) * 1000.0
#             ham = float(hidden.detach().abs().mean().item()) if isinstance(hidden, torch.Tensor) else float('nan')
#             try:
#                 w = self.embed_tokens.weight
#                 enorm = float(w.detach().norm().item())
#             except Exception:
#                 enorm = float('nan')
#             print(f"[rank{rid}] TextStage.forward: B={B} T={T} attn_sum={attn_sum} input_ids[min,max]=({uid_min},{uid_max}) took={_ms:.2f}ms; parts: emb={emb_ms:.2f}ms, attn={attn_ms:.2f}ms, pos={pos_ms:.2f}ms; hidden_abs_mean={ham:.3e}, embed_norm={enorm:.3e}")
#         except Exception:
#             pass
#         return out
# # -----------------------------
# # Stage1: 先做 pack（按特殊 token 替换 embedding），再计算/补全 position_ids，最后跑 L1 层 Transformer
# # -----------------------------
# class Stage1(nn.Module):
#     def __init__(self, text_model, L1):
#         super().__init__()
#         self.layers = nn.ModuleList(text_model.layers[:L1])
#         self.rotary_emb = text_model.rotary_emb
#         self.get_rope_index = getattr(text_model, "get_rope_index", None)

#         cfg = getattr(text_model, "config", None)
#         self.image_token_id = getattr(cfg, "image_token_id", None)
#         self.audio_token_id = getattr(cfg, "audio_token_id", None)

#         # 推断与适配工具（不引入新可训练参数，保持最小侵入）

#     @staticmethod
#     def _adapt_feats_to_hidden(feats: torch.Tensor, hidden_dim: int) -> torch.Tensor:
#         """将任意形状 [..., D] 的特征适配为 [N, hidden_dim]：
#         - 若 feats.dim() == 3，拉平成 [N, D]
#         - 若 D < hidden_dim：在最后一维右侧 zero-pad 保持梯度传递（使用 cat）
#         - 若 D > hidden_dim：直接截断到左侧 hidden_dim（保持梯度传递）
#         不引入可训练参数，最大限度减少侵入。
#         """
#         if feats is None:
#             return feats
#         if feats.dim() == 3:
#             feats = feats.reshape(-1, feats.size(-1))
#         elif feats.dim() == 2:
#             pass
#         else:
#             feats = feats.reshape(-1, feats.size(-1))

#         D = feats.size(-1)
#         if D == hidden_dim:
#             return feats
#         if D < hidden_dim:
#             pad = torch.zeros(feats.size(0), hidden_dim - D, device=feats.device, dtype=feats.dtype)
#             return torch.cat([feats, pad], dim=-1)
#         else:  # D > hidden_dim
#             return feats[:, :hidden_dim]

#     @staticmethod
#     def _infer_token_id_by_count(input_ids: torch.Tensor, expected_count: int, prefer_large_id: bool = True) -> int | None:
#         """在 input_ids 中根据出现次数推断特殊 token 的 id。
#         - 统计每个 id 的出现次数，寻找与 expected_count 恰好相等的 id；
#         - 若有多个候选，返回数值更大的 id（通常新增特殊 token id 较大）；
#         - 若找不到，返回 None。
#         """
#         if input_ids is None or expected_count <= 0:
#             return None
#         ids = input_ids.reshape(-1)
#         uniq, cnt = torch.unique(ids, return_counts=True)
#         mask = (cnt == expected_count)
#         cand = uniq[mask]
#         if cand.numel() == 0:
#             return None
#         if cand.numel() == 1:
#             return int(cand.item())
#         # 多候选：选取较大的 id（启发式）
#         return int(torch.max(cand).item()) if prefer_large_id else int(torch.min(cand).item())

#     @staticmethod
#     def _infer_token_id_by_longest_run(input_ids: torch.Tensor) -> tuple[int | None, int]:
#         """在 input_ids 的单样本序列中寻找最长连续相同 id 的片段，返回 (id, run_len)。
#         B>1 时取第一个样本。若失败，返回 (None, 0)。
#         """
#         if input_ids is None or not isinstance(input_ids, torch.Tensor) or input_ids.dim() != 2 or input_ids.numel() == 0:
#             return None, 0
#         ids = input_ids[0].detach()
#         best_id = None
#         best_len = 0
#         cur_id = None
#         cur_len = 0
#         for v in ids:
#             v = int(v.item())
#             if cur_id is None or v != cur_id:
#                 if cur_len > best_len:
#                     best_id, best_len = cur_id, cur_len
#                 cur_id, cur_len = v, 1
#             else:
#                 cur_len += 1
#         if cur_len > best_len:
#             best_id, best_len = cur_id, cur_len
#         return best_id, best_len

#     @staticmethod
#     def _align_feats_length(feats: torch.Tensor, target_len: int) -> torch.Tensor:
#         """将 feats 的第一维（N）对齐到 target_len：
#         - N == target_len: 直接返回
#         - N > target_len: 截断到前 target_len
#         - N < target_len: 重复最后一个向量或零填充到 target_len（使用重复以保持梯度可传）
#         """
#         if feats is None:
#             return feats
#         N = feats.size(0)
#         if N == target_len:
#             return feats
#         if N > target_len:
#             return feats[:target_len]
#         # N < target_len
#         if N == 0:
#             # 退化：构造全零（保持 dtype/device）
#             return torch.zeros(target_len, feats.size(1), device=feats.device, dtype=feats.dtype)
#         # 重复最后一个向量填充到目标长度
#         pad_count = target_len - N
#         last_vec = feats[-1:].expand(pad_count, -1)
#         return torch.cat([feats, last_vec], dim=0)

#     @staticmethod
#     def _replace_feats_by_token_id(input_ids, inputs_embeds, feats, special_token_id):
#         if feats is None or special_token_id is None or input_ids is None:
#             try:
                
#                 rid = dist.get_rank() if dist.is_initialized() else -1
#                 print(f"[rank{rid}] Stage1._replace_feats_by_token_id: skip replacement due to feats/input_ids/token_id None. token_id={special_token_id} feats={(None if feats is None else tuple(feats.shape))} input_ids_is_none={input_ids is None}")
#             except Exception:
#                 pass
#             return inputs_embeds
#         flat_mask = (input_ids == special_token_id).reshape(-1)
#         n_tokens = int(flat_mask.sum().item())
#         if n_tokens == 0:
#             try:
                
#                 rid = dist.get_rank() if dist.is_initialized() else -1
#                 print(f"[rank{rid}] Stage1._replace_feats_by_token_id: no tokens for token_id={special_token_id}; feats_shape={(tuple(feats.shape) if hasattr(feats,'shape') else type(feats).__name__)}")
#             except Exception:
#                 pass
#             return inputs_embeds
#         if feats.size(0) != n_tokens:
#             raise RuntimeError(
#                 f"Feature count mismatch for token_id={special_token_id}: "
#                 f"tokens={n_tokens} vs feats={feats.size(0)}"
#             )
#         # 使用 out-of-place scatter 构造替换后的新张量，避免对 view/leaf 的就地写入
#         emb_flat = inputs_embeds.reshape(-1, inputs_embeds.size(-1))
#         feats = feats.to(device=emb_flat.device, dtype=emb_flat.dtype)
#         idx = torch.nonzero(flat_mask, as_tuple=False).squeeze(1).to(dtype=torch.long)
#         idx2 = idx.unsqueeze(1).expand(-1, emb_flat.size(1))
#         out_flat = emb_flat.scatter(0, idx2, feats)
#         try:
            
#             rid = dist.get_rank() if dist.is_initialized() else -1
#             print(f"[rank{rid}] Stage1._replace_feats_by_token_id: replaced token_id={special_token_id} count={n_tokens} feats_shape={tuple(feats.shape)} emb_dim={emb_flat.size(-1)}")
#         except Exception:
#             pass
#         return out_flat.view_as(inputs_embeds)

#     def forward(self, *args, **kwargs):
#         """
#         期望输入（来自各 stage 的聚合）：
#             基础三元组：
#                 hidden, attn_mask_4d, position_ids (可能为 None)
#             以及 pack/位置计算所需的 kwargs：
#                 input_ids: [B, T]（用于替换定位 & 重新计算 position_ids）
#                 attention_mask_2d: [B, T]（若需用 get_rope_index 计算位置）
#                 image_embeds: [sum_img_tokens, H] 或 None
#                 audio_embeds: [sum_aud_tokens, H] 或 None
#                 grid_thw: 视觉网格元信息（VisionStage 产出），用于 get_rope_index
#                 image_token_id/audio_token_id: 可覆盖 config 默认
#         """
#         # 兼容三元组
#         if len(args) >= 3:
#             hidden, attn_mask, position_ids = args[:3]
#         else:
#             hidden = args[0] if len(args) > 0 else kwargs['hidden']
#             attn_mask = args[1] if len(args) > 1 else kwargs['attn_mask']
#             position_ids = args[2] if len(args) > 2 else kwargs.get('position_ids', None)

#         # pack 所需
#         input_ids = kwargs.get('input_ids', None)
#         attention_mask_2d = kwargs.get('attention_mask_2d', None)
#         grid_thw = kwargs.get('grid_thw', None)

#         image_embeds = kwargs.get('image_embeds', None)
#         audio_embeds = kwargs.get('audio_embeds', None)
#         image_token_id = kwargs.get('image_token_id', self.image_token_id)
#         audio_token_id = kwargs.get('audio_token_id', self.audio_token_id)

#         # Debug: summarize inputs to Stage1
#         try:
            
#             rid = dist.get_rank() if dist.is_initialized() else -1
#             def _s(x):
#                 return tuple(x.shape) if isinstance(x, torch.Tensor) else None
#             img_cnt = None
#             aud_cnt = None
#             if isinstance(input_ids, torch.Tensor):
#                 try:
#                     img_cnt = int((input_ids == (image_token_id if image_token_id is not None else -999999)).sum().item())
#                     aud_cnt = int((input_ids == (audio_token_id if audio_token_id is not None else -999999)).sum().item())
#                 except Exception:
#                     pass
#             print(
#                 f"[rank{rid}] Stage1.forward: input_ids={_s(input_ids)} attention_mask_2d={_s(attention_mask_2d)} grid_thw={_s(grid_thw)} "
#                 f"image_embeds={_s(image_embeds)} audio_embeds={_s(audio_embeds)} image_token_id={image_token_id} audio_token_id={audio_token_id} "
#                 f"counts(img,aud)=({img_cnt},{aud_cnt})"
#             )
#         except Exception:
#             pass

#         # 1) pack：若提供了任一模态特征且有 input_ids，则做按位替换
#         if input_ids is not None:
#             # 目标 hidden 维度
#             H = hidden.size(-1)

#             # 视觉替换：维度自适配 + id 推断（计数匹配失败时用最长连续片段回退，并对齐长度）
#             if image_embeds is not None:
#                 img_feats = self._adapt_feats_to_hidden(image_embeds, H)
#                 # 若未提供 token_id，依据计数推断
#                 if image_token_id is None:
#                     image_token_id = self._infer_token_id_by_count(input_ids, expected_count=img_feats.size(0))
#                     try:
                        
#                         rid = dist.get_rank() if dist.is_initialized() else -1
#                         print(f"[rank{rid}] Stage1.forward: inferred image_token_id={image_token_id} from count={img_feats.size(0)}")
#                     except Exception:
#                         pass
#                 if image_token_id is None:
#                     # 回退：最长连续片段
#                     run_id, run_len = self._infer_token_id_by_longest_run(input_ids)
#                     if run_id is not None and run_len > 0:
#                         image_token_id = run_id
#                         try:
                            
#                             rid = dist.get_rank() if dist.is_initialized() else -1
#                             print(f"[rank{rid}] Stage1.forward: fallback longest-run image_token_id={image_token_id}, run_len={run_len}")
#                         except Exception:
#                             pass
#                         # 根据 run_len 对齐特征长度
#                         img_feats = self._align_feats_length(img_feats, run_len)
#                 if image_token_id is not None:
#                     try:
#                         # 计算 mask 长度，确保与特征长度一致
#                         flat_mask = (input_ids == image_token_id).reshape(-1)
#                         n_tokens = int(flat_mask.sum().item())
#                         if img_feats.size(0) != n_tokens:
#                             img_feats = self._align_feats_length(img_feats, n_tokens)
#                         hidden = self._replace_feats_by_token_id(input_ids, hidden, img_feats, image_token_id)
#                     except Exception as e:
#                         try:
                            
#                             rid = dist.get_rank() if dist.is_initialized() else -1
#                             print(f"[rank{rid}] Stage1.forward: image replace failed with {type(e).__name__}: {e}")
#                         except Exception:
#                             pass
#                 else:
#                     try:
                        
#                         rid = dist.get_rank() if dist.is_initialized() else -1
#                         print(f"[rank{rid}] Stage1.forward: could not infer image_token_id; skip image replace")
#                     except Exception:
#                         pass
#             else:
#                 try:
                    
#                     rid = dist.get_rank() if dist.is_initialized() else -1
#                     print(f"[rank{rid}] Stage1.forward: skip image replace; image_embeds is None")
#                 except Exception:
#                     pass

#             # 音频替换：拉平为 [N, D] + 维度自适配 + id 推断（计数失败回退最长连续片段，并对齐长度）
#             if audio_embeds is not None:
#                 aud_feats = self._adapt_feats_to_hidden(audio_embeds, H)
#                 if audio_token_id is None:
#                     audio_token_id = self._infer_token_id_by_count(input_ids, expected_count=aud_feats.size(0))
#                     try:
                        
#                         rid = dist.get_rank() if dist.is_initialized() else -1
#                         print(f"[rank{rid}] Stage1.forward: inferred audio_token_id={audio_token_id} from count={aud_feats.size(0)}")
#                     except Exception:
#                         pass
#                 if audio_token_id is None:
#                     run_id, run_len = self._infer_token_id_by_longest_run(input_ids)
#                     if run_id is not None and run_len > 0:
#                         audio_token_id = run_id
#                         try:
                            
#                             rid = dist.get_rank() if dist.is_initialized() else -1
#                             print(f"[rank{rid}] Stage1.forward: fallback longest-run audio_token_id={audio_token_id}, run_len={run_len}")
#                         except Exception:
#                             pass
#                         aud_feats = self._align_feats_length(aud_feats, run_len)
#                 if audio_token_id is not None:
#                     try:
#                         flat_mask = (input_ids == audio_token_id).reshape(-1)
#                         n_tokens = int(flat_mask.sum().item())
#                         if aud_feats.size(0) != n_tokens:
#                             aud_feats = self._align_feats_length(aud_feats, n_tokens)
#                         hidden = self._replace_feats_by_token_id(input_ids, hidden, aud_feats, audio_token_id)
#                     except Exception as e:
#                         try:
                            
#                             rid = dist.get_rank() if dist.is_initialized() else -1
#                             print(f"[rank{rid}] Stage1.forward: audio replace failed with {type(e).__name__}: {e}")
#                         except Exception:
#                             pass
#                 else:
#                     try:
                        
#                         rid = dist.get_rank() if dist.is_initialized() else -1
#                         print(f"[rank{rid}] Stage1.forward: could not infer audio_token_id; skip audio replace")
#                     except Exception:
#                         pass
#             else:
#                 try:
                    
#                     rid = dist.get_rank() if dist.is_initialized() else -1
#                     print(f"[rank{rid}] Stage1.forward: skip audio replace; audio_embeds is None")
#                 except Exception:
#                     pass
#         else:
#             try:
                
#                 rid = dist.get_rank() if dist.is_initialized() else -1
#                 print(f"[rank{rid}] Stage1.forward: input_ids is None; cannot perform multimodal replacement")
#             except Exception:
#                 pass

#         # 2) 计算/补全 position_ids：
#         #    - 若 TextStage 已给出 position_ids，可直接沿用；
#         #    - 否则（为 None），优先使用 get_rope_index(感知 grid_thw) 计算；
#         #    - 若模型无 get_rope_index，则回退到等距 1D 位置编码。
#         if position_ids is None:
#             if self.get_rope_index is not None and input_ids is not None and attention_mask_2d is not None:
#                 position_ids, _ = self.get_rope_index(
#                     input_ids,
#                     image_grid_thw=grid_thw,
#                     video_grid_thw=None,
#                     attention_mask=attention_mask_2d
#                 )
#             else:
#                 # 回退：三路堆叠（与原实现保持一致的形状）
#                 B, T, _ = hidden.shape
#                 base_pos = torch.arange(T, device=hidden.device).unsqueeze(0).repeat(B, 1)
#                 position_ids = torch.stack([base_pos, base_pos, base_pos], dim=0).contiguous()

#         try:
            
#             rid = dist.get_rank() if dist.is_initialized() else -1
#             print(f"[rank{rid}] Stage1.forward: final position_ids shape={(tuple(position_ids.shape) if isinstance(position_ids, torch.Tensor) else None)}")
#         except Exception:
#             pass

#         # 3) 原有 Transformer 前向保持不变
#         if position_ids.dim() == 2:
#             position_ids = position_ids.unsqueeze(0).repeat(3, 1, 1)

#         pos_emb = self.rotary_emb(hidden, position_ids)
#         for blk in self.layers:
#             hidden = blk(
#                 hidden_states=hidden,
#                 attention_mask=attn_mask,
#                 position_ids=position_ids,
#                 position_embeddings=pos_emb,
#                 output_attentions=False,
#                 use_cache=False
#             )[0]

#         return hidden.contiguous(), attn_mask.contiguous(), position_ids.contiguous()
# 注意
# 1 本补丁仅包含新增的两个 stage 定义类 以及 对 Stage1 与 main 的关键修改示例
# 2 请将本文件中的类与改动片段合并到你的 qwen2.5_omni_3b_devide_devide_head_video.py 中
# 3 不改动你现有的 VisionFrontAndTwoLayers 与 VisionEncoderMidRest 实现 直接复用

# =====================
# 新增一 Text+Video 前段阶段
# =====================
class TextAndVideoFrontAndTwoLayers(nn.Module):
    """
    负责两件事
    1 完整执行文本的 head 逻辑 产出与原 TextStage 完全一致的五元组
       (hidden, attn_4d, position_ids, input_ids, attention_mask)
    2 借用 VisionFrontAndTwoLayers 对视频执行前半段编码 返回给下游承接

    forward 返回一个长度为 8 的张量元组：
      (hidden, attn_4d, position_ids, input_ids, attention_mask,
       vid_hidden_front, vid_cu_window_seqlens, video_grid_thw)
    若无视频输入，对应张量会退化为 0 长度占位，保持纯张量结构
    """
    def __init__(self, text_model: nn.Module, vision_front: nn.Module):
        super().__init__()
        self.text_model = text_model
        self.embed_tokens = text_model.embed_tokens
        self.vision_front = vision_front

    @torch.no_grad()
    def _build_base_pos_ids(self, B: int, T: int, device: torch.device) -> torch.Tensor:
        base = torch.arange(T, device=device).unsqueeze(0).repeat(B, 1)
        return torch.stack([base, base, base], dim=0).contiguous()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        video_inputs: Optional[Dict[str, torch.Tensor]] = None,
    ):
        device = input_ids.device
        try:
            rid = dist.get_rank() if dist.is_initialized() else -1
            if isinstance(video_inputs, dict):
                vk = sorted(list(video_inputs.keys()))
                pv = video_inputs.get("pixel_values_videos") or video_inputs.get("pixel_values")
                pv_shape = tuple(pv.shape) if isinstance(pv, torch.Tensor) else None
                print(f"[rank{rid}] TextAndVideoFront.forward: input_ids={tuple(input_ids.shape)} attention_mask={tuple(attention_mask.shape) if isinstance(attention_mask, torch.Tensor) else None} "
                      f"video_keys={vk} pixel_values_shape={pv_shape}")
            else:
                print(f"[rank{rid}] TextAndVideoFront.forward: video_inputs type={type(video_inputs).__name__}")
        except Exception:
            pass
        # 文本 head 与你原 TextStage 一致
        B, T = input_ids.shape
        hidden = self.embed_tokens(input_ids)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        # 构造 [B,1,T,T] 因果与 pad 遮罩 复用已有逻辑 这里给出等价占位
        causal = torch.tril(torch.ones(T, T, device=device))
        pad = attention_mask[:, None, None, :]
        attn_4d = causal[None, None, :, :] * pad
        position_ids = self._build_base_pos_ids(B, T, device)
        hidden = hidden.contiguous()
        attn_4d = attn_4d.contiguous()
        position_ids = position_ids.contiguous()
        input_ids = input_ids.contiguous()
        attention_mask = attention_mask.contiguous()

        # 视频前段：统一转成张量占位，避免 None 进入管线
        vid_hidden_front: torch.Tensor
        vid_cu_window_seqlens: torch.Tensor
        video_grid_thw: torch.Tensor
        if isinstance(video_inputs, dict):
            rid = dist.get_rank() if dist.is_initialized() else -1
            print(f"[rank{rid}] TextAndVideoFront.forward: raw video_inputs keys={list(video_inputs.keys())}")

            # Normalize incoming keys to what vision_front expects
            norm_inputs: dict[str, torch.Tensor] = {}
            pv = video_inputs.get("pixel_values_videos")
            if pv is None:
                pv = video_inputs.get("pixel_values")
            if isinstance(pv, torch.Tensor):
                norm_inputs["pixel_values_videos"] = pv
                print(f"[rank{rid}] TextAndVideoFront.forward: pixel_values_videos shape={tuple(pv.shape)}")

            grid = video_inputs.get("video_grid_thw") or video_inputs.get("grid_thw") or video_inputs.get("image_grid_thw")
            if isinstance(grid, torch.Tensor):
                norm_inputs["video_grid_thw"] = grid
                print(f"[rank{rid}] TextAndVideoFront.forward: video_grid_thw shape={tuple(grid.shape)}")

            if norm_inputs:
                if any(t.numel() == 0 for t in norm_inputs.values() if isinstance(t, torch.Tensor)):
                    print(f"[rank{rid}] TextAndVideoFront.forward: norm_inputs contain empty tensors; skip vision front")
                else:
                    print(f"[rank{rid}] TextAndVideoFront.forward: invoking vision_front with keys {list(norm_inputs.keys())}")
                    vh, vc, vg = self.vision_front(norm_inputs)
            try:
                rid = dist.get_rank() if dist.is_initialized() else -1
                print(f"[rank{rid}] TextAndVideoFront.forward: vision_front -> hidden={tuple(vh.shape) if isinstance(vh, torch.Tensor) else None} "
                      f"cu={tuple(vc.shape) if isinstance(vc, torch.Tensor) else None} grid={tuple(vg.shape) if isinstance(vg, torch.Tensor) else None}")
            except Exception:
                pass
            if isinstance(vh, torch.Tensor) and vh.numel() > 0:
                vid_hidden_front = vh.contiguous()
            else:
                vid_hidden_front = hidden.new_empty(0, hidden.size(-1))
            if isinstance(vc, torch.Tensor) and vc.numel() > 0:
                vid_cu_window_seqlens = vc.contiguous()
            else:
                vid_cu_window_seqlens = torch.zeros(
                    1,
                    dtype=torch.int32,
                    device=hidden.device,
                )
            if isinstance(vg, torch.Tensor) and vg.numel() > 0:
                video_grid_thw = vg.to(dtype=torch.long, device=hidden.device).contiguous()
            else:
                video_grid_thw = torch.zeros(
                    (0, 3),
                    dtype=torch.long,
                    device=hidden.device,
                )
        else:
            vid_hidden_front = hidden.new_empty(0, hidden.size(-1))
            vid_cu_window_seqlens = torch.zeros(
                1,
                dtype=torch.int32,
                device=hidden.device,
            )
            video_grid_thw = torch.zeros(
                (0, 3),
                dtype=torch.long,
                device=hidden.device,
            )
        try:
            rid = dist.get_rank() if dist.is_initialized() else -1
            print(f"[rank{rid}] TextAndVideoFront.forward: return video_front shapes hidden={tuple(vid_hidden_front.shape)} cu={tuple(vid_cu_window_seqlens.shape)} grid={tuple(video_grid_thw.shape)}")
        except Exception:
            pass

        return (
            hidden,
            attn_4d,
            position_ids,
            input_ids,
            attention_mask,
            vid_hidden_front,
            vid_cu_window_seqlens,
            video_grid_thw,
        )


# =====================
# 新增二 Text+Video 后段阶段
# =====================
class TextAndVideoEncoderMidRest(nn.Module):
    """
    承接 TextAndVideoFrontAndTwoLayers 给出的视频前段表示
    继续用 VisionEncoderMidRest 计算得到最终 video_embeds
    文本相关张量原样透传 不做任何改变

    forward 输入
      与前一阶段输出一致的 8 个张量：
      (hidden, attn_4d, position_ids, input_ids, attention_mask,
       vid_hidden_front, vid_cu_window_seqlens, video_grid_thw)

    forward 输出一个张量元组：
      (hidden, attn_4d, position_ids, input_ids, attention_mask,
       video_embeds, video_grid_thw)
    若当前 microbatch 无视频，则 video_embeds / video_grid_thw 为空张量
    """
    def __init__(self, vision_midrest: nn.Module):
        super().__init__()
        self.vision_midrest = vision_midrest

    def forward(self, *args):
        if len(args) < 5:
            raise RuntimeError(
                "TextAndVideoEncoderMidRest expects at least 5 tensors: hidden, attn_4d, position_ids, input_ids, attention_mask."
            )

        hidden = args[0].contiguous()
        attn_4d = args[1].contiguous()
        position_ids = args[2].contiguous()
        input_ids = args[3].contiguous()
        attention_mask = args[4].contiguous()

        # 默认的占位，避免 None 传播
        device = hidden.device if isinstance(hidden, torch.Tensor) else torch.device("cpu")
        hidden_dim = hidden.size(-1) if isinstance(hidden, torch.Tensor) and hidden.dim() >= 2 else 0
        video_embeds = (
            hidden.new_empty(0, hidden_dim)
            if isinstance(hidden, torch.Tensor) and hidden_dim > 0
            else torch.empty(0, device=device)
        )
        video_grid_thw = (
            input_ids.new_empty(0, 3, dtype=torch.long)
            if isinstance(input_ids, torch.Tensor)
            else torch.empty(0, 3, dtype=torch.long, device=device)
        )

        if len(args) >= 8:
            vid_hidden_front = args[5]
            vid_cu_window_seqlens = args[6]
            video_grid_front = args[7]
            if (
                isinstance(vid_hidden_front, torch.Tensor)
                and vid_hidden_front.numel() > 0
                and isinstance(vid_cu_window_seqlens, torch.Tensor)
                and isinstance(video_grid_front, torch.Tensor)
            ):
                vid_out, _, grid_out = self.vision_midrest(
                    vid_hidden_front,
                    vid_cu_window_seqlens,
                    video_grid_front,
                )
                video_embeds = vid_out.contiguous()
                video_grid_thw = grid_out.to(dtype=torch.long).contiguous()
                try:
                    rid = dist.get_rank() if dist.is_initialized() else -1
                    print(f"[rank{rid}] TextAndVideoEncoderMidRest.forward: vision_midrest -> video_embeds={tuple(video_embeds.shape)} grid={tuple(video_grid_thw.shape)}")
                except Exception:
                    pass
            else:
                try:
                    rid = dist.get_rank() if dist.is_initialized() else -1
                    print(f"[rank{rid}] TextAndVideoEncoderMidRest.forward: video inputs empty -> vid_hidden_front={type(vid_hidden_front)} numel={(vid_hidden_front.numel() if isinstance(vid_hidden_front, torch.Tensor) else 'NA')}")
                except Exception:
                    pass
        else:
            try:
                rid = dist.get_rank() if dist.is_initialized() else -1
                print(f"[rank{rid}] TextAndVideoEncoderMidRest.forward: args len={len(args)} (<8) video branch skipped")
            except Exception:
                pass

        return (
            hidden,
            attn_4d,
            position_ids,
            input_ids,
            attention_mask,
            video_embeds,
            video_grid_thw,
        )


# =====================
# 修改 Stage1 使其能注入视频特征
# 说明
# 1 保持你原有图像与音频路径不变
# 2 新增 video_token_id 与 video_embeds 支持
# 3 若传入 video_outputs 则依据 video_token_id 将其按位替换到 hidden 中
# =====================
class Stage1(nn.Module):
    def __init__(self, text_model, L1):
        super().__init__()
        self.layers = nn.ModuleList(text_model.layers[:L1])
        self.rotary_emb = text_model.rotary_emb
        self.get_rope_index = getattr(text_model, "get_rope_index", None)

        cfg = getattr(text_model, "config", None)
        self.image_token_id = getattr(cfg, "image_token_id", None)
        self.audio_token_id = getattr(cfg, "audio_token_id", None)
        self.video_token_id = getattr(cfg, "video_token_id", None)

    @staticmethod
    def _adapt_feats_to_hidden(feats: torch.Tensor, hidden_dim: int) -> torch.Tensor:
        """将任意形状 [..., D] 的特征适配为 [N, hidden_dim]。"""
        if feats is None:
            return feats
        if feats.dim() == 3:
            feats = feats.reshape(-1, feats.size(-1))
        elif feats.dim() != 2:
            feats = feats.reshape(-1, feats.size(-1))

        D = feats.size(-1)
        if D == hidden_dim:
            return feats
        if D < hidden_dim:
            pad = torch.zeros(feats.size(0), hidden_dim - D, device=feats.device, dtype=feats.dtype)
            return torch.cat([feats, pad], dim=-1)
        return feats[:, :hidden_dim]

    @staticmethod
    def _infer_token_id_by_count(input_ids: torch.Tensor, expected_count: int, prefer_large_id: bool = True) -> Optional[int]:
        if input_ids is None or expected_count <= 0:
            return None
        ids = input_ids.reshape(-1)
        uniq, cnt = torch.unique(ids, return_counts=True)
        mask = cnt == expected_count
        cand = uniq[mask]
        if cand.numel() == 0:
            return None
        if cand.numel() == 1:
            return int(cand.item())
        return int(torch.max(cand).item()) if prefer_large_id else int(torch.min(cand).item())

    @staticmethod
    def _infer_token_id_by_longest_run(input_ids: torch.Tensor) -> tuple[Optional[int], int]:
        if (
            input_ids is None
            or not isinstance(input_ids, torch.Tensor)
            or input_ids.dim() != 2
            or input_ids.numel() == 0
        ):
            return None, 0
        ids = input_ids[0].detach()
        best_id = None
        best_len = 0
        cur_id = None
        cur_len = 0
        for v in ids:
            v = int(v.item())
            if cur_id is None or v != cur_id:
                if cur_len > best_len:
                    best_id, best_len = cur_id, cur_len
                cur_id, cur_len = v, 1
            else:
                cur_len += 1
        if cur_len > best_len:
            best_id, best_len = cur_id, cur_len
        return best_id, best_len

    @staticmethod
    def _align_feats_length(feats: torch.Tensor, target_len: int) -> torch.Tensor:
        if feats is None:
            return feats
        N = feats.size(0)
        if N == target_len:
            return feats
        if N > target_len:
            return feats[:target_len]
        if N == 0:
            return torch.zeros(target_len, feats.size(1), device=feats.device, dtype=feats.dtype)
        pad_count = target_len - N
        last_vec = feats[-1:].expand(pad_count, -1)
        return torch.cat([feats, last_vec], dim=0)

    @staticmethod
    def _replace_feats_by_token_id(input_ids, inputs_embeds, feats, special_token_id):
        if feats is None or special_token_id is None or input_ids is None:
            try:
                rid = dist.get_rank() if dist.is_initialized() else -1
                print(f"[rank{rid}] Stage1._replace_feats_by_token_id: skip replacement token_id={special_token_id} feats={(None if feats is None else tuple(feats.shape))}")
            except Exception:
                pass
            return inputs_embeds

        flat_mask = (input_ids == special_token_id).reshape(-1)
        n_tokens = int(flat_mask.sum().item())
        if n_tokens == 0:
            try:
                rid = dist.get_rank() if dist.is_initialized() else -1
                print(f"[rank{rid}] Stage1._replace_feats_by_token_id: no tokens for token_id={special_token_id}")
            except Exception:
                pass
            return inputs_embeds

        if feats.size(0) != n_tokens:
            raise RuntimeError(
                f"Feature count mismatch for token_id={special_token_id}: tokens={n_tokens} vs feats={feats.size(0)}"
            )

        emb_flat = inputs_embeds.reshape(-1, inputs_embeds.size(-1))
        feats = feats.to(device=emb_flat.device, dtype=emb_flat.dtype)
        idx = torch.nonzero(flat_mask, as_tuple=False).squeeze(1).to(dtype=torch.long)
        idx2 = idx.unsqueeze(1).expand(-1, emb_flat.size(1))
        out_flat = emb_flat.scatter(0, idx2, feats)
        try:
            rid = dist.get_rank() if dist.is_initialized() else -1
            print(f"[rank{rid}] Stage1._replace_feats_by_token_id: replaced token_id={special_token_id} count={n_tokens}")
        except Exception:
            pass
        return out_flat.view_as(inputs_embeds)

    def forward(self, *args, **kwargs):
        if len(args) >= 3:
            hidden, attn_mask, position_ids = args[:3]
        else:
            hidden = kwargs["hidden"]
            attn_mask = kwargs["attn_mask"]
            position_ids = kwargs.get("position_ids", None)

        input_ids = kwargs.get("input_ids", None)
        attention_mask_2d = kwargs.get("attention_mask_2d", None)
        grid_thw = kwargs.get("grid_thw", None)

        image_embeds = kwargs.get("image_embeds", None)
        audio_embeds = kwargs.get("audio_embeds", None)
        video_outputs = kwargs.get("video_outputs", None)
        video_front = kwargs.get("video_front", None)
        video_embeds = kwargs.get("video_embeds", None)

        image_token_id = kwargs.get("image_token_id", self.image_token_id)
        audio_token_id = kwargs.get("audio_token_id", self.audio_token_id)
        video_token_id = kwargs.get("video_token_id", self.video_token_id)

        if isinstance(video_outputs, dict):
            video_embeds = video_outputs.get("video_embeds", video_embeds)
            if grid_thw is None:
                grid_thw = video_outputs.get("video_grid_thw", grid_thw)
        if video_embeds is None and video_front is not None:
            try:
                rid = dist.get_rank() if dist.is_initialized() else -1
                print(f"[rank{rid}] Stage1.forward: video_front provided but no decoder available; skip.")
            except Exception:
                pass

        try:
            rid = dist.get_rank() if dist.is_initialized() else -1
            def _s(x):
                return tuple(x.shape) if isinstance(x, torch.Tensor) else None
            img_cnt = aud_cnt = vid_cnt = None
            if isinstance(input_ids, torch.Tensor):
                try:
                    if image_token_id is not None:
                        img_cnt = int((input_ids == image_token_id).sum().item())
                    if audio_token_id is not None:
                        aud_cnt = int((input_ids == audio_token_id).sum().item())
                    if video_token_id is not None:
                        vid_cnt = int((input_ids == video_token_id).sum().item())
                except Exception:
                    pass
            print(
                f"[rank{rid}] Stage1.forward: input_ids={_s(input_ids)} attention_mask_2d={_s(attention_mask_2d)} grid_thw={_s(grid_thw)} "
                f"image_embeds={_s(image_embeds)} audio_embeds={_s(audio_embeds)} video_embeds={_s(video_embeds)} "
                f"token_ids(img,aud,vid)=({image_token_id},{audio_token_id},{video_token_id}) "
                f"counts(img,aud,vid)=({img_cnt},{aud_cnt},{vid_cnt})"
            )
        except Exception:
            pass

        if input_ids is not None:
            H = hidden.size(-1)

            if image_embeds is not None:
                img_feats = self._adapt_feats_to_hidden(image_embeds, H)
                if image_token_id is None:
                    image_token_id = self._infer_token_id_by_count(input_ids, img_feats.size(0))
                if image_token_id is None:
                    run_id, run_len = self._infer_token_id_by_longest_run(input_ids)
                    if run_id is not None and run_len > 0:
                        image_token_id = run_id
                        img_feats = self._align_feats_length(img_feats, run_len)
                if image_token_id is not None:
                    flat_mask = (input_ids == image_token_id).reshape(-1)
                    n_tokens = int(flat_mask.sum().item())
                    if img_feats.size(0) != n_tokens:
                        img_feats = self._align_feats_length(img_feats, n_tokens)
                    hidden = self._replace_feats_by_token_id(input_ids, hidden, img_feats, image_token_id)

            if audio_embeds is not None:
                aud_feats = self._adapt_feats_to_hidden(audio_embeds, H)
                if audio_token_id is None:
                    audio_token_id = self._infer_token_id_by_count(input_ids, aud_feats.size(0))
                if audio_token_id is None:
                    run_id, run_len = self._infer_token_id_by_longest_run(input_ids)
                    if run_id is not None and run_len > 0:
                        audio_token_id = run_id
                        aud_feats = self._align_feats_length(aud_feats, run_len)
                if audio_token_id is not None:
                    flat_mask = (input_ids == audio_token_id).reshape(-1)
                    n_tokens = int(flat_mask.sum().item())
                    if aud_feats.size(0) != n_tokens:
                        aud_feats = self._align_feats_length(aud_feats, n_tokens)
                    hidden = self._replace_feats_by_token_id(input_ids, hidden, aud_feats, audio_token_id)

            if video_embeds is not None:
                vid_feats = self._adapt_feats_to_hidden(video_embeds, H)
                if video_token_id is None:
                    video_token_id = self._infer_token_id_by_count(input_ids, vid_feats.size(0))
                if video_token_id is None:
                    run_id, run_len = self._infer_token_id_by_longest_run(input_ids)
                    if run_id is not None and run_len > 0:
                        video_token_id = run_id
                        vid_feats = self._align_feats_length(vid_feats, run_len)
                if video_token_id is not None:
                    flat_mask = (input_ids == video_token_id).reshape(-1)
                    n_tokens = int(flat_mask.sum().item())
                    if vid_feats.size(0) != n_tokens:
                        vid_feats = self._align_feats_length(vid_feats, n_tokens)
                    hidden = self._replace_feats_by_token_id(input_ids, hidden, vid_feats, video_token_id)

        else:
            try:
                rid = dist.get_rank() if dist.is_initialized() else -1
                print(f"[rank{rid}] Stage1.forward: input_ids is None; skip multimodal injection")
            except Exception:
                pass

        if position_ids is None:
            if self.get_rope_index is not None and input_ids is not None and attention_mask_2d is not None:
                position_ids, _ = self.get_rope_index(
                    input_ids,
                    image_grid_thw=grid_thw,
                    video_grid_thw=None,
                    attention_mask=attention_mask_2d,
                )
            else:
                B, T, _ = hidden.shape
                base_pos = torch.arange(T, device=hidden.device).unsqueeze(0).repeat(B, 1)
                position_ids = torch.stack([base_pos, base_pos, base_pos], dim=0).contiguous()

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
                use_cache=False,
            )[0]

        return hidden.contiguous(), attn_mask.contiguous(), position_ids.contiguous()

class Stage2(nn.Module):
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


class Stage3(nn.Module):
    def __init__(self, full_thinker, text_model, L2):
        super().__init__()
        self.layers = nn.ModuleList(text_model.layers[L2:])
        self.norm = text_model.norm
        self.lm_head = full_thinker.lm_head
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
        hidden = self.norm(hidden)
        logits = self.lm_head(hidden)
        return logits

def create_pipeline_actions():

    # [stage],[rank],[id],[action type],[microbatch],[dest_rank],[upstream],[dependency], [split_parts], [chunk_deps], [multimodality] id不用管，不影响运行
    # Rank 0 (Stage 0)
    rank0_actions = [
        _Action(0, 0, 0, _ComputationType.FORWARD, (0), None, None, None, None, None, ["audio"]),
        _Action(0, 0, 1, _ComputationType.SEND_F, (0,), 6, None, None, None, None, ["audio"]),
        _Action(0, 0, 2, _ComputationType.FORWARD, (1), None, None, None, None, None, ["audio"]),
        _Action(0, 0, 3, _ComputationType.SEND_F, (1,), 6, None, None, None, None, ["audio"]),
        _Action(0, 0, 4, _ComputationType.FORWARD, (2), None, None, None, None, None, ["audio"]),
        _Action(0, 0, 5, _ComputationType.SEND_F, (2,), 6, None, None, None, None, ["audio"]),
        _Action(0, 0, 6, _ComputationType.FORWARD, (3), None, None, None, None, None, ["audio"]),
        _Action(0, 0, 7, _ComputationType.SEND_F, (3,), 6, None, None, None, None, ["audio"]),

        _Action(0, 0, 8, _ComputationType.RECV_B, (0,), 6, None, None, None, None, ["audio"]),
        _Action(0, 0, 9, _ComputationType.FULL_BACKWARD, (0), None, None, None, None, None, ["audio"]),
        _Action(0, 0, 10, _ComputationType.RECV_B, (1,), 6, None, None, None, None, ["audio"]),
        _Action(0, 0, 11, _ComputationType.FULL_BACKWARD, (1), None, None, None, None, None, ["audio"]),
        _Action(0, 0, 12, _ComputationType.RECV_B, (2,), 6, None, None, None, None, ["audio"]),
        _Action(0, 0, 13, _ComputationType.FULL_BACKWARD, (2), None, None, None, None, None, ["audio"]),
        _Action(0, 0, 14, _ComputationType.RECV_B, (3,), 6, None, None, None, None, ["audio"]),
        _Action(0, 0, 15, _ComputationType.FULL_BACKWARD, (3), None, None, None, None, None, ["audio"]),
    ]
    
    rank6_actions = [
        _Action(0, 6, 0, _ComputationType.RECV_F, (0,), 0, None, None, None, None, ["audio"]),
        _Action(0, 6, 1, _ComputationType.FORWARD, (0), None, None, None, None, None, ["audio"]),
        _Action(0, 6, 2, _ComputationType.SEND_F, (0,), 3, None, None, None, None, ["audio"]),
    
        _Action(0, 6, 3, _ComputationType.RECV_F, (1,), 0, None, None, None, None, ["audio"]),
        _Action(0, 6, 4, _ComputationType.FORWARD, (1), None, None, None, None, None, ["audio"]),
        _Action(0, 6, 5, _ComputationType.SEND_F, (1,), 3, None, None, None, None, ["audio"]),
    
        _Action(0, 6, 6, _ComputationType.RECV_F, (2,), 0, None, None, None, None, ["audio"]),
        _Action(0, 6, 7, _ComputationType.FORWARD, (2), None, None, None, None, None, ["audio"]),
        _Action(0, 6, 8, _ComputationType.SEND_F, (2,), 3, None, None, None, None, ["audio"]),
    
        _Action(0, 6, 9, _ComputationType.RECV_F, (3,), 0, None, None, None, None, ["audio"]),
        _Action(0, 6, 10, _ComputationType.FORWARD, (3), None, None, None, None, None, ["audio"]),
        _Action(0, 6, 11, _ComputationType.SEND_F, (3,), 3, None, None, None, None, ["audio"]),
        
        _Action(0, 6, 12, _ComputationType.RECV_B, (0,), 3, None, None, None, None, ["audio"]),
        _Action(0, 6, 13, _ComputationType.FULL_BACKWARD, (0), None, None, None, None, None, ["audio"]),
        _Action(0, 6, 14, _ComputationType.SEND_B, (0,), 0, None, None, None, None, ["audio"]),
    
        _Action(0, 6, 15, _ComputationType.RECV_B, (1,), 3, None, None, None, None, ["audio"]),
        _Action(0, 6, 16, _ComputationType.FULL_BACKWARD, (1), None, None, None, None, None, ["audio"]),
        _Action(0, 6, 17, _ComputationType.SEND_B, (1,), 0, None, None, None, None, ["audio"]),
    
        _Action(0, 6, 18, _ComputationType.RECV_B, (2,), 3, None, None, None, None, ["audio"]),
        _Action(0, 6, 19, _ComputationType.FULL_BACKWARD, (2), None, None, None, None, None, ["audio"]),
        _Action(0, 6, 20, _ComputationType.SEND_B, (2,), 0, None, None, None, None, ["audio"]),
    
        _Action(0, 6, 21, _ComputationType.RECV_B, (3,), 3, None, None, None, None, ["audio"]),
        _Action(0, 6, 22, _ComputationType.FULL_BACKWARD, (3), None, None, None, None, None, ["audio"]),
        _Action(0, 6, 23, _ComputationType.SEND_B, (3,), 0, None, None, None, None, ["audio"]),
    ]

    rank1_actions = [
        _Action(0, 1, 0, _ComputationType.FORWARD, (0), None, None, None, None, None, ["vision"]),
        _Action(0, 1, 1, _ComputationType.SEND_F, (0,), 7, None, None, None, None, ["vision"]),
        _Action(0, 1, 2, _ComputationType.FORWARD, (1), None, None, None, None, None, ["vision"]),
        _Action(0, 1, 3, _ComputationType.SEND_F, (1,), 7, None, None, None, None, ["vision"]),
        _Action(0, 1, 4, _ComputationType.FORWARD, (2), None, None, None, None, None, ["vision"]),
        _Action(0, 1, 5, _ComputationType.SEND_F, (2,), 7, None, None, None, None, ["vision"]),
        _Action(0, 1, 6, _ComputationType.FORWARD, (3), None, None, None, None, None, ["vision"]),
        _Action(0, 1, 7, _ComputationType.SEND_F, (3,), 7, None, None, None, None, ["vision"]),

        _Action(0, 1, 8, _ComputationType.RECV_B, (0,), 7, None, None, None, None, ["vision"]),
        _Action(0, 1, 9, _ComputationType.FULL_BACKWARD, (0), None, None, None, None, None, ["vision"]),
        _Action(0, 1, 10, _ComputationType.RECV_B, (1,), 7, None, None, None, None, ["vision"]),
        _Action(0, 1, 11, _ComputationType.FULL_BACKWARD, (1), None, None, None, None, None, ["vision"]),
        _Action(0, 1, 12, _ComputationType.RECV_B, (2,), 7, None, None, None, None, ["vision"]),
        _Action(0, 1, 13, _ComputationType.FULL_BACKWARD, (2), None, None, None, None, None, ["vision"]),
        _Action(0, 1, 14, _ComputationType.RECV_B, (3,), 7, None, None, None, None, ["vision"]),
        _Action(0, 1, 15, _ComputationType.FULL_BACKWARD, (3), None, None, None, None, None, ["vision"]),
    ]
    
    rank7_actions = [
        _Action(0, 7, 0, _ComputationType.RECV_F, (0,), 1, None, None, None, None, ["vision"]),
        _Action(0, 7, 1, _ComputationType.FORWARD, (0), None, None, None, None, None, ["vision"]),
        _Action(0, 7, 2, _ComputationType.SEND_F, (0,), 3, None, None, None, None, ["vision"]),
    
        _Action(0, 7, 3, _ComputationType.RECV_F, (1,), 1, None, None, None, None, ["vision"]),
        _Action(0, 7, 4, _ComputationType.FORWARD, (1), None, None, None, None, None, ["vision"]),
        _Action(0, 7, 5, _ComputationType.SEND_F, (1,), 3, None, None, None, None, ["vision"]),
    
        _Action(0, 7, 6, _ComputationType.RECV_F, (2,), 1, None, None, None, None, ["vision"]),
        _Action(0, 7, 7, _ComputationType.FORWARD, (2), None, None, None, None, None, ["vision"]),
        _Action(0, 7, 8, _ComputationType.SEND_F, (2,), 3, None, None, None, None, ["vision"]),
    
        _Action(0, 7, 9, _ComputationType.RECV_F, (3,), 1, None, None, None, None, ["vision"]),
        _Action(0, 7, 10, _ComputationType.FORWARD, (3), None, None, None, None, None, ["vision"]),
        _Action(0, 7, 11, _ComputationType.SEND_F, (3,), 3, None, None, None, None, ["vision"]),
        
        _Action(0, 7, 12, _ComputationType.RECV_B, (0,), 3, None, None, None, None, ["vision"]),
        _Action(0, 7, 13, _ComputationType.FULL_BACKWARD, (0), None, None, None, None, None, ["vision"]),
        _Action(0, 7, 14, _ComputationType.SEND_B, (0,), 1, None, None, None, None, ["vision"]),
    
        _Action(0, 7, 15, _ComputationType.RECV_B, (1,), 3, None, None, None, None, ["vision"]),
        _Action(0, 7, 16, _ComputationType.FULL_BACKWARD, (1), None, None, None, None, None, ["vision"]),
        _Action(0, 7, 17, _ComputationType.SEND_B, (1,), 1, None, None, None, None, ["vision"]),
    
        _Action(0, 7, 18, _ComputationType.RECV_B, (2,), 3, None, None, None, None, ["vision"]),
        _Action(0, 7, 19, _ComputationType.FULL_BACKWARD, (2), None, None, None, None, None, ["vision"]),
        _Action(0, 7, 20, _ComputationType.SEND_B, (2,), 1, None, None, None, None, ["vision"]),
    
        _Action(0, 7, 21, _ComputationType.RECV_B, (3,), 3, None, None, None, None, ["vision"]),
        _Action(0, 7, 22, _ComputationType.FULL_BACKWARD, (3), None, None, None, None, None, ["vision"]),
        _Action(0, 7, 23, _ComputationType.SEND_B, (3,), 1, None, None, None, None, ["vision"]),
    ]
    
    rank2_actions = [
        _Action(0, 2, 0, _ComputationType.FORWARD, (0), None, None, None, None, None, ["text"]),
        _Action(0, 2, 1, _ComputationType.SEND_F, (0,), 8, None, None, None, None, ["text"]),
        _Action(0, 2, 2, _ComputationType.FORWARD, (1), None, None, None, None, None, ["text"]),
        _Action(0, 2, 3, _ComputationType.SEND_F, (1,), 8, None, None, None, None, ["text"]),
        _Action(0, 2, 4, _ComputationType.FORWARD, (2), None, None, None, None, None, ["text"]),
        _Action(0, 2, 5, _ComputationType.SEND_F, (2,), 8, None, None, None, None, ["text"]),
        _Action(0, 2, 6, _ComputationType.FORWARD, (3), None, None, None, None, None, ["text"]),
        _Action(0, 2, 7, _ComputationType.SEND_F, (3,), 8, None, None, None, None, ["text"]),
        
        _Action(0, 2, 8, _ComputationType.RECV_B, (0,), 8, None, None, None, None, ["text"]),
        _Action(0, 2, 9, _ComputationType.FULL_BACKWARD, (0), None, None, None, None, None, ["text"]),
        _Action(0, 2, 10, _ComputationType.RECV_B, (1,), 8, None, None, None, None, ["text"]),
        _Action(0, 2, 11, _ComputationType.FULL_BACKWARD, (1), None, None, None, None, None, ["text"]),
        _Action(0, 2, 12, _ComputationType.RECV_B, (2,), 8, None, None, None, None, ["text"]),
        _Action(0, 2, 13, _ComputationType.FULL_BACKWARD, (2), None, None, None, None, None, ["text"]),
        _Action(0, 2, 14, _ComputationType.RECV_B, (3,), 8, None, None, None, None, ["text"]),
        _Action(0, 2, 15, _ComputationType.FULL_BACKWARD, (3), None, None, None, None, None, ["text"]),
    ]
    
    rank8_actions = [
        _Action(0, 8, 0, _ComputationType.RECV_F, (0,), 2, None, None, None, None, ["text"]),
        _Action(0, 8, 1, _ComputationType.FORWARD, (0), None, None, None, None, None, ["text"]),
        _Action(0, 8, 2, _ComputationType.SEND_F, (0,), 3, None, None, None, None, ["text"]),
    
        _Action(0, 8, 3, _ComputationType.RECV_F, (1,), 2, None, None, None, None, ["text"]),
        _Action(0, 8, 4, _ComputationType.FORWARD, (1), None, None, None, None, None, ["text"]),
        _Action(0, 8, 5, _ComputationType.SEND_F, (1,), 3, None, None, None, None, ["text"]),
    
        _Action(0, 8, 6, _ComputationType.RECV_F, (2,), 2, None, None, None, None, ["text"]),
        _Action(0, 8, 7, _ComputationType.FORWARD, (2), None, None, None, None, None, ["text"]),
        _Action(0, 8, 8, _ComputationType.SEND_F, (2,), 3, None, None, None, None, ["text"]),
    
        _Action(0, 8, 9, _ComputationType.RECV_F, (3,), 2, None, None, None, None, ["text"]),
        _Action(0, 8, 10, _ComputationType.FORWARD, (3), None, None, None, None, None, ["text"]),
        _Action(0, 8, 11, _ComputationType.SEND_F, (3,), 3, None, None, None, None, ["text"]),
        
        _Action(0, 8, 12, _ComputationType.RECV_B, (0,), 3, None, None, None, None, ["text"]),
        _Action(0, 8, 13, _ComputationType.FULL_BACKWARD, (0), None, None, None, None, None, ["text"]),
        _Action(0, 8, 14, _ComputationType.SEND_B, (0,), 2, None, None, None, None, ["text"]),
    
        _Action(0, 8, 15, _ComputationType.RECV_B, (1,), 3, None, None, None, None, ["text"]),
        _Action(0, 8, 16, _ComputationType.FULL_BACKWARD, (1), None, None, None, None, None, ["text"]),
        _Action(0, 8, 17, _ComputationType.SEND_B, (1,), 2, None, None, None, None, ["text"]),
    
        _Action(0, 8, 18, _ComputationType.RECV_B, (2,), 3, None, None, None, None, ["text"]),
        _Action(0, 8, 19, _ComputationType.FULL_BACKWARD, (2), None, None, None, None, None, ["text"]),
        _Action(0, 8, 20, _ComputationType.SEND_B, (2,), 2, None, None, None, None, ["text"]),
    
        _Action(0, 8, 21, _ComputationType.RECV_B, (3,), 3, None, None, None, None, ["text"]),
        _Action(0, 8, 22, _ComputationType.FULL_BACKWARD, (3), None, None, None, None, None, ["text"]),
        _Action(0, 8, 23, _ComputationType.SEND_B, (3,), 2, None, None, None, None, ["text"]),
    ]
    
    rank3_actions = [
        _Action(1, 3, 0, _ComputationType.RECV_F, (0,), 6, None, None, None, None, ["audio"]),
        _Action(1, 3, 1, _ComputationType.RECV_F, (0,), 7, None, None, None, None, ["vision"]),
        _Action(1, 3, 2, _ComputationType.RECV_F, (0,), 8, None, None, None, None, ["text"]),
        _Action(1, 3, 3, _ComputationType.FORWARD, (0), None, None, None, None, None, ["audio","vision","text"]),
        _Action(1, 3, 4, _ComputationType.SEND_F, (0,), 4, None, None, None, None, None),

        _Action(1, 3, 5, _ComputationType.RECV_F, (1,), 6, None, None, None, None, ["audio"]),
        _Action(1, 3, 6, _ComputationType.RECV_F, (1,), 7, None, None, None, None, ["vision"]),
        _Action(1, 3, 7, _ComputationType.RECV_F, (1,), 8, None, None, None, None, ["text"]),
        _Action(1, 3, 8, _ComputationType.FORWARD, (1), None, None, None, None, None, ["audio","vision","text"]),
        _Action(1, 3, 9, _ComputationType.SEND_F, (1,), 4, None, None, None, None, None),

        _Action(1, 3, 10, _ComputationType.RECV_F, (2,), 6, None, None, None, None, ["audio"]),
        _Action(1, 3, 11, _ComputationType.RECV_F, (2,), 7, None, None, None, None, ["vision"]),
        _Action(1, 3, 12, _ComputationType.RECV_F, (2,), 8, None, None, None, None, ["text"]),
        _Action(1, 3, 13, _ComputationType.FORWARD, (2), None, None, None, None, None, ["audio","vision","text"]),
        _Action(1, 3, 14, _ComputationType.SEND_F, (2,), 4, None, None, None, None, None),

        _Action(1, 3, 15, _ComputationType.RECV_F, (3,), 6, None, None, None, None, ["audio"]),
        _Action(1, 3, 16, _ComputationType.RECV_F, (3,), 7, None, None, None, None, ["vision"]),
        _Action(1, 3, 17, _ComputationType.RECV_F, (3,), 8, None, None, None, None, ["text"]),
        _Action(1, 3, 18, _ComputationType.FORWARD, (3), None, None, None, None, None, ["audio","vision","text"]),
        _Action(1, 3, 19, _ComputationType.SEND_F, (3,), 4, None, None, None, None, None),
        
        
        
        
        _Action(1, 3, 20, _ComputationType.RECV_B, (0,), 4, None, None, None, None, None),
        _Action(1, 3, 21, _ComputationType.FULL_BACKWARD, (0), None, None, None, None, None, ["audio","vision","text"]),
        _Action(1, 3, 22, _ComputationType.SEND_B, (0,), 6, None, None, None, None, ["audio"]),
        _Action(1, 3, 23, _ComputationType.SEND_B, (0,), 7, None, None, None, None, ["vision"]),
        _Action(1, 3, 24, _ComputationType.SEND_B, (0,), 8, None, None, None, None, ["text"]),
        
        _Action(1, 3, 25, _ComputationType.RECV_B, (1,), 4, None, None, None, None, None),
        _Action(1, 3, 26, _ComputationType.FULL_BACKWARD, (1), None, None, None, None, None, ["audio","vision","text"]),
        _Action(1, 3, 27, _ComputationType.SEND_B, (1,), 6, None, None, None, None, ["audio"]),
        _Action(1, 3, 28, _ComputationType.SEND_B, (1,), 7, None, None, None, None, ["vision"]),
        _Action(1, 3, 29, _ComputationType.SEND_B, (1,), 8, None, None, None, None, ["text"]),
        
        _Action(1, 3, 30, _ComputationType.RECV_B, (2,), 4, None, None, None, None, None),
        _Action(1, 3, 31, _ComputationType.FULL_BACKWARD, (2), None, None, None, None, None, ["audio","vision","text"]),
        _Action(1, 3, 32, _ComputationType.SEND_B, (2,), 6, None, None, None, None, ["audio"]),
        _Action(1, 3, 33, _ComputationType.SEND_B, (2,), 7, None, None, None, None, ["vision"]),
        _Action(1, 3, 34, _ComputationType.SEND_B, (2,), 8, None, None, None, None, ["text"]),
        
        _Action(1, 3, 35, _ComputationType.RECV_B, (3,), 4, None, None, None, None, None),
        _Action(1, 3, 36, _ComputationType.FULL_BACKWARD, (3), None, None, None, None, None, ["audio","vision","text"]),
        _Action(1, 3, 37, _ComputationType.SEND_B, (3,), 6, None, None, None, None, ["audio"]),
        _Action(1, 3, 38, _ComputationType.SEND_B, (3,), 7, None, None, None, None, ["vision"]),
        _Action(1, 3, 39, _ComputationType.SEND_B, (3,), 8, None, None, None, None, ["text"]),
    ]
    
    rank4_actions = [
        _Action(2, 4, 0, _ComputationType.RECV_F, (0,), 3, None, None, None, None, None),
        _Action(2, 4, 1, _ComputationType.FORWARD, (0), None, None, None, None, None, None),
        _Action(2, 4, 2, _ComputationType.SEND_F, (0,), 5, None, None, None, None, None),

        _Action(2, 4, 3, _ComputationType.RECV_F, (1,), 3, None, None, None, None, None),
        _Action(2, 4, 4, _ComputationType.FORWARD, (1), None, None, None, None, None, None),
        _Action(2, 4, 5, _ComputationType.SEND_F, (1,), 5, None, None, None, None, None),

        _Action(2, 4, 6, _ComputationType.RECV_F, (2,), 3, None, None, None, None, None),
        _Action(2, 4, 7, _ComputationType.FORWARD, (2), None, None, None, None, None, None),
        _Action(2, 4, 8, _ComputationType.SEND_F, (2,), 5, None, None, None, None, None),

        _Action(2, 4, 9, _ComputationType.RECV_F, (3,), 3, None, None, None, None, None),
        _Action(2, 4, 10, _ComputationType.FORWARD, (3), None, None, None, None, None, None),
        _Action(2, 4, 11, _ComputationType.SEND_F, (3,), 5, None, None, None, None, None),




        _Action(2, 4, 12, _ComputationType.RECV_B, (0,), 5, None, None, None, None, None),
        _Action(2, 4, 13, _ComputationType.FULL_BACKWARD, (0), None, None, None, None, None, None),
        _Action(2, 4, 14, _ComputationType.SEND_B, (0,), 3, None, None, None, None, None),

        _Action(2, 4, 15, _ComputationType.RECV_B, (1,), 5, None, None, None, None, None),
        _Action(2, 4, 16, _ComputationType.FULL_BACKWARD, (1), None, None, None, None, None, None),
        _Action(2, 4, 17, _ComputationType.SEND_B, (1,), 3, None, None, None, None, None),

        _Action(2, 4, 18, _ComputationType.RECV_B, (2,), 5, None, None, None, None, None),
        _Action(2, 4, 19, _ComputationType.FULL_BACKWARD, (2), None, None, None, None, None, None),
        _Action(2, 4, 20, _ComputationType.SEND_B, (2,), 3, None, None, None, None, None),

        _Action(2, 4, 21, _ComputationType.RECV_B, (3,), 5, None, None, None, None, None),
        _Action(2, 4, 22, _ComputationType.FULL_BACKWARD, (3), None, None, None, None, None, None),
        _Action(2, 4, 23, _ComputationType.SEND_B, (3,), 3, None, None, None, None, None),
    ]
    
    rank5_actions = [
        _Action(3, 5, 0, _ComputationType.RECV_F, (0,), 4, None, None, None, None, None),
        _Action(3, 5, 1, _ComputationType.FORWARD, (0), None, None, None, None, None, None),

        _Action(3, 5, 2, _ComputationType.RECV_F, (1,), 4, None, None, None, None, None),
        _Action(3, 5, 3, _ComputationType.FORWARD, (1), None, None, None, None, None, None),

        _Action(3, 5, 4, _ComputationType.RECV_F, (2,), 4, None, None, None, None, None),
        _Action(3, 5, 5, _ComputationType.FORWARD, (2), None, None, None, None, None, None),

        _Action(3, 5, 6, _ComputationType.RECV_F, (3,), 4, None, None, None, None, None),
        _Action(3, 5, 7, _ComputationType.FORWARD, (3), None, None, None, None, None, None),

        _Action(3, 5, 8, _ComputationType.FULL_BACKWARD, (0), None, None, None, None, None, None),
        _Action(3, 5, 9, _ComputationType.SEND_B, (0,), 4, None, None, None, None, None),
        
        _Action(3, 5, 10, _ComputationType.FULL_BACKWARD, (1), None, None, None, None, None, None),
        _Action(3, 5, 11, _ComputationType.SEND_B, (1,), 4, None, None, None, None, None),

        _Action(3, 5, 12, _ComputationType.FULL_BACKWARD, (2), None, None, None, None, None, None),
        _Action(3, 5, 13, _ComputationType.SEND_B, (2,), 4, None, None, None, None, None),

        _Action(3, 5, 14, _ComputationType.FULL_BACKWARD, (3), None, None, None, None, None, None),
        _Action(3, 5, 15, _ComputationType.SEND_B, (3,), 4, None, None, None, None, None),
    ]


    return {0: rank0_actions, 1: rank1_actions, 2: rank2_actions, 3: rank3_actions, 4: rank4_actions, 5: rank5_actions, 6: rank6_actions, 7: rank7_actions, 8: rank8_actions}

parser = argparse.ArgumentParser()
parser.add_argument("--train_steps", type=int, default=1,
                    help="The total number of steps for training. If omitted, run the entire DataLoader.")
parser.add_argument("--batch_size", type=int,
                    default=int(os.getenv("BATCH_SIZE", 20)),
                    help="The batch size of each rank (the environment variable BATCH_SIZE can be overridden)")
parser.add_argument("--microbatch_num", type=int,
                    default=int(os.getenv("MICROBATCH_NUM", 5)),
                    help="Micro-batch number (the environment variable MICROBATCH_NUM can be overridden)")
parser.add_argument("--sudo_pass", default=os.getenv("SUDO_PASS"),
                    help='Write the password of root')
parser.add_argument("--upstream", default=os.getenv("upstream"),
                    help='Write the upstream in mbps')
parser.add_argument("--plan_loc", type=str, required=True,
                    help='the json file that stores the sharding plans...')
args = parser.parse_args()
def main():
    import torch  # ensure local binding to avoid UnboundLocalError in some environments
    dist.init_process_group("gloo", init_method="env://")

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])

    device = torch.device("cpu")     

    MODEL_ID = "Qwen/Qwen2.5-Omni-3B"
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    thinker = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(MODEL_ID, trust_remote_code=True)
    if hasattr(thinker, "audio_tower") and hasattr(thinker.audio_tower, "config"):
        try:
            thinker.audio_tower.config.max_source_positions = 1000
        except Exception as e:
            print("[warn] cannot set max_source_positions:", e)
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    
    # Cache for one-shot local video pack
    video_pack_cache = None
    # 便捷访问
    text_model   = thinker.model           # 纯解码器主干（有 embed_tokens / layers / norm）
    audio_enc    = getattr(thinker, "audio_tower", None)
    vision_enc   = getattr(thinker, "visual", None)
    rotary_emb   = getattr(text_model, "rotary_emb", None)
    vocab_size   = tok.vocab_size
    if rank == 0 and vision_enc is not None and getattr(vision_enc, "config", None) is not None:
        cfg = vision_enc.config
        try:
            keys = [k for k in dir(cfg) if not k.startswith("_") and not callable(getattr(cfg, k))]
            summary = {k: getattr(cfg, k) for k in keys if isinstance(getattr(cfg, k), (int, float, tuple, list))}
            print(f"[rank0] vision config summary: {summary}")
        except Exception:
            pass

    # 自动切分点
    L  = len(text_model.layers)
    L1 = L // 3
    L2 = (2 * L) // 3
    
    if rank == 0:
        stage_mod = AudioFrontAndTwoLayers(audio_enc)
        stage_mod.to(device)
        stage = PipelineStage_Multimodality(stage_mod, stage_index=0,
                            num_stages=world, device=device,
                            group=dist.group.WORLD,
                            prev_group=None, this_group=[0], next_group=[6],
                            model_type = "audio",
                            mm_prev_groups = None)
        setattr(stage, "modal_type", "audio")
    elif rank == 6:
        stage_mod = AudioEncoderMidFive(audio_enc)
        stage_mod.to(device)
        stage = PipelineStage_with_mutiple_ranks(stage_mod, stage_index=0,
                            num_stages=world, device=device,
                            group=dist.group.WORLD,
                            prev_group=[0], this_group=[6], next_group=[3])
        setattr(stage, "modal_type", "audio")
    elif rank == 1:
        stage_mod = VisionFrontAndTwoLayers(vision_enc)
        stage_mod.to(device)
        stage = PipelineStage_Multimodality(stage_mod, stage_index=0,
                            num_stages=world, device=device,
                            group=dist.group.WORLD,
                            prev_group=None, this_group=[1], next_group=[7],
                            model_type = "vision",
                            mm_prev_groups = None)
        setattr(stage, "modal_type", "vision")
    elif rank == 7:
        stage_mod = VisionEncoderMidRest(vision_enc)
        stage_mod.to(device)
        stage = PipelineStage_with_mutiple_ranks(stage_mod, stage_index=0,
                            num_stages=world, device=device,
                            group=dist.group.WORLD,
                            prev_group=[1], this_group=[7], next_group=[3])
        setattr(stage, "modal_type", "vision")
    elif rank == 2:
        stage_mod = TextAndVideoFrontAndTwoLayers(text_model,VisionFrontAndTwoLayers(vision_enc))
        stage_mod.to(device)
        stage = PipelineStage_Multimodality(stage_mod, stage_index=0,
                            num_stages=world, device=device,
                            group=dist.group.WORLD,
                            prev_group=None, this_group=[2], next_group=[8],
                            model_type = "text",
                            mm_prev_groups = None)
        setattr(stage, "modal_type", "text")
        
    elif rank == 8:
        stage_mod = TextAndVideoEncoderMidRest(VisionEncoderMidRest(vision_enc))
        stage_mod.to(device)
        stage = PipelineStage_with_mutiple_ranks(stage_mod, stage_index=0,
                            num_stages=world, device=device,
                            group=dist.group.WORLD,
                            prev_group=[2], this_group=[8], next_group=[3])
        setattr(stage, "modal_type", "text")
        
    elif rank == 3:
        stage_mod = Stage1(text_model, L1)
        stage_mod.to(device)
        stage = PipelineStage_Multimodality(stage_mod, stage_index=1,
                            num_stages=world, device=device,
                            group=dist.group.WORLD,
                            prev_group=[6,7,8], this_group=[3], next_group=[4],
                            model_type = "packing",
                            mm_prev_groups = {"audio":[6],"vision":[7],"text":[8]})
        setattr(stage, "modal_type", "packing")
    elif rank == 4:
        stage_mod = Stage2(text_model, L1, L2)
        stage_mod.to(device)
        stage = PipelineStage_with_mutiple_ranks(stage_mod, stage_index=2,
                            num_stages=world, device=device,
                            group=dist.group.WORLD,
                            prev_group=[3], this_group=[4], next_group=[5])
    elif rank == 5:
        stage_mod = Stage3(thinker, text_model, L2)
        stage_mod.to(device)
        stage = PipelineStage_with_mutiple_ranks(stage_mod, stage_index=3,
                            num_stages=world, device=device,
                            group=dist.group.WORLD,
                            prev_group=[4], this_group=[5], next_group=None)
    
    del thinker                        
    import gc; gc.collect()

    raw = load_dataset("jxie/flickr8k", split="train")

    def pick_caption(example):
        text = example.get("caption_0", None)
        if text is None:
            caps = example.get("captions")
            text = caps[0] if isinstance(caps, list) and caps else ""
        return {"text": text}

    keep_cols = {"image", "text"}
    raw = raw.map(pick_caption, remove_columns=[c for c in raw.column_names if c not in keep_cols])

    from PIL import Image

    noise_path = os.path.join(os.getcwd(), "noise.mp3")

    def collate_fn(batch):
        conversations = []
        max_audio_frames = getattr(collate_fn, "_max_audio_frames", 5096)
        if not hasattr(collate_fn, "_audio_cache"):
            collate_fn._audio_cache = {}
        audio_cache = collate_fn._audio_cache
        audio_paths = []
        for ex in batch:
            img = ex["image"]
            txt = ex.get("text", "") if isinstance(ex.get("text", ""), str) else ""
            
            convo = []
            if USE_TTS_SYS:
                convo.append({
                    "role": "system",
                    "content": [{"type": "text", "text": DEFAULT_QWEN_OMNI_SYS}],
                })

               
            convo.append({
                "role": "user",
                "content": [
                    {"type": "audio", "audio": noise_path},
                    {"type": "image", "image": img},
                    {"type": "text",  "text": txt}
                ],
            })
            conversations.append(convo)
            audio_paths.append(noise_path)
            
            
        pack = proc.apply_chat_template(
            conversations,
            add_generation_prompt=USE_TTS_SYS,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            padding="max_length",
            max_length=512, #原先是512
            truncation=True
        )

        # Debug once or twice: what keys does processor return, and are there audio keys?
        try:
            if not hasattr(collate_fn, "_dbg_count"):
                collate_fn._dbg_count = 0
            if collate_fn._dbg_count < 2:
                def _shape(x):
                    if isinstance(x, torch.Tensor):
                        return (tuple(x.shape), str(x.dtype))
                    return type(x).__name__
                keys = list(pack.keys()) if hasattr(pack, 'keys') else []
                print(f"[rank0] collate_fn: processor pack keys: {keys}")
                for k in ("pixel_values", "image_grid_thw", "input_values", "input_features", "audio_values", "feature_attention_mask"):
                    if hasattr(pack, 'get'):
                        v = pack.get(k, None)
                        if v is not None:
                            print(f"[rank0] collate_fn: pack[{k}] -> {_shape(v)}")
                        else:
                            print(f"[rank0] collate_fn: pack[{k}] -> None")
                collate_fn._dbg_count += 1
        except Exception:
            pass

        input_ids = pack["input_ids"]
        attention_mask = pack["attention_mask"]
        labels = input_ids.clone()

        special_ids = set([
            tok.pad_token_id,
            getattr(tok, "eos_token_id", None),
            getattr(tok, "bos_token_id", None),
            getattr(cfg, "image_token_id", None),
            getattr(cfg, "video_token_id", None),
            getattr(cfg, "audio_token_id", None),
        ])
        special_ids = {i for i in special_ids if i is not None}
        for sid in special_ids:
            labels[labels == sid] = -100

        # Vision packing
        vision_inputs = None
        pixel_values = pack.get("pixel_values", None)
        image_grid_thw = pack.get("image_grid_thw", None)
        if image_grid_thw is not None:
            image_grid_thw = torch.as_tensor(image_grid_thw, dtype=torch.long)
        if pixel_values is not None and image_grid_thw is not None and image_grid_thw.numel() > 0:
            counts = (image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2])
            slices, off = [], 0
            for n in counts.tolist():
                slices.append(pixel_values[off: off + n])
                off += n
            assert off == pixel_values.size(0), f"visual tokens mismatch: {off} != {pixel_values.size(0)}"
            vision_inputs = {
                "pixel_values_list": slices,
                "grid_thw": image_grid_thw,
            }

        # Audio packing: re-extract features per sample, keep real lengths (仅此处做“只按最短截断”)
        audio_inputs = None
        reencoded_feats = []
        reencoded_masks = []
        reencode_success = True
        if audio_paths:
            for path in audio_paths:
                cached = audio_cache.get(path)
                current_mtime = None
                try:
                    current_mtime = os.path.getmtime(path)
                except OSError:
                    pass

                if cached is None or cached.get("mtime") != current_mtime:
                    try:
                        waveform, sr = torchaudio.load(path)
                    except Exception as e:
                        print(f"[collate_fn] failed to load audio {path}: {e}")
                        reencode_success = False
                        break

                    waveform = waveform.to(torch.float32)
                    if sr != 16000:
                        waveform = AF.resample(waveform, sr, 16000)
                    if waveform.dim() == 1:
                        waveform = waveform.unsqueeze(0)
                    if waveform.size(0) > 1:
                        waveform = waveform.mean(dim=0, keepdim=True)

                    wave = waveform.squeeze(0).contiguous().cpu()
                    audio_out = proc.feature_extractor(
                        wave.numpy(),
                        sampling_rate=16000,
                        padding=False,
                        return_attention_mask=True,
                    )

                    feats_np = audio_out.get("input_features")
                    mask_np = audio_out.get("attention_mask")
                    cached = {
                        "feats": feats_np,
                        "mask": mask_np,
                        "mtime": current_mtime,
                    }
                    audio_cache[path] = cached
                else:
                    feats_np = cached.get("feats")
                    mask_np = cached.get("mask")

                if feats_np is None:
                    reencode_success = False
                    break

                feats = torch.tensor(feats_np, dtype=torch.float32)
                if mask_np is None:
                    mask = torch.ones((feats.shape[-1],), dtype=torch.int32)
                else:
                    mask = torch.tensor(mask_np, dtype=torch.int32)

                if feats.dim() == 3:
                    feats = feats.squeeze(0)
                if mask.dim() == 2:
                    mask = mask.squeeze(0)

                T_feat = feats.shape[-1]
                T_mask = mask.shape[-1]
                T_target = min(T_feat, T_mask, max_audio_frames)
                if T_target <= 0:
                    reencode_success = False
                    break

                reencoded_feats.append(feats[:, :T_target])
                reencoded_masks.append(mask[:T_target])

        if reencode_success and reencoded_feats and len(reencoded_feats) == len(audio_paths):
            max_len = max(feat.shape[-1] for feat in reencoded_feats)
            n_mels = reencoded_feats[0].shape[0]
            batch_count = len(reencoded_feats)

            feat_batch = reencoded_feats[0].new_zeros((batch_count, n_mels, max_len))
            mask_batch = reencoded_masks[0].new_zeros((batch_count, max_len))

            for idx, (feat, mask) in enumerate(zip(reencoded_feats, reencoded_masks)):
                t = feat.shape[-1]
                feat_batch[idx, :, :t] = feat
                mask_batch[idx, :t] = mask

            pack["input_features"] = feat_batch
            pack["feature_attention_mask"] = mask_batch

            # 只按最短截断，避免再次扩张
            T_feat = pack["input_features"].shape[-1]
            T_mask = pack["feature_attention_mask"].shape[-1]
            T_target = min(T_feat, T_mask)
            pack["input_features"] = pack["input_features"][..., :T_target]
            pack["feature_attention_mask"] = pack["feature_attention_mask"][..., :T_target]

            audio_inputs = {
                "input_features": pack["input_features"],
                "feature_attention_mask": pack["feature_attention_mask"],
            }
        elif hasattr(pack, "get"):
            feats = pack.get("input_features", None)
            mask = pack.get("feature_attention_mask", None)
            if isinstance(feats, torch.Tensor) and isinstance(mask, torch.Tensor):
                T_target = min(feats.shape[-1], mask.shape[-1], max_audio_frames)
                pack["input_features"] = feats[..., :T_target]
                pack["feature_attention_mask"] = mask[..., :T_target]
                audio_inputs = {
                    "input_features": pack["input_features"],
                    "feature_attention_mask": pack["feature_attention_mask"],
                }
            else:
                audio_inputs = None

        # Debug once or twice: what did we build for audio_inputs?
        try:
            if not hasattr(collate_fn, "_dbg_audio_count"):
                collate_fn._dbg_audio_count = 0
            if collate_fn._dbg_audio_count < 2:
                if audio_inputs is None:
                    print(f"[rank0] collate_fn: built audio_inputs=None")
                else:
                    kv = {k: (tuple(v.shape), str(v.dtype)) if isinstance(v, torch.Tensor) else type(v).__name__
                          for k, v in audio_inputs.items()}
                    print(f"[rank0] collate_fn: built audio_inputs keys/shapes: {kv}")
                collate_fn._dbg_audio_count += 1
        except Exception:
            pass

        try:
            if audio_inputs is not None:
                if not hasattr(collate_fn, "_dbg_audio_eff"):
                    collate_fn._dbg_audio_eff = 0
                rank_id = dist.get_rank() if dist.is_initialized() else 0
                if collate_fn._dbg_audio_eff < 3 and rank_id == 0:
                    eff = audio_inputs["feature_attention_mask"].sum(dim=1).tolist()
                    print(
                        f"[rank{rank_id}] collate_fn[audio]: input_features {tuple(audio_inputs['input_features'].shape)} "
                        f"mask {tuple(audio_inputs['feature_attention_mask'].shape)} effective_frames={eff}"
                    )
                    collate_fn._dbg_audio_eff += 1
        except Exception:
            pass

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "vision_inputs": vision_inputs,
            "audio_inputs": audio_inputs,
        }


    batch_size = args.batch_size
    microbatch_num = args.microbatch_num
    block = 512

    if rank == 0:
        loader = torch.utils.data.DataLoader(
            raw,
            batch_size=batch_size,
            drop_last=True,
            collate_fn=collate_fn,
        )

    def loss_fn(output, target):
        if output is None or target is None:
            return None
        vocab_size = output.size(-1)
        T_logits = output.size(1)
        T_labels = target.size(1)
        T = min(T_logits, T_labels)
        logits = output[:, :T-1, :].reshape(-1, vocab_size)
        labels = target[:, 1:T].reshape(-1)
        valid_mask = (labels >= -100) & (labels < vocab_size)
        if not valid_mask.all():
            invalid_labels = labels[~valid_mask]
            print(f"[rank{dist.get_rank()}] WARNING: Found invalid labels: {invalid_labels[:10]}...")
        return F.cross_entropy(logits, labels, ignore_index=-100)

    sched = PipelineScheduleRuntimeWithDirection([stage], n_microbatches=microbatch_num,
                                                loss_fn=loss_fn, root_pass=args.sudo_pass)
    actions = create_pipeline_actions()
    sched._load_actions(actions, format="compute_comms")

    opt = optim.Adam(stage_mod.parameters(), lr=1e-4)
    prev_loss = None
    
    
    for epoch in range(1):
        if rank == 0:
            steps_tensor = torch.tensor(len(loader) if args.train_steps is None else args.train_steps, device=device)
            dist.broadcast(steps_tensor, src=0)
            data_iter = iter(loader)
            print(f"Total training steps: {steps_tensor.item()}")
        else:
            steps_tensor = torch.tensor(0, device=device)
            dist.broadcast(steps_tensor, src=0)

        total_steps = int(steps_tensor.item())

        if rank == 0:
            pbar = tqdm(
                total=int(total_steps),
                desc=f"Training Epoch {epoch+1}",
                unit="step",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            )
            start_time = time.time()

        for step in range(total_steps):
            step_start_time = time.time()
            opt.zero_grad(set_to_none=True)

            # Prepare per-head inputs on rank 0 then broadcast
            if rank == 0:
                batch = next(data_iter)
                inp_ids = batch["input_ids"].to(device)
                attn = batch["attention_mask"].to(device)
                vis_pack = batch.get("vision_inputs", None)
                aud_pack = batch.get("audio_inputs", None)
                tgt = batch["labels"].to(device)

                # Broadcast target tensor to all ranks
                dist.broadcast(tgt, src=0)

                # Broadcast text tensors (for other heads to use splitting if needed)
                dist.broadcast(inp_ids, src=0)
                dist.broadcast(attn, src=0)

                # Broadcast vision/audio packs as Python objects
                buf_vis = [vis_pack]

                # Build video pack from ./video.mp4 once and broadcast
                if video_pack_cache is None:
                    video_pack_cache = _load_video_pack(proc, "./video.mp4", vision_enc, max_tubelets=16)
                    video_pack_cache = _replicate_video_pack_for_microbatches(video_pack_cache, microbatch_num)
                    if video_pack_cache is None or video_pack_cache.get("pixel_values_videos") is None:
                        print("[rank0] warning: _load_video_pack returned empty pack")
                    if step < 2:
                        try:
                            pv = video_pack_cache["pixel_values_videos"]
                            vthw = video_pack_cache["video_grid_thw"]
                            print(f"[rank0] video pack: pv={tuple(pv.shape) if pv is not None else None}, grid={tuple(vthw.shape) if vthw is not None else None}")
                        except Exception:
                            print("[rank0] video pack prepared")
                buf_vid = [video_pack_cache]
                dist.broadcast_object_list(buf_vid, src=0)
                dist.broadcast_object_list(buf_vis, src=0)
                buf_aud = [aud_pack]
                dist.broadcast_object_list(buf_aud, src=0)
                if step < 2:
                    print(f"[rank0] broadcast video_pack keys={list(video_pack_cache.keys()) if isinstance(video_pack_cache, dict) else None}")

                # Debug first two steps: what will we broadcast for audio
                if step < 2:
                    if aud_pack is None:
                        print(f"[rank0] train-step {step}: aud_pack=None (no audio_inputs)")
                    else:
                        try:
                            kv = {k: (tuple(v.shape), str(v.dtype)) if isinstance(v, torch.Tensor) else type(v).__name__
                                  for k, v in aud_pack.items()}
                            print(f"[rank0] train-step {step}: aud_pack keys/shapes: {kv}")
                        except Exception:
                            print(f"[rank0] train-step {step}: aud_pack present but cannot summarize")

                # Local step for audio head：作为 kwargs 传入，并带上 attention_mask 以便 microbatch 大小推断
                sched.step(audio_inputs=aud_pack, attention_mask=attn, target=tgt)

            else:
                # Receive target
                tgt = torch.zeros(batch_size, block, dtype=torch.long, device=device)
                dist.broadcast(tgt, src=0)

                # Receive text tensors for rank 2 (text head) or keep but ignore on others
                inp_ids = torch.zeros(batch_size, block, dtype=torch.long, device=device)
                attn = torch.zeros(batch_size, block, dtype=torch.long, device=device)
                dist.broadcast(inp_ids, src=0)
                dist.broadcast(attn, src=0)

                # Receive video/vision/audio packs in the same order as rank 0 broadcasts
                buf_vid = [None]
                dist.broadcast_object_list(buf_vid, src=0)
                video_pack = buf_vid[0]
                buf_vis = [None]
                dist.broadcast_object_list(buf_vis, src=0)
                vis_pack = buf_vis[0]
                buf_aud = [None]
                dist.broadcast_object_list(buf_aud, src=0)
                aud_pack = buf_aud[0]
                # Debug first two steps: confirm reception on non-zero ranks
                if step < 2:
                    rk = rank
                    if aud_pack is None:
                        print(f"[rank{rk}] train-step {step}: received aud_pack=None")
                    else:
                        try:
                            kv = {k: (tuple(v.shape), str(v.dtype)) if isinstance(v, torch.Tensor) else type(v).__name__
                                  for k, v in aud_pack.items()}
                            print(f"[rank{rk}] train-step {step}: received aud_pack keys/shapes: {kv}")
                        except Exception:
                            print(f"[rank{rk}] train-step {step}: received aud_pack present but cannot summarize")
                    try:
                        if video_pack is None:
                            print(f"[rank{rk}] train-step {step}: received video pack=None")
                        else:
                            pv = video_pack.get("pixel_values_videos", None)
                            vthw = video_pack.get("video_grid_thw", None)
                            print(f"[rank{rk}] train-step {step}: received video pack pv={tuple(pv.shape) if pv is not None else None}, grid={tuple(vthw.shape) if vthw is not None else None}")
                    except Exception as e:
                        print(f"[rank{rk}] train-step {step}: video pack summarize failed {type(e).__name__}: {e}")

                if rank == 1:
                    # Vision head executes with vision inputs：作为 kwargs 传入，并带上 attention_mask 以便 microbatch 大小推断
                    sched.step(vision_inputs=vis_pack, attention_mask=attn, target=tgt)
                elif rank == 2:
                    # Text head executes with text inputs
                    sched.step(inp_ids, attention_mask=attn, target=tgt, video_inputs=video_pack)
                else:
                    # Packing and later stages only need target to drive schedule
                    sched.step(target=tgt)

            if (step + 1) % 50 == 0:
                try:
                    sched.timeline_rec.events.clear()
                except Exception:
                    pass

            opt.step()

            if rank == 0:
                step_time = time.time() - step_start_time
                tokens_processed = batch_size * block
                tokens_per_second = tokens_processed / step_time
                pbar.set_postfix({
                    'tokens/s': f'{tokens_per_second:.0f}',
                    'step_time': f'{step_time:.2f}s',
                    'lr': f'{opt.param_groups[0]["lr"]:.2e}'
                })
                pbar.update(1)

            cur_loss = getattr(sched, "last_step_loss", None)
            if cur_loss is not None and rank == 0:
                print(f"[rank0] step {step+1} loss {cur_loss:.4f}")
                prev_loss = cur_loss

            dist.barrier()

        if rank == 0:
            pbar.close()
            total_time = time.time() - start_time
            print(f"\nEpoch {epoch+1} completed in {total_time:.2f}s")
            print(f"Average speed: {total_steps / total_time:.2f} steps/s")

    # ===== Gather states and merge full model =====
    # Collect from each rank the submodule state_dict
    buf_audio = [stage_mod.state_dict()] if rank == 0 else [None]
    dist.broadcast_object_list(buf_audio, src=0)
    if rank == 0:
        audio_state = buf_audio[0]

    buf_vision = [stage_mod.state_dict()] if rank == 1 else [None]
    dist.broadcast_object_list(buf_vision, src=1)
    if rank == 0:
        vision_state = buf_vision[0]

    buf_text = [stage_mod.state_dict()] if rank == 2 else [None]
    dist.broadcast_object_list(buf_text, src=2)
    if rank == 0:
        text_state = buf_text[0]

    buf_s1 = [stage_mod.state_dict()] if rank == 3 else [None]
    dist.broadcast_object_list(buf_s1, src=3)
    if rank == 0:
        s1_state = buf_s1[0]

    buf_s2 = [stage_mod.state_dict()] if rank == 4 else [None]
    dist.broadcast_object_list(buf_s2, src=4)
    if rank == 0:
        s2_state = buf_s2[0]

    buf_s3 = [stage_mod.state_dict()] if rank == 5 else [None]
    dist.broadcast_object_list(buf_s3, src=5)
    if rank == 0:
        s3_state = buf_s3[0]

    if rank == 0:
        print("\nMerging and saving model (divide-head version)...")
        merged = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(MODEL_ID, trust_remote_code=True)
        merged_state = merged.state_dict()

        # 1) Text embed_tokens from rank2
        for k, v in text_state.items():
            if k.startswith("embed_tokens."):
                newk = "model.embed_tokens." + k[len("embed_tokens."):]
                merged_state[newk] = v

        # 2) Rotary from Stage1 (rank3) if present
        for k, v in s1_state.items():
            if k.startswith("rotary_emb."):
                newk = "model.rotary_emb." + k[len("rotary_emb."):]
                if newk in merged_state:
                    merged_state[newk] = v

        # 3) Audio tower from rank0
        for k, v in audio_state.items():
            if k.startswith("audio_enc."):
                newk = "audio_tower." + k[len("audio_enc."):]
                if newk in merged_state:
                    merged_state[newk] = v

        # 4) Vision tower from rank1
        for k, v in vision_state.items():
            if k.startswith("vision_enc."):
                newk = "visual." + k[len("vision_enc."):]
                if newk in merged_state:
                    merged_state[newk] = v

        # Helper to map layers
        def _map_layer_key(local_key: str, global_offset: int) -> str:
            parts = local_key.split(".")
            assert parts[0] == "layers", f"unexpected key {local_key}"
            li = int(parts[1]) + global_offset
            rest = ".".join(parts[2:])
            return f"model.layers.{li}.{rest}"

        # 5) Layers 0..L1-1 from Stage1
        for k, v in s1_state.items():
            if k.startswith("layers."):
                newk = _map_layer_key(k, 0)
                merged_state[newk] = v

        # 6) Layers L1..L2-1 from Stage2
        for k, v in s2_state.items():
            if k.startswith("layers."):
                newk = _map_layer_key(k, L1)
                merged_state[newk] = v

        # 7) Layers L2..end + norm + lm_head from Stage3
        for k, v in s3_state.items():
            if k.startswith("layers."):
                newk = _map_layer_key(k, L2)
                merged_state[newk] = v
            elif k == "norm.weight":
                merged_state["model.norm.weight"] = v
            elif k == "norm.bias":
                if "model.norm.bias" in merged_state:
                    merged_state["model.norm.bias"] = v
            elif k == "lm_head.weight":
                merged_state["lm_head.weight"] = v
            elif k == "lm_head.bias":
                if "lm_head.bias" in merged_state:
                    merged_state["lm_head.bias"] = v

        merged.load_state_dict(merged_state, strict=False)
        save_dir = "trained_qwen_pp_devide_head"
        merged.save_pretrained(save_dir)
        tok.save_pretrained(save_dir)
        print(f"Saved merged model to ./{save_dir}")

    dist.destroy_process_group()



if __name__ == "__main__":
    main()
