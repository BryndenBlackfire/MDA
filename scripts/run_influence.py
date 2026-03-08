# scripts/run_influence.py

import os
import argparse
import yaml
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformer_lens import HookedTransformer

from core.ekfac_blocks import EKFAC_QK_Head, EKFAC_QKVO_Head
from core.ekfac_fit import stage1A_accumulate_AS, stage1B_fit_lambda
from core.influence_phase2 import phase2_score_qkonly, phase2_score_qkvo
from probes import get_probe
from utils.gather import gather_heap_as_tensors
from data.loader import build_dataloaders


def run(cfg: dict, model: DDP, rank: int, world_size: int) -> None:
    is_main    = (rank == 0)
    device     = next(model.parameters()).device
    inner      = model.module if hasattr(model, "module") else model
    mode       = cfg["target"]["mode"]
    layer_idx  = cfg["target"]["layer"]
    head_idx   = cfg["target"]["head"]
    d_model    = inner.cfg.d_model
    d_head     = inner.cfg.d_head
    seq_length = cfg["data"]["seq_length"]
    dtype      = getattr(torch, cfg.get("dtype", "float32"))
    top_k      = cfg["output"]["top_k"]
    output_dir = cfg["output"]["dir"]
    ekfac_cfg  = cfg.get("ekfac", {})

    dl_ekfac, dl_influence = build_dataloaders(
        npy_path=cfg["data"]["npy_path"],
        num_train_samples=cfg["data"]["num_train_samples"],
        batch_size_ekfac=cfg["data"]["batch_size_ekfac"],
        batch_size_influence=cfg["data"]["batch_size_influence"],
        rank=rank,
        world_size=world_size,
        num_workers=cfg["data"].get("num_workers", 2),
        seq_length=cfg["data"]["seq_length"],
    )

    # Probe gradient
    if is_main:
        print(f"[{rank}/{world_size}] Computing probe gradient (mode={mode})")
    probe       = get_probe(cfg["probe"]["type"])()
    probe_grads = probe.compute_grad(model, cfg, mode)
    if is_main:
        print(f"[{rank}/{world_size}] Probe gradient ready.")

    # Stage 1A
    if is_main:
        print(f"[{rank}/{world_size}] Stage 1A: accumulating A/S")
    if mode == "qk":
        ekfac = EKFAC_QK_Head(
            d_model=d_model, d_head=d_head,
            damping=ekfac_cfg.get("damping", 1e-5),
            damping_alpha=ekfac_cfg.get("damping_alpha", 0.1),
        )
    else:
        ekfac = EKFAC_QKVO_Head(
            d_model=d_model, d_head=d_head,
            damping=ekfac_cfg.get("damping", 1e-5),
            damping_alpha=ekfac_cfg.get("damping_alpha", 0.1),
        )

    elapsed_1A = stage1A_accumulate_AS(
        model_ddp=model, dataloader=dl_ekfac, ekfac=ekfac,
        layer_idx=layer_idx, head_idx=head_idx, seq_length=seq_length,
        d_model=d_model, d_head=d_head, dtype=dtype, device=device, verbose=is_main,
    )
    if is_main:
        print(f"[{rank}/{world_size}] Stage 1A done ({elapsed_1A:.1f}s)")

    # Stage 1B
    if is_main:
        print(f"[{rank}/{world_size}] Stage 1B: fitting Lambda")
    elapsed_1B = stage1B_fit_lambda(
        model_ddp=model, dataloader=dl_ekfac, ekfac=ekfac,
        layer_idx=layer_idx, head_idx=head_idx, seq_length=seq_length,
        d_model=d_model, d_head=d_head, dtype=dtype, device=device, verbose=is_main,
    )
    if is_main:
        print(f"[{rank}/{world_size}] Stage 1B done ({elapsed_1B:.1f}s)")

    # p = H^{-1} v
    if mode == "qk":
        p_qk = ekfac.inverse_hvp(probe_grads[0])
    else:
        p_qk = ekfac.inverse_hvp(0, probe_grads[0])
        p_v  = ekfac.inverse_hvp(1, probe_grads[1])
        p_o  = ekfac.inverse_hvp(2, probe_grads[2])
    if is_main:
        print(f"[{rank}/{world_size}] p ready.")

    # Stage 2
    if is_main:
        print(f"[{rank}/{world_size}] Stage 2: scoring")
    if mode == "qk":
        pos_heap, neg_heap, n_samples, elapsed_2 = phase2_score_qkonly(
            model_ddp=model, dataloader=dl_influence, p_qk=p_qk,
            layer_idx=layer_idx, head_idx=head_idx, seq_length=seq_length,
            dtype=dtype, device=device, top_k=top_k, verbose=is_main,
        )
    else:
        pos_heap, neg_heap, n_samples, elapsed_2 = phase2_score_qkvo(
            model_ddp=model, dataloader=dl_influence,
            p_qk=p_qk, p_v=p_v, p_o=p_o,
            layer_idx=layer_idx, head_idx=head_idx, seq_length=seq_length,
            dtype=dtype, device=device, top_k=top_k, verbose=is_main,
        )
    if is_main:
        print(f"[{rank}/{world_size}] Stage 2 done ({elapsed_2:.1f}s, {n_samples} samples)")

    # Gather and save
    pos_all = gather_heap_as_tensors(pos_heap, world_size, device, rank)
    neg_all = gather_heap_as_tensors(neg_heap, world_size, device, rank)

    if is_main:
        pos_all = pos_all[pos_all[:, 0].argsort()[::-1]][:top_k]
        neg_all = neg_all[neg_all[:, 0].argsort()[::-1]][:top_k]
        os.makedirs(output_dir, exist_ok=True)
        np.save(f"{output_dir}/top_pos.npy", pos_all)
        np.save(f"{output_dir}/top_neg.npy", neg_all)
        print(f"[{rank}/{world_size}] Saved to {output_dir}/top_pos.npy, top_neg.npy")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    rank       = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HookedTransformer.from_pretrained_no_processing(
        cfg["model"]["path"],
        dtype=getattr(torch, cfg.get("dtype", "float32")),
    )
    model = model.to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank],find_unused_parameters=True)

    run(cfg, model, rank, world_size)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()





