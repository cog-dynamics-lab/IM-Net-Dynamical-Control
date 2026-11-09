# experiments/02_grad_diagnostics.py
"""
Gradient magnitude-dominance diagnostics (Section 5.1, Fig. 7).

Outputs under results_reproduction/:
    fig7_grad_imbalance_<dataset>.csv        (natural)
    fig7_grad_imbalance_<dataset>_sim.csv    (with --simulation)

Each CSV row = one batch: columns ['cos', 'rho'].

Run from repo root:
    python experiments/02_grad_diagnostics.py \
        --dataset yelp   --data_path ./data/yelp_processed_for_meta          --mode single
    python experiments/02_grad_diagnostics.py \
        --dataset amazon --data_path ./data/amazon_books_processedDataV3     --mode single
"""
import os
import sys
import argparse

# Allow `from data_loaderMeta import ...` when run from experiments/
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "core"))

import torch
import random
import numpy as np
import pandas as pd
import torch.nn.functional as F

from data_loaderMeta import DataProcessor
from backbone import LightGCN
from train_engine import IMNetTrainer


REPRO_DIR = os.environ.get("REPRO_DIR", "results_reproduction")
os.makedirs(REPRO_DIR, exist_ok=True)


def calc_cos(model, lm, la):
    """Return (cos_sim, rho) between main gradient and auxiliary gradient."""
    ps = [p for p in model.parameters() if p.requires_grad]
    gm = torch.autograd.grad(lm, ps, retain_graph=True, allow_unused=True)
    ga = torch.autograd.grad(la, ps, retain_graph=False, allow_unused=True)
    vm = torch.cat([
        g.detach().flatten() if g is not None else torch.zeros_like(p).flatten()
        for g, p in zip(gm, ps)
    ])
    va = torch.cat([
        g.detach().flatten() if g is not None else torch.zeros_like(p).flatten()
        for g, p in zip(ga, ps)
    ])
    cos = F.cosine_similarity(vm.unsqueeze(0), va.unsqueeze(0)).item()
    rho = va.norm().item() / (vm.norm().item() + 1e-8)
    return cos, rho


def main(dataset, data_path, mode, epochs, simulation, conflict_scale, output):
    # Fix seed for reproducibility
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    config = {
        'data_path': data_path, 'dataset': dataset,
        'batch_size': 2048, 'lr': 0.001, 'meta_lr': 1e-3,
        'n_layers': 3, 'wd': 1e-4, 'dropout': 0.0, 'embed_dim': 64,
        'model_name': 'LightGCN', 'mode': mode, 'epochs': epochs,
        'eval_freq': 1, 'grad_clip_norm': 1.0,
        'spectral_reg_weight': 0.05, 'use_spectral_reg': False,
        'meta_loss_weight': 0.0448, 'proactive_meta_loss_weight': 0.0579,
        'meta_warmup_epochs': 30, 'meta_update_freq': 50,
        'multi_step_lookahead': 1, 'proactive_hvp_eps': 5e-4,
        'hvp_eps': 5e-4, 'hvp_max_eps': 5e-4, 'proactive_hvp_max_eps': 5e-4,
        'adaptive_epsilon': True, 'disable_lookahead': False,
        'lookahead_switch_epoch': None, 'force_meta_epoch': 1,
        'reg_val_yelp': 5e-5, 'cl_temp_yelp': 0.2, 'cl_eps_yelp': 0.18,
        'proactive_loss_explosion_ratio': 12.12, 'force_meta_patience': 12,
        'meta_decrease_window': 4, 'meta_max_steps': 5,
        'fixed_meta_val_batch': False, 'fast_validation': True,
        'cl_lambda': 0.1, 'cl_temp': 0.2,
    }

    dc = {'data_path': data_path, 'dataset': dataset, 'batch_size': 2048}
    if simulation:
        dc['simulation_conflict'] = True
        config['simulation_conflict'] = True
        # FIX: original code wrote `confllict_scale` (typo), so the value
        # was never read by the trainer. Use the correct key.
        config['conflict_scale'] = conflict_scale

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Device: {device}")

    dp = DataProcessor(dc)
    model = LightGCN(
        num_users=dp.n_users, num_items=dp.n_items,
        embed_dim=64, n_layers=3,
    ).to(device)
    trainer = IMNetTrainer(dp, model, config, device)
    trainer.model.train()

    recs = []
    for ep in range(1, epochs + 1):
        for bidx, batch in enumerate(dp.train_loader):
            if bidx >= 50:
                break
            model.zero_grad(set_to_none=True)
            losses, _ = trainer._compute_losses_meta(batch)

            # Select auxiliary loss for the diagnostic.
            # Priority: cl_loss > aux_loss > aux_loss2, first non-zero wins.
            # (Matches the reference experiment that produced Fig.7.)
            la = losses[3] if losses[3].abs().item() > 1e-8 else (
                losses[1] if losses[1].abs().item() > 1e-8 else losses[2]
            )
            if la.abs().item() < 1e-8:
                continue
            try:
                c, r = calc_cos(model, losses[0], la)
                recs.append({'cos': c, 'rho': r})
            except Exception as e:
                # grad may be None if a branch has no params; skip batch
                print(f"[WARN] batch {bidx} skipped: {repr(e)}")

    df = pd.DataFrame(recs)
    if df.empty:
        print("[ERROR] No valid batch produced gradients; CSV not written.")
        return

    # Default output name follows the README convention.
    if output is None:
        suffix = "_sim" if simulation else ""
        output = os.path.join(REPRO_DIR,
                              f"fig7_grad_imbalance_{dataset}{suffix}.csv")
    else:
        output = os.path.join(REPRO_DIR, os.path.basename(output))

    df.to_csv(output, index=False, float_format="%.6f")
    print(f"frac_neg={(df['cos'] < 0).mean():.2%}, "
          f"cos_mean={df['cos'].mean():+.3f}, "
          f"rho_mean={df['rho'].mean():.3f}")
    print(f"saved: {output}")


if __name__ == '__main__':
    _REPO_ROOT = os.path.abspath(os.path.join(_HERE, ".."))

    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='yelp', choices=['yelp', 'amazon'])
    p.add_argument(
        '--data_path', default=None,
        help='Dataset directory. Defaults to <repo_root>/data/'
             'yelp_processed_for_meta (yelp) or amazon_books_processedDataV3 (amazon).'
    )
    p.add_argument('--mode', default='single')
    p.add_argument('--epochs', type=int, default=2,
                   help='Number of epochs; each epoch uses at most 50 batches.')
    p.add_argument('--simulation', action='store_true',
                   help='Enable simulated anti-parallel conflict.')
    p.add_argument('--conflict_scale', type=float, default=1.0,
                   help='Conflict scale for --simulation. Default 1.0 gives rho≈0.978 '
                        '(manuscript Fig.7); Fig.9 uses 1.0/2.0/5.0.')
    p.add_argument('--output', default=None,
                   help='Override output filename; written under results_reproduction/.')
    a = p.parse_args()

    if a.data_path is None:
        if a.dataset == 'yelp':
            a.data_path = os.path.join(_REPO_ROOT, "data", "yelp_processed_for_meta")
        else:
            a.data_path = os.path.join(_REPO_ROOT, "data", "amazon_books_processedDataV3")

    main(a.dataset, a.data_path, a.mode, a.epochs,
         a.simulation, a.conflict_scale, a.output)