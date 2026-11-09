"""
Code Availability: Model-agnostic Proactive meta-control resolves dynamical instability in multi-objective learning via spectral regularization
File: utils.py
Description:
    High-performance scientific utilities for spectral analysis of the loss
    landscape. Implements matrix-free Power Iteration for Hessian spectral
    radius estimation, supporting the verification of dynamical stability.
Date: 2026-02-14
"""

import torch
import numpy as np


def compute_hessian_spectral_radius(model, loss, num_iter=10, tolerance=1e-10,
                                    min_spectral_radius=1e-10, verbose=False,
                                    differentiable=False):
    """
    Quantifies the maximum eigenvalue (lambda_max) of the Hessian matrix.

    Mathematical Basis:
    Matrix-free Power Iteration via Hessian-Vector Product (HVP).
    This computes the 'sharpness' of the optimization manifold, enabling
    the Proactive Meta-Control (IM-Net) to regularize spectral instability.

    Args:
        model: The neural manifold encoder (e.g., LightGCN).
        loss: The scalar energy functional (Weighted Loss). Must be a tensor
              that still holds the computation graph.
        num_iter: Iterations for spectral convergence.
        tolerance: Numerical tolerance for normalization.
        min_spectral_radius: Lower bound returned to avoid zero spectral radius.
        verbose: If True, print HVP diagnostics.
        differentiable: If True, returns a differentiable 0-dim tensor that
            preserves gradient flow through the HVP. This enables spectral
            regularization of the neural network parameters, i.e. the
            gradient can backpropagate from λ_max to the model params.
            If False (default), returns a Python float and matches the
            original behavior EXACTLY for backward compatibility with
            logging / analysis call sites.

    Returns:
        float | torch.Tensor:
            - If `differentiable=False`: Python float (original behavior).
            - If `differentiable=True`: 0-dim tensor with gradient flow to
              `model.parameters()`.
    """
    # 1. Isolate the differentiable manifold
    params = [p for p in model.parameters() if p.requires_grad]
    if len(params) == 0:
        if differentiable:
            return torch.tensor(float(min_spectral_radius))
        return float(min_spectral_radius)

    # 2. Compute first-order gradient field
    # create_graph=True is essential for higher-order derivatives
    grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
    grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, params)]

    def flatten_tensors(tensors):
        return torch.cat([t.contiguous().view(-1) for t in tensors])

    flat_grads = flatten_tensors(grads)
    norm_grads = torch.norm(flat_grads)

    if verbose:
        print(f"[HVP] norm_grads={norm_grads.item():.6e}")

    # 3. Initialize curvature probing vector (v), detached so that it does
    #    not accumulate a graph across Power Iteration steps.
    v = torch.randn(flat_grads.size(), device=flat_grads.device, dtype=flat_grads.dtype)
    v = v / (torch.norm(v) + tolerance)
    v = v.detach()

    # 4. Iterative Spectral Refinement
    # NOTE: We deliberately do NOT break early when norm_hvp is small,
    # because in late-stage training the gradient magnitude is tiny and
    # an early break would return lambda_max ~ 0.
    best_spectral = 0.0
    for _ in range(num_iter):
        # Implicit HVP: H * v = ∇(∇L · v)
        gv_product = torch.sum(flat_grads * v)
        hvp_tensors = torch.autograd.grad(
            gv_product, params, retain_graph=True, allow_unused=True
        )
        hvp_tensors = [h if h is not None else torch.zeros_like(p)
                       for h, p in zip(hvp_tensors, params)]
        hvp = flatten_tensors(hvp_tensors)

        norm_hvp = torch.norm(hvp)
        if verbose:
            print(f"[HVP] iter={_}, norm_hvp={norm_hvp.item():.6e}")

        # Track the best (non-differentiable) scalar for the legacy return path.
        with torch.no_grad():
            current_spectral = torch.dot(v, hvp).item()
        if abs(current_spectral) > abs(best_spectral):
            best_spectral = current_spectral

        if norm_hvp < 1e-12:
            break

        # Update and re-detach the principal eigenvector.
        v = (hvp / (norm_hvp + tolerance)).detach()

    # 5. Final Rayleigh quotient: λ_max ≈ vᵀ H v
    #    Recompute flat_grads with a fresh graph for the final HVP.
    grads_final = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
    grads_final = [g if g is not None else torch.zeros_like(p)
                   for g, p in zip(grads_final, params)]
    flat_grads_final = flatten_tensors(grads_final)

    # The final HVP is the ONLY one that needs a graph when differentiable=True.
    # In the legacy path we skip `create_graph` to save memory.
    gv_product = torch.sum(flat_grads_final * v)
    hvp_tensors = torch.autograd.grad(
        gv_product, params, retain_graph=True,
        create_graph=differentiable,   # <-- key
        allow_unused=True,
    )
    hvp_tensors = [h if h is not None else torch.zeros_like(p)
                   for h, p in zip(hvp_tensors, params)]
    hvp = flatten_tensors(hvp_tensors)

    # λ_max ≈ vᵀ (H v). `v` is detached, so this is a differentiable
    # function of the model parameters through H only.
    final_spectral_tensor = torch.dot(v, hvp)

    # 6. Return path
    if differentiable:
        # Differentiable path: return a 0-dim tensor with gradient flow.
        result_tensor = final_spectral_tensor.abs()
        # Enforce lower bound without breaking differentiability.
        return result_tensor.clamp(min=float(min_spectral_radius))
    else:
        # Legacy path: identical behavior to the original function.
        final_spectral = final_spectral_tensor.item()
        if abs(final_spectral) > abs(best_spectral):
            best_spectral = final_spectral

        # Explicit memory reclamation (only on the non-differentiable path,
        # since the differentiable path needs to keep the graph alive).
        del flat_grads, flat_grads_final, hvp
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        result = abs(best_spectral)
        if result < min_spectral_radius:
            result = float(min_spectral_radius)
        return result


def fix_random_seed(seed=2026):
    """
    Ensures deterministic behavior across stochastic tensor operations.
    Critical for the 'Digital Twin' reproducibility protocol.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[*] Deterministic seed locked: {seed}")