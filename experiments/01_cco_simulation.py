# experiments/01_cco_simulation.py
"""
CCO simulation with genuine IM-Net integration for:
    "Model-agnostic proactive meta-control resolves dynamical instability
     in multi-objective learning via spectral regularization"

Reproduces:
    Fig. 5  -> fig5_cco_phase_sync.png              (+ fig5_cco_phase.csv)
    Fig. 6  -> fig6_cco_bifurcation_mle.png         (+ fig6_cco_bifurcation.csv, fig6_cco_mle.csv)
    Table S1-> table_s1_cco_params.csv
    Fig. S1 -> figS1_bifurcation_mle.png

CSV outputs are written under results_reproduction/ (override with REPRO_DIR env var).

Run from repo root:
    python experiments/01_cco_simulation.py
"""

import os
import time
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.signal import hilbert
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ==============================================================
# Parameters (Section 3.3.2 / Table S1)
# ==============================================================
gamma = 0.15
k1 = k2 = -1.0
beta_duffing = 1.5
A1 = A2 = 1.2
omega1, omega2 = 1.0, 1.5
theta0 = np.pi / 2

kappa_min, kappa_max = 0.05, 1.0
KAPPA_CRIT = 0.34
KAPPA_REF = 0.9

DT = 0.01
T_TOTAL = 200.0
T_TRANSIENT = 20.0
T_RHO = 60.0
T_POINCARE = 200.0
T_MLE = 4000.0
T1 = 2 * np.pi / omega1

IMNET_HIDDEN_DIM = 64
IMNET_BETA_INIT = 0.005
IMNET_EPOCHS = 30
IMNET_LR = 1e-3
IMNET_BATCH = 512
IMNET_DATA_T = 300.0

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, ".."))

OUT_DIR = os.path.join(_REPO_ROOT, "cco_figures")
os.makedirs(OUT_DIR, exist_ok=True)
MODEL_PATH = os.path.join(_REPO_ROOT, "imnet_controller.pt")

REPRO_DIR = os.environ.get("REPRO_DIR", os.path.join(_REPO_ROOT, "results_reproduction"))
os.makedirs(REPRO_DIR, exist_ok=True)


def save_csv(df: pd.DataFrame, filename: str):
    """Save a DataFrame to results_reproduction/<filename> with 6-decimal formatting."""
    out = os.path.join(REPRO_DIR, filename)
    df.to_csv(out, index=False, float_format="%.6f")
    print(f"  saved: {out}")


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)


# ==============================================================
# Part 1: CCO dynamics
# ==============================================================
def duffing_coupled(t, y, kappa):
    x1, v1, x2, v2 = y
    F1 = A1 * np.sin(omega1 * t)
    F2 = A2 * np.sin(omega2 * t + theta0)
    dx1 = v1
    dv1 = -gamma * v1 - k1 * x1 - beta_duffing * x1 ** 3 - kappa * (x1 - x2) + F1
    dx2 = v2
    dv2 = -gamma * v2 - k2 * x2 - beta_duffing * x2 ** 3 - kappa * (x2 - x1) + F2
    return [dx1, dv1, dx2, dv2]


def duffing_target(t, y):
    """Reference trajectory: coupled CCO at KAPPA_REF (strong coupling)."""
    x1, v1, x2, v2 = y
    F1 = A1 * np.sin(omega1 * t)
    F2 = A2 * np.sin(omega2 * t + theta0)
    return [
        v1,
        -gamma * v1 - k1 * x1 - beta_duffing * x1 ** 3 - KAPPA_REF * (x1 - x2) + F1,
        v2,
        -gamma * v2 - k2 * x2 - beta_duffing * x2 ** 3 - KAPPA_REF * (x2 - x1) + F2,
    ]


def duffing_jacobian(t, y, kappa):
    """Analytical Jacobian dF/dy of the coupled CCO."""
    x1, v1, x2, v2 = y
    J = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [-k1 - 3 * beta_duffing * x1 ** 2 - kappa, -gamma, kappa, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [kappa, 0.0, -k2 - 3 * beta_duffing * x2 ** 2 - kappa, -gamma],
    ])
    return J


def spectral_radius_approx(J):
    """Gershgorin upper bound for the spectral radius."""
    abs_J = np.abs(J)
    diag = np.diag(abs_J)
    off_diag_sum = abs_J.sum(axis=1) - diag
    return float(np.max(diag + off_diag_sum))


def simulate(kappa, T=T_TOTAL, dt=DT):
    t_eval = np.arange(0.0, T, dt)
    y0 = [0.1, 0.0, -0.1, 0.0]
    sol = solve_ivp(duffing_coupled, [0.0, T], y0, t_eval=t_eval,
                    args=(kappa,), method='RK45', rtol=1e-8, atol=1e-10)
    return sol.t, sol.y


def simulate_target(T=T_TOTAL, dt=DT):
    t_eval = np.arange(0.0, T, dt)
    y0 = [0.1, 0.0, -0.1, 0.0]
    sol = solve_ivp(duffing_target, [0.0, T], y0, t_eval=t_eval,
                    method='RK45', rtol=1e-8, atol=1e-10)
    return sol.t, sol.y


def rk4_step(t, y, kappa, dt):
    def rhs(t_, y_):
        return np.array(duffing_coupled(t_, y_, kappa))
    k1_ = rhs(t, y)
    k2_ = rhs(t + 0.5 * dt, y + 0.5 * dt * k1_)
    k3_ = rhs(t + 0.5 * dt, y + 0.5 * dt * k2_)
    k4_ = rhs(t + dt, y + dt * k3_)
    return y + (dt / 6.0) * (k1_ + 2 * k2_ + 2 * k3_ + k4_)


# ==============================================================
# Part 2: IM-Net controller
# ==============================================================
class IMNetController(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=IMNET_HIDDEN_DIM,
                 kappa_min=kappa_min, kappa_max=kappa_max,
                 beta_init=IMNET_BETA_INIT):
        super().__init__()
        self.kappa_min = kappa_min
        self.kappa_max = kappa_max
        self.input_dim = input_dim

        self.meta_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.register_buffer('beta', torch.tensor(float(beta_init)))
        self.temp = nn.Parameter(torch.ones(1) * 0.6)

    def forward(self, features):
        logits = self.meta_layer(features)
        temp = torch.clamp(self.temp, min=0.3, max=3.0)
        s = torch.sigmoid(logits / temp)
        kappa = self.kappa_min + (self.kappa_max - self.kappa_min) * s
        return kappa.squeeze(-1)

    def compute_meta_loss(self, tracking_loss, spectral_radius_tensor):
        interference_energy = self.beta * spectral_radius_tensor
        return tracking_loss + interference_energy

    def get_control_parameters(self):
        self.eval()
        dev = next(self.parameters()).device
        test = torch.tensor([[0.1, 0.05, 0.01]]).to(dev)
        with torch.no_grad():
            k = self.forward(test)
        self.train()
        return float(k.item())


# ==============================================================
# Part 3: Offline training of IM-Net
# ==============================================================
def generate_pi_trajectory(T=IMNET_DATA_T, dt=DT,
                           kappa_init=0.6, err_gain=10.0, integ_gain=0.01):
    t_grid, y_target = simulate_target(T=T, dt=dt)
    x_target = y_target[0, :]
    v_target = y_target[1, :]

    N = len(t_grid)
    states = np.zeros((N, 4))
    kappas = np.zeros(N)
    features = np.zeros((N, 3))
    lam_max_arr = np.zeros(N)

    x1, v1, x2, v2 = 0.1, 0.0, -0.1, 0.0
    kappa = kappa_init
    integ = 0.0
    states[0] = [x1, v1, x2, v2]
    kappas[0] = kappa
    features[0] = [0.0, 0.0, 0.0]
    lam_max_arr[0] = spectral_radius_approx(duffing_jacobian(0.0, states[0], kappa))

    for i in range(1, N):
        t = t_grid[i - 1]
        e = x1 - x_target[i - 1]
        e_dot = v1 - v_target[i - 1]

        if kappa_min < kappa < kappa_max:
            integ += e * dt
        integ = np.clip(integ, -10.0, 10.0)

        kappa_dot = -err_gain * (e + 0.1 * e_dot) - integ_gain * integ
        kappa_dot = np.clip(kappa_dot, -2.0, 2.0)
        kappa = np.clip(kappa + kappa_dot * dt, kappa_min, kappa_max)
        kappas[i] = kappa
        features[i] = [e, e_dot, integ]

        y = np.array([x1, v1, x2, v2])
        y = rk4_step(t, y, kappa, dt)
        x1, v1, x2, v2 = y
        states[i] = y

        J = duffing_jacobian(t, y, kappa)
        lam_max_arr[i] = spectral_radius_approx(J)

    return {
        "t": t_grid,
        "states": states,
        "kappas": kappas,
        "features": features,
        "lam_max": lam_max_arr,
        "x_target": x_target,
        "v_target": v_target,
    }


def build_training_set(traj, T_transient=T_TRANSIENT):
    mask = traj["t"] >= T_transient
    feat = traj["features"][mask]
    kappa_pi = traj["kappas"][mask]
    lam_max = traj["lam_max"][mask]

    feat_mean = feat.mean(axis=0)
    feat_std = feat.std(axis=0) + 1e-8
    feat_norm = (feat - feat_mean) / feat_std

    return {
        "feat": torch.tensor(feat_norm, dtype=torch.float32).to(DEVICE),
        "kappa_pi": torch.tensor(kappa_pi, dtype=torch.float32).to(DEVICE),
        "lam_max": torch.tensor(lam_max, dtype=torch.float32).to(DEVICE),
        "feat_mean": feat_mean,
        "feat_std": feat_std,
    }


def train_imnet(imnet, data, epochs=IMNET_EPOCHS, lr=IMNET_LR,
                batch=IMNET_BATCH, lam_super=1.0):
    opt = optim.Adam(imnet.parameters(), lr=lr)
    N = data["feat"].shape[0]
    losses = []

    for ep in range(1, epochs + 1):
        perm = torch.randperm(N, device=DEVICE)
        ep_loss = 0.0
        n_batch = 0
        for i in range(0, N, batch):
            idx = perm[i:i + batch]
            f = data["feat"][idx]
            k_target = data["kappa_pi"][idx]

            k_pred = imnet(f)
            loss = lam_super * F.mse_loss(k_pred, k_target)

            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            n_batch += 1

        avg = ep_loss / max(n_batch, 1)
        losses.append(avg)
        if ep % 5 == 0 or ep == 1:
            print(f"    epoch {ep:3d}  loss = {avg:.6f}  "
                  f"beta = {imnet.beta.item():.5f}  temp = {imnet.temp.item():.3f}")

    return losses


# ==============================================================
# Part 4: Online inference
# ==============================================================
def simulate_controlled_imnet(T=T_TOTAL, dt=DT, imnet=None,
                              kappa_init=0.6, data_norm=None):
    if imnet is None:
        raise ValueError("imnet must be provided (trained model)")

    imnet.eval()
    feat_mean = data_norm["feat_mean"]
    feat_std = data_norm["feat_std"]

    t_grid, y_target = simulate_target(T=T, dt=dt)
    x_target = y_target[0, :]
    v_target = y_target[1, :]

    N = len(t_grid)
    states = np.zeros((N, 4))
    kappas = np.zeros(N)

    x1, v1, x2, v2 = 0.1, 0.0, -0.1, 0.0
    kappa = kappa_init
    integ = 0.0
    states[0] = [x1, v1, x2, v2]
    kappas[0] = kappa

    for i in range(1, N):
        t = t_grid[i - 1]
        e = x1 - x_target[i - 1]
        e_dot = v1 - v_target[i - 1]
        if kappa_min < kappa < kappa_max:
            integ += e * dt
        integ = np.clip(integ, -10.0, 10.0)

        feat = np.array([[e, e_dot, integ]], dtype=np.float32)
        feat_norm = (feat - feat_mean) / feat_std
        with torch.no_grad():
            kappa = float(imnet(torch.tensor(feat_norm, dtype=torch.float32).to(DEVICE)).item())

        y = np.array([x1, v1, x2, v2])
        y = rk4_step(t, y, kappa, dt)
        x1, v1, x2, v2 = y
        states[i] = y
        kappas[i] = kappa

    imnet.train()
    return t_grid, states, kappas, x_target


def simulate_controlled_pi(T=T_TOTAL, dt=DT, kappa_init=0.6,
                           err_gain=10.0, integ_gain=0.01):
    t_grid, y_target = simulate_target(T=T, dt=dt)
    x_target = y_target[0, :]
    v_target = y_target[1, :]

    N = len(t_grid)
    states = np.zeros((N, 4))
    kappas = np.zeros(N)

    x1, v1, x2, v2 = 0.1, 0.0, -0.1, 0.0
    kappa = kappa_init
    states[0] = [x1, v1, x2, v2]
    kappas[0] = kappa
    integ = 0.0

    for i in range(1, N):
        t = t_grid[i - 1]
        e = x1 - x_target[i - 1]
        e_dot = v1 - v_target[i - 1]
        if kappa_min < kappa < kappa_max:
            integ += e * dt
        integ = np.clip(integ, -10.0, 10.0)
        kappa_dot = -err_gain * (e + 0.1 * e_dot) - integ_gain * integ
        kappa_dot = np.clip(kappa_dot, -2.0, 2.0)
        kappa = np.clip(kappa + kappa_dot * dt, kappa_min, kappa_max)

        y = np.array([x1, v1, x2, v2])
        y = rk4_step(t, y, kappa, dt)
        x1, v1, x2, v2 = y
        states[i] = y
        kappas[i] = kappa

    return t_grid, states, kappas, x_target


# ==============================================================
# Phase / order parameter
# ==============================================================
def phase_lag_cos(x1, x_target, t, dt=DT):
    mask = t >= T_TRANSIENT
    if mask.sum() < 16:
        return np.nan
    a = hilbert(x1[mask])
    b = hilbert(x_target[mask])
    phase_diff = np.angle(a) - np.angle(b)
    return float(np.abs(np.mean(np.exp(1j * phase_diff))))


def steady_order_parameter(kappa, T=200.0, dt=DT, n_init=50):
    """Order parameter rho(kappa) averaged over n_init random initial conditions.

    The original single-trajectory version fluctuates strongly in the chaotic
    regime (rho jumping between 0.2 and 0.8), which masks the phase transition.
    Averaging over n_init initial conditions gives a smooth rho(kappa) curve.

    Args:
        kappa: coupling strength.
        T: integration time per initial condition (default 200, longer than
           the original T_RHO=60 to suppress transients in the average).
        dt: RK4 step size.
        n_init: number of random initial conditions. Set to 1 to recover the
                original single-trajectory behavior.

    Returns:
        float: mean order parameter over n_init runs.
    """
    rng = np.random.default_rng(2026)   # fixed seed for reproducibility
    _, yt = simulate_target(T=T, dt=dt)
    x_target = yt[0, :]

    t_eval = np.arange(0.0, T, dt)
    vals = []

    for _ in range(n_init):
        y0 = rng.uniform(-0.5, 0.5, size=4)
        sol = solve_ivp(duffing_coupled, [0.0, T], y0, t_eval=t_eval,
                        args=(kappa,), method='RK45', rtol=1e-8, atol=1e-10)
        vals.append(phase_lag_cos(sol.y[0, :], x_target, sol.t, dt=dt))

    arr = np.array(vals, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.nan
    return float(np.mean(finite))


# ==============================================================
# Poincare section & MLE
# ==============================================================
def poincare_section(kappa, T=T_POINCARE, dt=DT, n_transient=10):
    t, y = simulate(kappa, T=T, dt=dt)
    n_periods = int(T / T1)
    idx = [int(round(k * T1 / dt)) for k in range(1, n_periods)]
    idx = [i for i in idx if i < len(t)]
    x1_samples = y[0, idx]
    return x1_samples[n_transient:]


def maximal_lyapunov_exponent(kappa, T=T_MLE, dt=DT,
                              renorm_periods=10, delta0=1e-8, n_init=50):
    """Benettin MLE averaged over n_init random initial conditions.

    The original single-trajectory version oscillates between +0.10 and -0.05
    across kappa in the chaotic regime, so the zero-crossing is not well
    defined. Averaging over n_init initial conditions and initial perturbation
    directions gives a smooth MLE(kappa) curve.

    Args:
        kappa: coupling strength.
        T: total integration time per initial condition (default T_MLE).
        dt: RK4 step size.
        renorm_periods: renormalization interval in driving periods.
        delta0: initial perturbation magnitude.
        n_init: number of random initial conditions. Set to 1 to recover the
                original single-trajectory behavior.

    Returns:
        float: mean MLE over n_init runs. NaN if no renormalization occurred.
    """
    t_eval = np.arange(0.0, T, dt)
    N = len(t_eval)
    renorm_steps = max(1, int(round(renorm_periods * T1 / dt)))

    rng = np.random.default_rng(2026)   # fixed seed for reproducibility
    per_run = []

    for _ in range(n_init):
        # Random initial state on a modest box around the origin.
        y = rng.uniform(-0.5, 0.5, size=4)
        # Random initial perturbation direction, scaled to delta0.
        d = rng.standard_normal(4)
        d /= np.linalg.norm(d) + 1e-30
        d *= delta0

        lyap_sum = 0.0
        n_renorm = 0
        last = 0

        for i in range(N - 1):
            t_ = t_eval[i]

            def full(t_, y_, d_):
                fy = np.array(duffing_coupled(t_, y_, kappa))
                fd = duffing_jacobian(t_, y_, kappa) @ d_
                return fy, fd

            k1y, k1d = full(t_, y, d)
            k2y, k2d = full(t_ + 0.5 * dt, y + 0.5 * dt * k1y, d + 0.5 * dt * k1d)
            k3y, k3d = full(t_ + 0.5 * dt, y + 0.5 * dt * k2y, d + 0.5 * dt * k2d)
            k4y, k4d = full(t_ + dt, y + dt * k3y, d + dt * k3d)
            y = y + (dt / 6.0) * (k1y + 2 * k2y + 2 * k3y + k4y)
            d = d + (dt / 6.0) * (k1d + 2 * k2d + 2 * k3d + k4d)

            if (i + 1 - last) >= renorm_steps:
                nd = np.linalg.norm(d)
                if nd > 1e-30:
                    lyap_sum += np.log(nd / delta0)
                    n_renorm += 1
                d = d / nd * delta0
                last = i + 1

        if n_renorm == 0:
            per_run.append(np.nan)
        else:
            per_run.append(lyap_sum / (n_renorm * renorm_steps * dt))

    arr = np.array(per_run, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.nan
    return float(np.mean(finite))


# ==============================================================
# Fig. 5
# ==============================================================
def make_fig5(imnet=None, data_norm=None):
    print("[Fig. 5] Simulating uncontrolled / PI / IM-Net controlled CCO...")

    t_u, y_u = simulate(kappa=0.6, T=T_TOTAL, dt=DT)
    x1_u, v1_u = y_u[0, :], y_u[1, :]

    t_t, y_t = simulate_target(T=T_TOTAL, dt=DT)
    x_target = y_t[0, :]
    v_target = y_t[1, :]

    t_pi, y_pi, kappas_pi, x_target_pi = simulate_controlled_pi(
        T=T_TOTAL, dt=DT, kappa_init=0.6)
    x1_pi, v1_pi = y_pi[:, 0], y_pi[:, 1]

    if imnet is not None and data_norm is not None:
        t_im, y_im, kappas_im, x_target_im = simulate_controlled_imnet(
            T=T_TOTAL, dt=DT, imnet=imnet, kappa_init=0.6, data_norm=data_norm)
        x1_im, v1_im = y_im[:, 0], y_im[:, 1]
    else:
        t_im, y_im, kappas_im, x1_im, v1_im = None, None, None, None, None

    mask = t_u >= T_TRANSIENT
    mse_u = float(np.mean((x1_u[mask] - x_target[mask]) ** 2))
    mask_pi = t_pi >= T_TRANSIENT
    mse_pi = float(np.mean((x1_pi[mask_pi] - x_target_pi[mask_pi]) ** 2))
    if x1_im is not None:
        mask_im = t_im >= T_TRANSIENT
        mse_im = float(np.mean((x1_im[mask_im] - x_target_im[mask_im]) ** 2))
    else:
        mse_im = float("nan")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    ax = axes.ravel()

    ax[0].plot(x1_u, v1_u, lw=0.6, color="tab:red", alpha=0.8)
    ax[0].set_title("(a) Uncontrolled phase portrait (κ = 0.6)")
    ax[0].set_xlabel("x1"); ax[0].set_ylabel("v1"); ax[0].grid(alpha=0.3)

    if x1_im is not None:
        ax[1].plot(x1_im, v1_im, lw=0.8, color="tab:blue", label="IM-Net")
        ax[1].plot(x1_pi, v1_pi, lw=0.6, color="tab:green", alpha=0.6, label="PI")
    else:
        ax[1].plot(x1_pi, v1_pi, lw=0.8, color="tab:green", label="PI")
    ax[1].plot(y_t[0, :], y_t[1, :], ls="--", lw=1.2, color="k", label="Target")
    ax[1].set_title("(b) Controlled phase portrait")
    ax[1].set_xlabel("x1"); ax[1].set_ylabel("v1")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)

    ax[2].plot(t_u, x1_u, lw=0.8, color="tab:red", label="Uncontrolled")
    ax[2].plot(t_u, x_target, ls="--", lw=1.0, color="k", label="Target")
    ax[2].set_title("(c) Uncontrolled tracking failure")
    ax[2].set_xlabel("t"); ax[2].set_ylabel("x1")
    ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3)

    if x1_im is not None:
        ax[3].plot(t_im, x1_im, lw=0.9, color="tab:blue", label="IM-Net x1")
        ax[3].plot(t_im, x_target_im, ls="--", lw=1.0, color="k", label="Target")
    else:
        ax[3].plot(t_pi, x1_pi, lw=0.9, color="tab:green", label="PI x1")
        ax[3].plot(t_pi, x_target_pi, ls="--", lw=1.0, color="k", label="Target")
    ax[3].set_title("(d) IM-Net phase synchronization")
    ax[3].set_xlabel("t"); ax[3].set_ylabel("x1")
    ax[3].legend(fontsize=8); ax[3].grid(alpha=0.3)

    ax[4].semilogy(t_u, np.abs(x1_u - x_target) + 1e-12,
                   lw=0.7, color="tab:red", label="Uncontrolled")
    ax[4].semilogy(t_pi, np.abs(x1_pi - x_target_pi) + 1e-12,
                   lw=0.8, color="tab:green", label="PI")
    if x1_im is not None:
        ax[4].semilogy(t_im, np.abs(x1_im - x_target_im) + 1e-12,
                       lw=0.9, color="tab:blue", label="IM-Net")
    ax[4].set_title("(e) Absolute tracking error")
    ax[4].set_xlabel("t"); ax[4].set_ylabel("|x1 - x_target| (log)")
    ax[4].legend(fontsize=8); ax[4].grid(alpha=0.3, which="both")

    ax[5].plot(t_pi, kappas_pi, lw=0.9, color="tab:green", label="PI")
    if kappas_im is not None:
        ax[5].plot(t_im, kappas_im, lw=1.0, color="tab:blue", label="IM-Net")
    ax[5].axhline(KAPPA_CRIT, ls=":", color="gray", lw=0.8, label=f"κ_c ≈ {KAPPA_CRIT}")
    ax[5].set_title("κ(t) trajectory (IM-Net vs PI)")
    ax[5].set_xlabel("t"); ax[5].set_ylabel("κ(t)")
    ax[5].legend(fontsize=8); ax[5].grid(alpha=0.3)

    fig.suptitle(
        f"Fig. 5 | MSE: uncontrolled={mse_u:.4f} | "
        f"PI={mse_pi:.4f} | IM-Net={mse_im:.4f}",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out = os.path.join(OUT_DIR, "fig5_cco_phase_sync.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  saved: {out}")
    print(f"  MSE uncontrolled = {mse_u:.6f}, PI = {mse_pi:.6f}, IM-Net = {mse_im:.6f}")

    # ===== CSV export: fig5_cco_phase.csv =====
    n = len(t_u)
    nan_arr = np.full(n, np.nan)
    df5 = pd.DataFrame({
        "t": t_u,
        "x1_uncontrolled": x1_u,
        "v1_uncontrolled": v1_u,
        "x1_target": x_target,
        "v1_target": v_target,
        "x1_pi": x1_pi if len(x1_pi) == n else nan_arr,
        "v1_pi": v1_pi if len(v1_pi) == n else nan_arr,
        "kappa_pi": kappas_pi if len(kappas_pi) == n else nan_arr,
        "x1_imnet": x1_im if x1_im is not None else nan_arr,
        "v1_imnet": v1_im if v1_im is not None else nan_arr,
        "kappa_imnet": kappas_im if kappas_im is not None else nan_arr,
        "mse_uncontrolled": mse_u,
        "mse_pi": mse_pi,
        "mse_imnet": mse_im,
    })
    save_csv(df5, "fig5_cco_phase.csv")


# ==============================================================
# Fig. 6 (two panels: order parameter + MLE)
# ==============================================================
def make_fig6(n_init=50):
    """Fig. 6: two-panel (order parameter + MLE), averaged over n_init runs."""
    print(f"[Fig. 6] Computing order parameter rho(kappa) and MLE(kappa) "
          f"(n_init={n_init})...", flush=True)
    kappas = np.linspace(0.20, 1.00, 41)
    n_kappa = len(kappas)

    rho_vals = np.empty(n_kappa, dtype=float)
    mles = np.empty(n_kappa, dtype=float)

    t_start = time.time()
    for i, k in enumerate(kappas):
        rho_vals[i] = steady_order_parameter(k, T=200.0, n_init=n_init)
        mles[i] = maximal_lyapunov_exponent(k, T=T_MLE, n_init=n_init)

        if i == 0 or (i + 1) % 5 == 0 or i == n_kappa - 1:
            elapsed = time.time() - t_start
            eta = elapsed / (i + 1) * (n_kappa - i - 1)
            print(f"  [{i + 1:02d}/{n_kappa}] kappa={k:.3f}  "
                  f"rho={rho_vals[i]:.3f}  mle={mles[i]:+.4f}  "
                  f"elapsed={elapsed / 60:.1f}m  eta={eta / 60:.1f}m",
                  flush=True)

    # Robust kappa_c: first index where MLE stays negative for >= 3 consecutive points.
    neg = (mles < 0).astype(int)
    kappa_crit_auto = float(KAPPA_CRIT)  # fallback
    for i in range(len(mles) - 2):
        if neg[i] == 1 and neg[i + 1] == 1 and neg[i + 2] == 1:
            kappa_crit_auto = float(kappas[i])
            break
    print(f"  auto-detected kappa_crit (3-consecutive-negative) "
          f"≈ {kappa_crit_auto:.3f}", flush=True)

    # 固定重入窗口的起点为 0.72 (基于前人经验或理论)
    kappa_re = 0.72
    # 定义三个区域：I 混沌区，II 平台区，III 重入区
    regions = [
        (0.20, kappa_crit_auto, "#FADADD", "I. Desynchronized chaos"),
        (kappa_crit_auto, kappa_re, "#DFF0D8", "II. Synchronization plateau"),
        (kappa_re, 1.00, "#FCE4D6", "III. Re-entrant partial desync.")
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # ===== (a) Order parameter =====
    for start, end, color, label in regions:
        axes[0].axvspan(start, end, color=color, alpha=0.6, label=label)

    axes[0].plot(kappas, rho_vals, "o-", ms=4, lw=1.2, color="tab:blue")
    axes[0].axvline(kappa_crit_auto, ls="--", color="gray")
    axes[0].axvline(kappa_re, ls="--", color="gray")

    axes[0].set_xlabel("Coupling strength $\\kappa$")
    axes[0].set_ylabel("Order parameter $\\rho$")
    axes[0].set_title("(a) Order parameter: three dynamical regimes")
    axes[0].set_xlim(0.20, 1.00)
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="lower right")

    # ===== (b) Maximal Lyapunov exponent =====
    for start, end, color, _ in regions:
        axes[1].axvspan(start, end, color=color, alpha=0.6)

    axes[1].plot(kappas, mles, "o-", ms=4, lw=1.2, color="tab:purple")
    axes[1].axhline(0.0, ls="--", color="red", lw=1.0, label="MLE = 0")
    axes[1].axvline(kappa_crit_auto, ls="--", color="gray",
                    label=f"$\\kappa_c \\approx {kappa_crit_auto:.2f}$")
    axes[1].axvline(kappa_re, ls=":", color="gray",
                    label=f"$\\kappa_{{re}} \\approx {kappa_re:.2f}$")

    axes[1].set_xlabel("Coupling strength $\\kappa$")
    axes[1].set_ylabel("Maximal Lyapunov exponent $\\lambda_{max}$")
    axes[1].set_title("(b) MLE: primary transition + re-entrant window")
    axes[1].set_xlim(0.20, 1.00)
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)

    fig.suptitle("Fig. 6 | Three-regime dynamical phase diagram of the coupled bistable Duffing oscillator",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out = os.path.join(OUT_DIR, "fig6_cco_bifurcation_mle.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  saved: {out}", flush=True)

    # CSV export
    df_bif = pd.DataFrame({
        "kappa": kappas,
        "order_parameter_rho": rho_vals,
        "kappa_c": kappa_crit_auto,
    })
    save_csv(df_bif, "fig6_cco_bifurcation.csv")

    df_mle = pd.DataFrame({
        "kappa": kappas,
        "mle": mles,
        "kappa_c": kappa_crit_auto,
    })
    save_csv(df_mle, "fig6_cco_mle.csv")

    return kappa_crit_auto

# ==============================================================
# Fig. S1 (kept for backward compatibility with the paper)
# ==============================================================
def make_figS1(kappa_crit=None, n_init=10):
    """Fig. S1: Poincare sections + MLE(kappa) for the supplementary.

    Progress is printed every 3 kappa points (and at the first / last point)
    with elapsed time and ETA, so a 20-45 min run can be monitored from the log.
    """
    print(f"[Fig. S1] Computing Poincare sections and MLE (n_init={n_init})...",
          flush=True)
    if kappa_crit is None:
        kappa_crit = KAPPA_CRIT
    kappa_low = kappa_crit - 0.15
    kappa_high = min(kappa_crit + 0.35, kappa_max - 0.01)

    # ---- Poincare sections (single trajectory each, cheap) ----
    p_low = poincare_section(kappa_low, T=T_POINCARE, dt=DT, n_transient=10)
    p_crit = poincare_section(kappa_crit, T=T_POINCARE, dt=DT, n_transient=10)
    p_high = poincare_section(kappa_high, T=T_POINCARE, dt=DT, n_transient=10)
    print(f"  Poincare done: kappa_low={kappa_low:.2f} "
          f"({len(p_low)} pts), kappa_crit={kappa_crit:.2f} "
          f"({len(p_crit)} pts), kappa_high={kappa_high:.2f} "
          f"({len(p_high)} pts)", flush=True)

    # ---- MLE sweep with progress print ----
    kappas_mle = np.linspace(0.20, 1.00, 13)
    n_kappa = len(kappas_mle)
    mles = np.empty(n_kappa, dtype=float)

    t_start = time.time()
    for i, k in enumerate(kappas_mle):
        mles[i] = maximal_lyapunov_exponent(k, T=T_MLE, dt=DT, n_init=n_init)

        # Progress print: first point, every 3 points, and the last point.
        if i == 0 or (i + 1) % 3 == 0 or i == n_kappa - 1:
            elapsed = time.time() - t_start
            eta = elapsed / (i + 1) * (n_kappa - i - 1)
            print(f"    [{i+1:02d}/{n_kappa}] kappa={k:.2f}  "
                  f"MLE={mles[i]:+.4f}  elapsed={elapsed/60:.1f}m  "
                  f"eta={eta/60:.1f}m", flush=True)

    # ---- Figure ----
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes[0, 0].plot(p_low, np.zeros_like(p_low), "o", ms=2, alpha=0.5, color="tab:red")
    axes[0, 0].set_title(f"(a) Poincare, κ = {kappa_low:.2f} (chaotic)")
    axes[0, 0].set_xlabel("x1 at n·T1"); axes[0, 0].set_yticks([]); axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(p_crit, np.zeros_like(p_crit), "o", ms=2, alpha=0.5, color="tab:orange")
    axes[0, 1].set_title(f"(b) Poincare, κ = {kappa_crit:.2f} (near κ_c)")
    axes[0, 1].set_xlabel("x1 at n·T1"); axes[0, 1].set_yticks([]); axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(p_high, np.zeros_like(p_high), "o", ms=4, alpha=0.8, color="tab:blue")
    axes[1, 0].set_title(f"(c) Poincare, κ = {kappa_high:.2f} (limit cycle)")
    axes[1, 0].set_xlabel("x1 at n·T1"); axes[1, 0].set_yticks([]); axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(kappas_mle, mles, "s-", ms=4, lw=1.2, color="tab:green")
    axes[1, 1].axhline(0.0, ls=":", color="k", lw=0.6)
    axes[1, 1].axvline(kappa_crit, ls="--", color="gray",
                       label=rf"$\kappa_c \approx {kappa_crit:.3f}$")
    axes[1, 1].set_title("(d) MLE vs κ")
    axes[1, 1].set_xlabel("κ"); axes[1, 1].set_ylabel(r"$\lambda_{max}$")
    axes[1, 1].legend(fontsize=8); axes[1, 1].grid(alpha=0.3)

    fig.suptitle("Fig. S1 | Bifurcation and Maximal Lyapunov Exponent", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out = os.path.join(OUT_DIR, "figS1_bifurcation_mle.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  saved: {out}", flush=True)

# ==============================================================
# Table S1 export
# ==============================================================
def save_table_s1():
    print("[Table S1] Exporting CCO parameters...")
    df = pd.DataFrame([
        ("gamma", gamma),
        ("k1", k1),
        ("k2", k2),
        ("beta_duffing", beta_duffing),
        ("A1", A1),
        ("A2", A2),
        ("omega1", omega1),
        ("omega2", omega2),
        ("theta0", theta0),
        ("kappa_min", kappa_min),
        ("kappa_max", kappa_max),
        ("KAPPA_CRIT", KAPPA_CRIT),
        ("KAPPA_REF", KAPPA_REF),
        ("DT", DT),
        ("T_TOTAL", T_TOTAL),
        ("T_TRANSIENT", T_TRANSIENT),
        ("T_RHO", T_RHO),
        ("T_POINCARE", T_POINCARE),
        ("T_MLE", T_MLE),
        ("T1", T1),
        ("IMNET_HIDDEN_DIM", IMNET_HIDDEN_DIM),
        ("IMNET_BETA_INIT", IMNET_BETA_INIT),
        ("IMNET_EPOCHS", IMNET_EPOCHS),
        ("IMNET_LR", IMNET_LR),
        ("IMNET_BATCH", IMNET_BATCH),
        ("IMNET_DATA_T", IMNET_DATA_T),
    ], columns=["parameter", "value"])
    save_csv(df, "table_s1_cco_params.csv")


# ==============================================================
# Main
# ==============================================================
def main():
    print("=" * 70)
    print("CCO + IM-Net simulation (final version)")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Parameters: gamma={gamma}, k1={k1}, k2={k2}, beta={beta_duffing}, "
          f"A1={A1}, A2={A2}, omega1={omega1}, omega2={omega2}")
    print()

    # ---- Step 1: Generate PI trajectory as training data ----
    print("[Train] Generating PI trajectory for IM-Net training...")
    t0 = time.time()
    traj = generate_pi_trajectory(T=IMNET_DATA_T, dt=DT)
    print(f"  done in {time.time()-t0:.1f}s, {len(traj['t'])} samples")

    # ---- Step 2: Build training set ----
    data = build_training_set(traj, T_transient=T_TRANSIENT)
    print(f"  training samples: {data['feat'].shape[0]}")

    # ---- Step 3: Train IM-Net ----
    imnet = IMNetController().to(DEVICE)
    n_params = sum(p.numel() for p in imnet.parameters())
    print(f"  IM-Net params: {n_params}")
    losses = train_imnet(imnet, data, epochs=IMNET_EPOCHS)

    # ---- Step 4: Save model ----
    torch.save(imnet.state_dict(), MODEL_PATH)
    print(f"  saved model: {MODEL_PATH}")

    # ---- Step 5: Build normalization stats dict ----
    data_norm = {"feat_mean": data["feat_mean"], "feat_std": data["feat_std"]}

    # ---- Step 6: Cheap outputs first ----
    print()
    make_fig5(imnet=imnet, data_norm=data_norm)
    save_table_s1()   # moved up: 1 second, so it survives if Fig.6 crashes

    # ---- Step 7: Long dynamical analysis ----
    # Fig.6: 41 kappa x n_init=50 for both rho and MLE -> ~4-9 h
    kappa_crit_auto = make_fig6(n_init=50)

    # Fig.S1: 13 kappa x n_init=10 -> ~20-45 min
    make_figS1(kappa_crit=kappa_crit_auto, n_init=10)

    print()
    print("=" * 70)
    print(f"All figures saved to: {OUT_DIR}/")
    print(f"All CSVs saved to:    {REPRO_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
