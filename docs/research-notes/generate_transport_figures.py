"""Generate the mathematical figures used by the transport research notes.

The figures use toy distributions and, where a learned model would normally be
required, an analytic oracle denoiser.  This keeps each plot focused on the
mathematical object rather than on the quality of a trained neural network.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from scipy.optimize import linear_sum_assignment
from scipy.stats import wasserstein_distance

FIGURE_DIR = Path(__file__).with_name("figures")
SEED = 17


def _finish(
    fig: plt.Figure,
    name: str,
    *,
    rect: tuple[float, float, float, float] | None = None,
) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    if rect is None:
        fig.tight_layout()
    else:
        fig.tight_layout(rect=rect)
    fig.savefig(FIGURE_DIR / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _style_axis(ax: Axes, title: str, equal: bool = False) -> None:
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    if equal:
        ax.set_aspect("equal", adjustable="box")


def sample_target(rng: np.random.Generator, n: int) -> np.ndarray:
    """Sample a four-component two-dimensional Gaussian mixture."""
    means = np.array([[-2.0, -1.2], [-1.5, 1.7], [1.6, -1.4], [2.1, 1.4]])
    component = rng.integers(0, len(means), size=n)
    return means[component] + 0.28 * rng.normal(size=(n, 2))


def brownian_motion() -> None:
    rng = np.random.default_rng(SEED)
    steps, paths, dt = 500, 14, 1 / 500
    increments = np.sqrt(dt) * rng.normal(size=(steps, paths))
    values = np.vstack([np.zeros(paths), np.cumsum(increments, axis=0)])
    time = np.linspace(0, 1, steps + 1)

    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    ax.plot(time, values, alpha=0.72, linewidth=1.0)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set(xlabel="$t$", ylabel="$W_t$")
    _style_axis(ax, "Brownian motion: identical law, different sample paths")
    _finish(fig, "brownian_motion.png")


def sde_evolution() -> None:
    """Simulate dX = -0.8 X dt + 1.1 dW with Euler--Maruyama."""
    rng = np.random.default_rng(SEED + 1)
    n, steps, dt = 20_000, 500, 1 / 500
    x = rng.normal(loc=-2.2, scale=0.32, size=n)
    snapshots = {0: x.copy()}
    requested = {100, 250, 500}
    for step in range(1, steps + 1):
        x += -0.8 * x * dt + 1.1 * np.sqrt(dt) * rng.normal(size=n)
        if step in requested:
            snapshots[step] = x.copy()

    fig, axes = plt.subplots(1, 4, figsize=(11.5, 2.8), sharex=True, sharey=True)
    bins = np.linspace(-3.4, 2.6, 90)
    for ax, step in zip(axes, [0, 100, 250, 500], strict=True):
        ax.hist(snapshots[step], bins=bins, density=True, color="#4C78A8", alpha=0.82)
        ax.set_xlabel("$x$")
        _style_axis(ax, f"$t={step * dt:.1f}$")
    axes[0].set_ylabel("density")
    fig.suptitle(r"Fokker--Planck view: marginals evolve under drift and diffusion")
    _finish(fig, "sde_evolution.png")


def ode_vector_field() -> None:
    """Show a deterministic non-autonomous flow and its pushforward snapshots."""
    rng = np.random.default_rng(SEED + 2)
    grid = np.linspace(-3.2, 3.2, 19)
    xx, yy = np.meshgrid(grid, grid)
    # A stable spiral field; all randomness is in the initial state.
    uu = 0.55 * xx - 1.05 * yy + 0.7
    vv = 1.05 * xx + 0.10 * yy - 0.25

    x0 = 0.45 * rng.normal(size=(32, 2)) + np.array([-1.2, -0.7])
    times = np.linspace(0, 1, 101)
    trajectories = np.empty((len(times), len(x0), 2))
    trajectories[0] = x0
    dt = times[1] - times[0]
    for i in range(1, len(times)):
        x = trajectories[i - 1]
        velocity = np.column_stack(
            [
                0.55 * x[:, 0] - 1.05 * x[:, 1] + 0.7,
                1.05 * x[:, 0] + 0.10 * x[:, 1] - 0.25,
            ]
        )
        trajectories[i] = x + dt * velocity

    fig, ax = plt.subplots(figsize=(6.2, 5.5))
    speed = np.sqrt(uu**2 + vv**2)
    ax.quiver(xx, yy, uu / speed, vv / speed, speed, cmap="Greys", alpha=0.45)
    for path in trajectories.transpose(1, 0, 2):
        ax.plot(path[:, 0], path[:, 1], color="#F58518", alpha=0.55, linewidth=0.9)
    for index, color, label in [
        (0, "#4C78A8", "$p_0$"),
        (50, "#54A24B", "$p_{0.5}$"),
        (100, "#E45756", "$p_1$"),
    ]:
        points = trajectories[index]
        ax.scatter(points[:, 0], points[:, 1], s=13, color=color, label=label, zorder=3)
    ax.set(xlim=(-3.2, 3.2), ylim=(-3.2, 3.2), xlabel="$x_1$", ylabel="$x_2$")
    _style_axis(
        ax, "ODE: vector field, trajectories, and evolving pushforward", equal=True
    )
    ax.legend(frameon=False)
    _finish(fig, "ode_vector_field.png")


def ode_vs_sde_fixed_initial() -> None:
    """Contrast initial-value randomness with pathwise Brownian randomness."""
    rng = np.random.default_rng(SEED + 11)
    times = np.linspace(0, 1, 250)
    dt = times[1] - times[0]
    initial = np.array([[-1.8, -0.4]])

    def drift(x: np.ndarray) -> np.ndarray:
        return np.column_stack(
            [
                -0.35 * x[:, 0] - 0.9 * x[:, 1] + 1.1,
                0.9 * x[:, 0] - 0.35 * x[:, 1] + 0.4,
            ]
        )

    ode = np.empty((len(times), 1, 2))
    ode[0] = initial
    sde = np.empty((len(times), 18, 2))
    sde[0] = np.repeat(initial, 18, axis=0)
    for i in range(1, len(times)):
        ode[i] = ode[i - 1] + dt * drift(ode[i - 1])
        sde[i] = (
            sde[i - 1]
            + dt * drift(sde[i - 1])
            + 0.38 * np.sqrt(dt) * rng.normal(size=(18, 2))
        )

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8), sharex=True, sharey=True)
    axes[0].plot(ode[:, 0, 0], ode[:, 0, 1], color="#F58518", linewidth=2.0)
    axes[0].scatter(*initial[0], marker="x", color="black", zorder=3)
    for path in sde.transpose(1, 0, 2):
        axes[1].plot(path[:, 0], path[:, 1], color="#4C78A8", alpha=0.52, linewidth=0.8)
    axes[1].scatter(*initial[0], marker="x", color="black", zorder=3)
    for ax, title in zip(
        axes,
        [
            "ODE: one path for a fixed initial state",
            "SDE: a path law for the same initial state",
        ],
        strict=True,
    ):
        ax.set(xlim=(-2.2, 1.5), ylim=(-1.2, 1.8), xlabel="$x_1$")
        _style_axis(ax, title, equal=True)
    axes[0].set_ylabel("$x_2$")
    _finish(fig, "ode_vs_sde_fixed_initial.png")


def flow_matching_paths() -> None:
    rng = np.random.default_rng(SEED + 3)
    n = 34
    source = 0.52 * rng.normal(size=(n, 2))
    target = sample_target(rng, n)
    random_target = target[rng.permutation(n)]
    cost = ((source[:, None] - target[None, :]) ** 2).sum(axis=2)
    rows, cols = linear_sum_assignment(cost)
    ot_target = target[cols[np.argsort(rows)]]
    ts = np.linspace(0, 1, 45)

    def straight_crossings(starts: np.ndarray, ends: np.ndarray) -> int:
        def orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
            ab = b - a
            ac = c - a
            return float(ab[0] * ac[1] - ab[1] * ac[0])

        crossings = 0
        for i in range(len(starts)):
            for j in range(i + 1, len(starts)):
                first = orientation(starts[i], ends[i], starts[j])
                second = orientation(starts[i], ends[i], ends[j])
                third = orientation(starts[j], ends[j], starts[i])
                fourth = orientation(starts[j], ends[j], ends[i])
                if first * second < 0 and third * fourth < 0:
                    crossings += 1
        return crossings

    random_length = np.linalg.norm(random_target - source, axis=1).mean()
    ot_length = np.linalg.norm(ot_target - source, axis=1).mean()
    random_crossings = straight_crossings(source, random_target)
    ot_crossings = straight_crossings(source, ot_target)

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.3), sharex=True, sharey=True)
    specifications = [
        (
            random_target,
            False,
            f"independent coupling: mean length {random_length:.2f}\n{random_crossings} straight-path crossings",
        ),
        (random_target, True, "same coupling\ncurved conditional paths"),
        (
            ot_target,
            False,
            f"minibatch OT: mean length {ot_length:.2f}\n{ot_crossings} straight-path crossings",
        ),
    ]
    for ax, (paired, curved, title) in zip(axes, specifications, strict=True):
        for start, end in zip(source, paired, strict=True):
            path = (1 - ts[:, None]) * start + ts[:, None] * end
            if curved:
                delta = end - start
                perpendicular = np.array([-delta[1], delta[0]])
                perpendicular /= np.linalg.norm(perpendicular) + 1e-8
                path += 0.55 * np.sin(np.pi * ts)[:, None] * perpendicular
            ax.plot(path[:, 0], path[:, 1], color="#999999", alpha=0.38, linewidth=0.75)
        ax.scatter(source[:, 0], source[:, 1], s=16, color="#4C78A8", label="$p_0$")
        ax.scatter(paired[:, 0], paired[:, 1], s=16, color="#E45756", label="$p_1$")
        ax.set(xlim=(-3.0, 3.0), ylim=(-2.5, 2.7), xlabel="$x_1$")
        _style_axis(ax, title, equal=True)
        ax.set_title(title, fontsize=10, pad=10, linespacing=1.25)
    axes[0].set_ylabel("$x_2$")
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle(
        "The endpoints do not determine the coupling or the probability path",
        fontsize=14,
        y=0.98,
    )
    _finish(fig, "flow_matching_paths.png", rect=(0.0, 0.0, 1.0, 0.84))


def flow_conditional_marginal_field() -> None:
    """Visualize conditional velocity targets and their local regression average."""
    rng = np.random.default_rng(SEED + 12)
    n, time = 420, 0.5
    source = 0.55 * rng.normal(size=(n, 2))
    target = sample_target(rng, n)
    target = target[rng.permutation(n)]
    xt = (1 - time) * source + time * target
    velocity = target - source

    grid = np.linspace(-2.5, 2.5, 17)
    xx, yy = np.meshgrid(grid, grid)
    query = np.column_stack([xx.ravel(), yy.ravel()])
    bandwidth = 0.42
    squared_distance = ((query[:, None] - xt[None, :]) ** 2).sum(axis=2)
    weights = np.exp(-squared_distance / (2 * bandwidth**2))
    effective_mass = weights.sum(axis=1)
    marginal = weights @ velocity / (effective_mass[:, None] + 1e-10)
    keep = effective_mass > np.quantile(effective_mass, 0.38)

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.0), sharex=True, sharey=True)
    subset = rng.choice(n, size=125, replace=False)
    axes[0].scatter(xt[:, 0], xt[:, 1], s=5, color="#BBBBBB", alpha=0.35)
    axes[0].quiver(
        xt[subset, 0],
        xt[subset, 1],
        velocity[subset, 0],
        velocity[subset, 1],
        color="#E45756",
        alpha=0.52,
        angles="xy",
        scale_units="xy",
        scale=4.5,
    )
    axes[1].scatter(xt[:, 0], xt[:, 1], s=5, color="#BBBBBB", alpha=0.22)
    axes[1].quiver(
        query[keep, 0],
        query[keep, 1],
        marginal[keep, 0],
        marginal[keep, 1],
        color="#4C78A8",
        alpha=0.9,
        angles="xy",
        scale_units="xy",
        scale=4.5,
    )
    for ax, title in zip(
        axes,
        [
            r"conditional targets $u_t(X_t\mid Z)$",
            r"local estimate of $\mathbb{E}[u_t\mid X_t=x]$",
        ],
        strict=True,
    ):
        ax.set(xlim=(-2.7, 2.7), ylim=(-2.7, 2.7), xlabel="$x_1$")
        _style_axis(ax, title, equal=True)
    axes[0].set_ylabel("$x_2$")
    fig.suptitle(
        "Conditional Flow Matching turns conflicting path labels into a marginal field"
    )
    _finish(fig, "flow_conditional_marginal_field.png")


def same_marginals_different_joint() -> None:
    """Connect identical q(x_t|x0) point clouds with two temporal couplings."""
    rng = np.random.default_rng(SEED + 13)
    n_paths = 32
    times = np.array([0.04, 0.20, 0.48, 0.72, 0.92, 1.0])
    alpha_bar = np.exp(-4.2 * times)
    x0 = 1.7
    # Each column is sampled once and reused in both panels.  Only the
    # cross-time pairing changes, so the empirical time slices are identical.
    slices = np.sqrt(alpha_bar)[:, None] * x0 + np.sqrt(1 - alpha_bar)[
        :, None
    ] * rng.normal(size=(len(times), n_paths))
    rank_coupled = np.sort(slices, axis=1)
    shuffled = np.empty_like(rank_coupled)
    shuffled[0] = rank_coupled[0]
    for index in range(1, len(times)):
        shuffled[index] = rank_coupled[index, rng.permutation(n_paths)]

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), sharex=True, sharey=True)
    for ax, process, title in [
        (axes[0], rank_coupled, "joint A: rank-preserving coupling"),
        (axes[1], shuffled, "joint B: shuffled temporal coupling"),
    ]:
        ax.plot(times, process, color="#4C78A8", alpha=0.43, linewidth=0.8)
        for t_index in range(1, len(times) - 1):
            ax.scatter(
                np.full(n_paths, times[t_index]),
                process[t_index],
                s=7,
                color="#E45756",
                alpha=0.45,
                zorder=3,
            )
        # The deterministic sample contains one value just above 3.0.  Keep a
        # small visual margin so the path and marker are not clipped.
        ax.set(xlabel="$t$", ylim=(-3.0, 3.3))
        _style_axis(ax, title)
    axes[0].set_ylabel("$X_t$")
    fig.suptitle(
        r"Identical sampled $q(X_t\mid x_0)$ slices, different temporal couplings"
    )
    _finish(fig, "same_marginals_different_joint.png")


MIXTURE_MEANS = np.array([[-2.0, -1.2], [-1.5, 1.7], [1.6, -1.4], [2.1, 1.4]])
DATA_STD = 0.28


def _mixture_posterior_mean(xt: np.ndarray, alpha_bar: float) -> np.ndarray:
    """Return E[x0 | xt] for the toy Gaussian-mixture data distribution."""
    signal_var = alpha_bar * DATA_STD**2
    observation_var = signal_var + 1 - alpha_bar
    noisy_means = np.sqrt(alpha_bar) * MIXTURE_MEANS
    diff = xt[:, None, :] - noisy_means[None, :, :]
    logits = -0.5 * (diff**2).sum(axis=2) / observation_var
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    weights /= weights.sum(axis=1, keepdims=True)

    gain = DATA_STD**2 * np.sqrt(alpha_bar) / observation_var
    component_posteriors = MIXTURE_MEANS[None, :, :] + gain * (
        xt[:, None, :] - noisy_means[None, :, :]
    )
    return (weights[:, :, None] * component_posteriors).sum(axis=1)


def _epsilon_oracle(xt: np.ndarray, alpha_bar: float) -> np.ndarray:
    x0_mean = _mixture_posterior_mean(xt, alpha_bar)
    return (xt - np.sqrt(alpha_bar) * x0_mean) / np.sqrt(1 - alpha_bar)


def score_field() -> None:
    alpha_bar = 0.52
    grid = np.linspace(-3.4, 3.4, 28)
    xx, yy = np.meshgrid(grid, grid)
    points = np.column_stack([xx.ravel(), yy.ravel()])
    epsilon = _epsilon_oracle(points, alpha_bar)
    score = -epsilon / np.sqrt(1 - alpha_bar)

    observation_var = alpha_bar * DATA_STD**2 + 1 - alpha_bar
    noisy_means = np.sqrt(alpha_bar) * MIXTURE_MEANS
    density = np.zeros(len(points))
    for mean in noisy_means:
        density += np.exp(-0.5 * ((points - mean) ** 2).sum(axis=1) / observation_var)
    density = density.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(6.1, 5.4))
    ax.contourf(xx, yy, density, levels=18, cmap="Blues", alpha=0.8)
    magnitude = np.linalg.norm(score, axis=1).reshape(xx.shape)
    ax.quiver(
        xx,
        yy,
        score[:, 0].reshape(xx.shape) / (magnitude + 1e-8),
        score[:, 1].reshape(xx.shape) / (magnitude + 1e-8),
        magnitude,
        cmap="magma",
        alpha=0.78,
    )
    ax.set(xlabel="$x_1$", ylabel="$x_2$")
    _style_axis(
        ax,
        r"Score field $\nabla_x\log p_t(x)$ follows local log-density ascent",
        equal=True,
    )
    _finish(fig, "score_field.png")


def forward_diffusion() -> None:
    rng = np.random.default_rng(SEED + 4)
    x0 = sample_target(rng, 4_000)
    alpha_bars = [1.0, 0.72, 0.28, 0.015]
    fig, axes = plt.subplots(1, 4, figsize=(12.2, 3.0), sharex=True, sharey=True)
    for ax, alpha_bar in zip(axes, alpha_bars, strict=True):
        noise = rng.normal(size=x0.shape)
        xt = np.sqrt(alpha_bar) * x0 + np.sqrt(1 - alpha_bar) * noise
        ax.scatter(
            xt[:, 0], xt[:, 1], s=2.2, alpha=0.22, color="#4C78A8", rasterized=True
        )
        ax.set(xlim=(-4, 4), ylim=(-4, 4), xlabel="$x_1$")
        _style_axis(ax, rf"$\bar{{\alpha}}_t={alpha_bar:g}$", equal=True)
    axes[0].set_ylabel("$x_2$")
    fig.suptitle(
        r"Closed-form perturbation: $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$"
    )
    _finish(fig, "forward_diffusion.png")


def forward_diffusion_paths() -> None:
    rng = np.random.default_rng(SEED + 14)
    steps, n = 90, 24
    beta = np.linspace(0.002, 0.075, steps)
    alpha = 1 - beta
    x = sample_target(rng, n)
    history = [x.copy()]
    for step in range(steps):
        x = np.sqrt(alpha[step]) * x + np.sqrt(beta[step]) * rng.normal(size=x.shape)
        history.append(x.copy())
    history_array = np.stack(history)

    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    for path in history_array.transpose(1, 0, 2):
        ax.plot(path[:, 0], path[:, 1], color="#888888", alpha=0.5, linewidth=0.8)
    ax.scatter(
        history_array[0, :, 0],
        history_array[0, :, 1],
        s=22,
        color="#E45756",
        label="$x_0$",
    )
    ax.scatter(
        history_array[-1, :, 0],
        history_array[-1, :, 1],
        marker="x",
        s=25,
        color="#4C78A8",
        label="$x_T$",
    )
    ax.set(xlim=(-4, 4), ylim=(-4, 4), xlabel="$x_1$", ylabel="$x_2$")
    _style_axis(
        ax,
        "Forward Markov trajectories: signal shrinks while noise accumulates",
        equal=True,
    )
    ax.legend(frameon=False)
    _finish(fig, "forward_diffusion_paths.png")


def ddpm_information_structure() -> None:
    """Draw the training/generation information gap without decorative elements."""
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 4.8))
    rows = [
        (
            axes[0],
            "training",
            [
                (0.08, "$x_0$\nobserved"),
                (0.30, r"$t,\epsilon$" + "\nsampled"),
                (0.53, "$x_t$\nconstructed"),
                (0.76, r"$\epsilon_\theta(x_t,t)$" + "\npredicted"),
            ],
            ["known", "known", "known", "learned"],
        ),
        (
            axes[1],
            "generation",
            [
                (0.08, "$x_T$\nsampled"),
                (0.30, "$x_t$\navailable"),
                (0.53, "$x_0$\nmissing"),
                (0.76, r"$p_\theta(x_{t-1}\mid x_t)$" + "\nconstructed"),
            ],
            ["known", "known", "missing", "learned"],
        ),
    ]
    colors = {"known": "#DCEAF7", "missing": "#FADBD8", "learned": "#DDEED8"}
    for ax, label, boxes, states in rows:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.text(0.01, 0.86, label, fontsize=11, weight="bold")
        for (x_position, text_value), state in zip(boxes, states, strict=True):
            ax.text(
                x_position,
                0.45,
                text_value,
                ha="center",
                va="center",
                fontsize=10,
                bbox={
                    "boxstyle": "round,pad=0.35",
                    "fc": colors[state],
                    "ec": "#555555",
                },
            )
        for left, right in pairwise(boxes):
            ax.annotate(
                "",
                xy=(right[0] - 0.08, 0.45),
                xytext=(left[0] + 0.08, 0.45),
                arrowprops={"arrowstyle": "->", "color": "#555555"},
            )
    fig.suptitle(
        "The denoiser learns to replace information available only during training"
    )
    _finish(fig, "ddpm_information_structure.png")


def _ddim_trajectory(
    initial: np.ndarray,
    alpha_bars: np.ndarray,
    eta: float,
    rng: np.random.Generator,
    selected_timesteps: np.ndarray | None = None,
) -> np.ndarray:
    """Sample a selected DDIM grid with an analytic epsilon predictor.

    ``alpha_bars`` includes the clean endpoint at index zero, where its value
    is exactly one. ``selected_timesteps`` is an increasing mathematical-time
    subsequence that includes both endpoints. Its reverse pairs determine the
    denoiser evaluations and transitions.
    """

    if selected_timesteps is None:
        selected_timesteps = np.arange(len(alpha_bars))
    if selected_timesteps.ndim != 1 or len(selected_timesteps) < 2:
        raise ValueError("selected_timesteps must contain at least two entries")
    if not np.issubdtype(selected_timesteps.dtype, np.integer):
        raise ValueError("selected_timesteps must contain integer indices")
    if selected_timesteps[0] != 0 or selected_timesteps[-1] != len(alpha_bars) - 1:
        raise ValueError(
            "selected_timesteps must include the clean and noisy endpoints"
        )
    if np.any(np.diff(selected_timesteps) <= 0):
        raise ValueError("selected_timesteps must be strictly increasing")

    x = initial.copy()
    history = [x.copy()]
    for timestep, previous_timestep in zip(
        selected_timesteps[:0:-1], selected_timesteps[-2::-1], strict=True
    ):
        alpha_t = float(alpha_bars[int(timestep)])
        alpha_prev = float(alpha_bars[int(previous_timestep)])
        epsilon = _epsilon_oracle(x, alpha_t)
        x0_hat = (x - np.sqrt(1 - alpha_t) * epsilon) / np.sqrt(alpha_t)
        sigma = eta * np.sqrt(
            (1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)
        )
        direction_scale = np.sqrt(max(1 - alpha_prev - sigma**2, 0.0))
        x = (
            np.sqrt(alpha_prev) * x0_hat
            + direction_scale * epsilon
            + sigma * rng.normal(size=x.shape)
        )
        history.append(x.copy())
    return np.stack(history)


def _toy_alpha_bars(num_train_steps: int) -> np.ndarray:
    """Create one fixed toy alpha-bar table with an explicit clean endpoint."""

    return np.linspace(1.0, 0.012, num_train_steps + 1)


def ddpm_reverse() -> None:
    rng = np.random.default_rng(SEED + 5)
    alpha_bars = _toy_alpha_bars(90)
    initial = rng.normal(size=(2_500, 2))
    trajectory = _ddim_trajectory(initial, alpha_bars, eta=1.0, rng=rng)
    indices = [0, 30, 60, 90]
    fig, axes = plt.subplots(1, 4, figsize=(12.2, 3.0), sharex=True, sharey=True)
    for ax, index in zip(axes, indices, strict=True):
        points = trajectory[index]
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=2.2,
            alpha=0.24,
            color="#E45756",
            rasterized=True,
        )
        ax.set(xlim=(-4, 4), ylim=(-4, 4), xlabel="$x_1$")
        _style_axis(ax, f"reverse step {index}", equal=True)
    axes[0].set_ylabel("$x_2$")
    fig.suptitle("DDPM-style reverse chain with an analytic oracle denoiser")
    _finish(fig, "ddpm_reverse.png")


def noise_score_relation() -> None:
    rng = np.random.default_rng(SEED + 6)
    x0 = sample_target(rng, 4_000)
    alpha_bars = [0.88, 0.52, 0.16]
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.5), sharex=True, sharey=True)
    for ax, alpha_bar in zip(axes, alpha_bars, strict=True):
        xt = np.sqrt(alpha_bar) * x0 + np.sqrt(1 - alpha_bar) * rng.normal(
            size=x0.shape
        )
        grid = np.linspace(-3.2, 3.2, 16)
        xx, yy = np.meshgrid(grid, grid)
        points = np.column_stack([xx.ravel(), yy.ravel()])
        epsilon = _epsilon_oracle(points, alpha_bar)
        score = -epsilon / np.sqrt(1 - alpha_bar)
        magnitude = np.linalg.norm(score, axis=1) + 1e-8
        ax.scatter(
            xt[:, 0], xt[:, 1], s=1.5, alpha=0.08, color="#4C78A8", rasterized=True
        )
        ax.quiver(
            points[:, 0],
            points[:, 1],
            score[:, 0] / magnitude,
            score[:, 1] / magnitude,
            magnitude,
            cmap="magma",
            alpha=0.72,
        )
        ax.set(xlim=(-3.4, 3.4), ylim=(-3.4, 3.4), xlabel="$x_1$")
        _style_axis(ax, rf"$\bar{{\alpha}}_t={alpha_bar}$", equal=True)
    axes[0].set_ylabel("$x_2$")
    fig.suptitle(
        r"Noise prediction and score: $s_t(x)=-\mathbb{E}[\epsilon\mid x_t=x]/\sqrt{1-\bar\alpha_t}$"
    )
    _finish(fig, "noise_score_relation.png")


def ddim_paths() -> None:
    initial_rng = np.random.default_rng(SEED + 7)
    initial = initial_rng.normal(size=(16, 2))
    alpha_bars = _toy_alpha_bars(25)
    settings = [
        (1.0, r"DDPM member: $\eta=1$"),
        (0.35, r"stochastic DDIM: $\eta=.35$"),
        (0.0, r"deterministic DDIM: $\eta=0$"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), sharex=True, sharey=True)
    for panel, (eta, title) in enumerate(settings):
        rng = np.random.default_rng(SEED + 80 + panel)
        history = _ddim_trajectory(initial, alpha_bars, eta=eta, rng=rng)
        ax = axes[panel]
        for sample in range(len(initial)):
            path = history[:, sample]
            ax.plot(path[:, 0], path[:, 1], color="#888888", linewidth=0.85, alpha=0.7)
        ax.scatter(
            initial[:, 0],
            initial[:, 1],
            marker="x",
            s=24,
            color="#4C78A8",
            label="$x_T$",
        )
        final = history[-1]
        ax.scatter(final[:, 0], final[:, 1], s=24, color="#E45756", label="$x_0$")
        ax.set(xlim=(-3.1, 3.1), ylim=(-2.7, 2.8), xlabel="$x_1$")
        _style_axis(ax, title, equal=True)
    axes[0].set_ylabel("$x_2$")
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle(
        "One trained marginal denoiser supports a family of reverse trajectories"
    )
    _finish(fig, "ddim_paths.png")


def ddim_randomness_sources() -> None:
    """Separate reverse-path randomness from initial-latent randomness."""
    alpha_bars = _toy_alpha_bars(37)
    fixed = np.array([[0.4, -0.7]])
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.0), sharex=True, sharey=True)
    for repeat in range(18):
        path = _ddim_trajectory(
            fixed,
            alpha_bars,
            eta=0.65,
            rng=np.random.default_rng(SEED + 130 + repeat),
        )[:, 0]
        axes[0].plot(path[:, 0], path[:, 1], color="#4C78A8", alpha=0.45, linewidth=0.8)
    axes[0].scatter(*fixed[0], marker="x", color="black", s=28)

    initial = np.random.default_rng(SEED + 150).normal(size=(18, 2))
    paths = _ddim_trajectory(
        initial, alpha_bars, eta=0.0, rng=np.random.default_rng(SEED + 151)
    )
    for path in paths.transpose(1, 0, 2):
        axes[1].plot(path[:, 0], path[:, 1], color="#E45756", alpha=0.55, linewidth=0.8)
    axes[1].scatter(initial[:, 0], initial[:, 1], marker="x", color="black", s=20)
    for ax, title in zip(
        axes,
        [
            "fixed $x_T$, stochastic reverse transitions",
            "random $x_T$, deterministic transitions",
        ],
        strict=True,
    ):
        ax.set(xlim=(-3.2, 3.2), ylim=(-2.8, 2.8), xlabel="$x_1$")
        _style_axis(ax, title, equal=True)
    axes[0].set_ylabel("$x_2$")
    fig.suptitle("Two distinct sources of generative randomness")
    _finish(fig, "ddim_randomness_sources.png")


def ddim_step_skipping() -> None:
    """Measure endpoint discrepancy on subgrids of one fixed noise schedule."""
    rng = np.random.default_rng(SEED + 16)
    target = sample_target(rng, 7_000)
    directions = rng.normal(size=(36, 2))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    def sliced_wasserstein(samples: np.ndarray) -> float:
        distances = [
            wasserstein_distance(samples @ direction, target @ direction)
            for direction in directions
        ]
        return float(np.mean(distances))

    num_train_steps = 90
    alpha_bars = _toy_alpha_bars(num_train_steps)
    initial = rng.normal(size=(7_000, 2))
    step_counts = np.array([6, 10, 18, 32, 56, 90])
    errors = {0.0: [], 1.0: []}
    for eta, eta_errors in errors.items():
        for count in step_counts:
            selected_timesteps = np.linspace(0, num_train_steps, count + 1, dtype=int)
            generated = _ddim_trajectory(
                initial,
                alpha_bars,
                eta=eta,
                rng=np.random.default_rng(SEED + int(eta * 1_000) + int(count)),
                selected_timesteps=selected_timesteps,
            )[-1]
            eta_errors.append(sliced_wasserstein(generated))

    fig, ax = plt.subplots(figsize=(6.2, 4.1))
    ax.plot(
        step_counts,
        errors[0.0],
        "o-",
        label=r"deterministic, $\eta=0$",
        color="#E45756",
    )
    ax.plot(
        step_counts, errors[1.0], "o-", label=r"stochastic, $\eta=1$", color="#4C78A8"
    )
    ax.set(
        xlabel="reverse steps / denoiser evaluations",
        ylabel="sliced Wasserstein (lower is better)",
    )
    _style_axis(ax, "Skipping trades computation for model/discretization error")
    ax.legend(frameon=False)
    _finish(fig, "ddim_step_skipping.png")


def probability_flow() -> None:
    """Compare stochastic reverse paths with their deterministic family member."""
    rng = np.random.default_rng(SEED + 8)
    initial = rng.normal(size=(22, 2))
    alpha_bars = _toy_alpha_bars(35)
    stochastic = _ddim_trajectory(
        initial, alpha_bars, eta=1.0, rng=np.random.default_rng(SEED + 9)
    )
    deterministic = _ddim_trajectory(
        initial, alpha_bars, eta=0.0, rng=np.random.default_rng(SEED + 10)
    )
    fig, axes = plt.subplots(1, 2, figsize=(8.3, 4.0), sharex=True, sharey=True)
    for ax, paths, title in [
        (axes[0], stochastic, "reverse stochastic dynamics"),
        (axes[1], deterministic, "deterministic probability-flow analogue"),
    ]:
        for path in paths.transpose(1, 0, 2):
            ax.plot(path[:, 0], path[:, 1], color="#777777", alpha=0.62, linewidth=0.8)
        ax.scatter(paths[0, :, 0], paths[0, :, 1], marker="x", color="#4C78A8", s=20)
        ax.scatter(paths[-1, :, 0], paths[-1, :, 1], color="#E45756", s=20)
        ax.set(xlim=(-3.2, 3.2), ylim=(-2.8, 2.8), xlabel="$x_1$")
        _style_axis(ax, title, equal=True)
    axes[0].set_ylabel("$x_2$")
    fig.suptitle("Path laws differ even when the intended marginal evolution agrees")
    _finish(fig, "probability_flow.png")


def main() -> None:
    brownian_motion()
    sde_evolution()
    ode_vector_field()
    ode_vs_sde_fixed_initial()
    flow_matching_paths()
    flow_conditional_marginal_field()
    same_marginals_different_joint()
    score_field()
    probability_flow()
    forward_diffusion()
    forward_diffusion_paths()
    ddpm_information_structure()
    ddpm_reverse()
    noise_score_relation()
    ddim_paths()
    ddim_randomness_sources()
    ddim_step_skipping()


if __name__ == "__main__":
    main()
