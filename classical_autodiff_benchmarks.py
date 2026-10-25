"""Classical Machine Learning and Automatic Differentiation Benchmarks.

This module provides an end-to-end framework demonstrating gradient-based
optimization via PennyLane's Autograd engine and PyTorch. It implements linear,
affine, and shallow non-linear neural network models on synthetic Gaussian
mixtures with accompanying landscape and decision boundary visualization utilities.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import numpy as pnp
import torch


@dataclass
class DatasetConfig:
    """Configuration parameters for synthetic Gaussian clustering data."""
    n_samples_per_class: int = 50
    centroid_class_0: Tuple[float, float] = (-1.0, 1.0)
    centroid_class_1: Tuple[float, float] = (1.0, -1.0)
    cluster_std: float = 0.8
    random_seed: int = 42


class SyntheticClassificationDataset:
    """Generates and manages linearly separable Gaussian mixture benchmarks."""

    def __init__(self, config: DatasetConfig = DatasetConfig()) -> None:
        self.config = config
        pnp.random.seed(self.config.random_seed)
        self.features, self.labels = self._generate_data()

    def _generate_data(self) -> Tuple[pnp.ndarray, pnp.ndarray]:
        n = self.config.n_samples_per_class
        x0 = pnp.random.normal(
            loc=self.config.centroid_class_0,
            scale=self.config.cluster_std,
            size=(n, 2),
        )
        x1 = pnp.random.normal(
            loc=self.config.centroid_class_1,
            scale=self.config.cluster_std,
            size=(n, 2),
        )

        features = pnp.concatenate([x0, x1], axis=0)
        labels = pnp.concatenate([-pnp.ones(n), pnp.ones(n)], axis=0)
        return features, labels

    def get_torch_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert underlying arrays into floating-point PyTorch tensors."""
        x_tensor = torch.tensor(self.features, dtype=torch.float32)
        y_tensor = torch.tensor(self.labels, dtype=torch.float32).unsqueeze(1)
        return x_tensor, y_tensor


class ClassicalDifferentiableModels:
    """Collection of vectorized analytical architectures under PennyLane Autograd."""

    @staticmethod
    def linear_forward(x: pnp.ndarray, w: pnp.ndarray) -> pnp.ndarray:
        r"""Compute f(x) = \langle x, w \rangle."""
        return pnp.dot(x, w)

    @staticmethod
    def affine_forward(x: pnp.ndarray, w: pnp.ndarray, b: float) -> pnp.ndarray:
        r"""Compute f(x) = \langle x, w \rangle + b."""
        return pnp.dot(x, w) + b

    @staticmethod
    def shallow_mlp_forward(
        x: pnp.ndarray, w_hidden: pnp.ndarray, w_out: pnp.ndarray
    ) -> pnp.ndarray:
        r"""Compute f(x) = \langle w_{out}, \sigma(W_{hidden} x) \rangle."""
        z = pnp.dot(x, w_hidden.T)
        activation = 1.0 / (1.0 + pnp.exp(-z))
        return pnp.dot(activation, w_out)


class LossObjectives:
    """Vectorized Mean Squared Error objectives for parameter optimization."""

    @staticmethod
    def mse_linear(w: pnp.ndarray, x: pnp.ndarray, y: pnp.ndarray) -> float:
        predictions = ClassicalDifferentiableModels.linear_forward(x, w)
        return pnp.mean((predictions - y) ** 2)

    @staticmethod
    def mse_affine(
        params: pnp.ndarray, x: pnp.ndarray, y: pnp.ndarray
    ) -> float:
        w = params[:2]
        b = params[2]
        predictions = ClassicalDifferentiableModels.affine_forward(x, w, b)
        return pnp.mean((predictions - y) ** 2)

    @staticmethod
    def mse_mlp(
        w_hidden: pnp.ndarray,
        w_out: pnp.ndarray,
        x: pnp.ndarray,
        y: pnp.ndarray,
    ) -> float:
        predictions = ClassicalDifferentiableModels.shallow_mlp_forward(x, w_hidden, w_out)
        return pnp.mean((predictions - y) ** 2)


class OptimizationEngine:
    """Manages gradient descent loops and historical trajectory tracking."""

    @staticmethod
    def train_affine_autograd(
        x: pnp.ndarray,
        y: pnp.ndarray,
        learning_rate: float = 0.05,
        steps: int = 40,
    ) -> Tuple[pnp.ndarray, float, List[float], List[pnp.ndarray]]:
        """Optimizes weights and scalar bias jointly using PennyLane Autograd."""
        # params layout: [w0, w1, b]
        init_params = pnp.array([0.1, -0.2, 0.0], requires_grad=True)
        grad_fn = qml.grad(LossObjectives.mse_affine, argnums=0)

        params = init_params
        loss_history: List[float] = []
        param_history: List[pnp.ndarray] = []

        for _ in range(steps):
            current_loss = LossObjectives.mse_affine(params, x, y)
            loss_history.append(float(current_loss))
            param_history.append(pnp.copy(params))

            gradient = grad_fn(params, x, y)
            params = params - learning_rate * gradient

        w_opt = params[:2]
        b_opt = float(params[2])
        return w_opt, b_opt, loss_history, param_history

    @staticmethod
    def train_torch_model(
        x: torch.Tensor,
        y: torch.Tensor,
        learning_rate: float = 0.05,
        steps: int = 40,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[float]]:
        """Optimizes a linear model via PyTorch autograd engine and Adam optimizer."""
        w = torch.randn(2, 1, requires_grad=True, dtype=torch.float32)
        b = torch.zeros(1, requires_grad=True, dtype=torch.float32)
        optimizer = torch.optim.Adam([w, b], lr=learning_rate)
        criterion = torch.nn.MSELoss()

        loss_history: List[float] = []

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            predictions = torch.mm(x, w) + b
            loss = criterion(predictions, y)
            loss.backward()
            return loss

        for _ in range(steps):
            loss_val = optimizer.step(closure)
            loss_history.append(loss_val.item())

        return w.detach(), b.detach(), loss_history


class MachineLearningVisualizer:
    """Visualization suite for trajectories, boundaries, and loss profiles."""

    def __init__(self, output_dir: str = "visualizations") -> None:
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_optimization_dynamics(
        self,
        x: pnp.ndarray,
        y: pnp.ndarray,
        w_opt: pnp.ndarray,
        b_opt: float,
        loss_history: List[float],
        param_history: List[pnp.ndarray],
    ) -> None:
        """Renders loss convergence, parameter trajectory, and decision landscape."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Subplot 1: Convergence profile
        axes[0].plot(loss_history, marker="o", color="#1f77b4", lw=2)
        axes[0].set_xlabel("Optimization Step")
        axes[0].set_ylabel("Empirical MSE Loss")
        axes[0].set_title("Loss Convergence Curve")
        axes[0].grid(True, linestyle="--", alpha=0.6)

        # Subplot 2: Trajectory on weight loss-surface (holding b fixed)
        w1_range = pnp.linspace(-1.5, 1.5, 60)
        w2_range = pnp.linspace(-1.5, 1.5, 60)
        grid_loss = pnp.zeros((len(w1_range), len(w2_range)))

        for i, w1 in enumerate(w1_range):
            for j, w2 in enumerate(w2_range):
                dummy_params = pnp.array([w1, w2, b_opt])
                grid_loss[j, i] = LossObjectives.mse_affine(dummy_params, x, y)

        cs = axes[1].contourf(w1_range, w2_range, grid_loss, levels=25, cmap="viridis")
        fig.colorbar(cs, ax=axes[1], label=r"$\mathcal{L}(w_1, w_2 \mid b_{opt})$")

        history_array = pnp.array(param_history)
        axes[1].plot(
            history_array[:, 0],
            history_array[:, 1],
            "-o",
            color="red",
            label="Gradient Steps",
            markersize=3,
        )
        axes[1].set_xlabel(r"$w_1$")
        axes[1].set_ylabel(r"$w_2$")
        axes[1].set_title(r"Parameter Trajectory Across Loss Surface")
        axes[1].legend()

        # Subplot 3: Final decision boundary and data projection
        idx_c0 = pnp.where(y == -1)[0]
        idx_c1 = pnp.where(y == 1)[0]

        axes[2].scatter(x[idx_c0, 0], x[idx_c0, 1], color="#2ca02c", label=r"Class $-1$", alpha=0.8)
        axes[2].scatter(x[idx_c1, 0], x[idx_c1, 1], color="#d62728", label=r"Class $+1$", alpha=0.8)

        # Compute hyperplane: w0*x + w1*y + b = 0 => y = -(w0*x + b) / w1
        x_domain = pnp.linspace(float(pnp.min(x[:, 0])) - 0.5, float(pnp.max(x[:, 0])) + 0.5, 100)
        if abs(w_opt[1]) > 1e-5:
            boundary_y = -(w_opt[0] * x_domain + b_opt) / w_opt[1]
            axes[2].plot(x_domain, boundary_y, "k-", lw=2, label="Decision Boundary")

        # Plot normal vector from origin
        axes[2].quiver(
            0,
            0,
            w_opt[0],
            w_opt[1],
            angles="xy",
            scale_units="xy",
            scale=1,
            color="blue",
            label=r"Weight Vector $\mathbf{w}$",
        )

        axes[2].set_xlabel(r"$x_1$")
        axes[2].set_ylabel(r"$x_2$")
        axes[2].set_title("Input Space & Hyperplane")
        axes[2].legend(loc="upper right")
        axes[2].grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, "optimization_summary.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Generated optimization report at {save_path}")


def main() -> None:
    # 1. Dataset Generation
    config = DatasetConfig(n_samples_per_class=60, cluster_std=0.7)
    dataset = SyntheticClassificationDataset(config)
    x, y = dataset.features, dataset.labels

    # 2. Optimization using PennyLane Autograd
    print("Executing PennyLane gradient descent optimization...")
    w_opt, b_opt, loss_hist, param_hist = OptimizationEngine.train_affine_autograd(
        x=x, y=y, learning_rate=0.08, steps=50
    )
    print(f"Optimal Parameters: w = {w_opt}, b = {b_opt:.4f}")
    print(f"Final Autograd Loss: {loss_hist[-1]:.6f}")

    # 3. Optimization using PyTorch Autograd Engine
    print("\nExecuting PyTorch Adam optimization...")
    x_t, y_t = dataset.get_torch_tensors()
    w_t, b_t, torch_loss_hist = OptimizationEngine.train_torch_model(
        x_t, y_t, learning_rate=0.05, steps=50
    )
    print(f"Torch Optimal Parameters: w = {w_t.squeeze().tolist()}, b = {b_t.item():.4f}")
    print(f"Final PyTorch Loss: {torch_loss_hist[-1]:.6f}")

    # 4. Generate Diagnostics and Plots
    visualizer = MachineLearningVisualizer()
    visualizer.plot_optimization_dynamics(x, y, w_opt, b_opt, loss_hist, param_hist)


if __name__ == "__main__":
    main()
