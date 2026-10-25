"""Variational Quantum Classification Benchmark Module.

This module provides an end-to-end implementation of a Variational Quantum Classifier (VQC)
using PennyLane and PyTorch. It features automated circuit visualization (qml.draw_mpl),
parameter-shift differentiation, native PennyLane optimizers, and non-linear quantum
decision boundary contour mapping.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import numpy as pnp
import torch


@dataclass
class VQCConfig:
    """Hyperparameter and architectural specifications for the VQC."""
    n_qubits: int = 2
    n_layers: int = 3
    n_samples_per_class: int = 40
    learning_rate: float = 0.08
    training_steps: int = 30
    cluster_std: float = 0.7
    centroid_0: Tuple[float, float] = (-1.0, 1.0)
    centroid_1: Tuple[float, float] = (1.0, -1.0)
    random_seed: int = 42


class QuantumDataPipeline:
    """Generates synthetic classification data and manages framework conversions."""

    def __init__(self, config: VQCConfig) -> None:
        self.config = config
        pnp.random.seed(self.config.random_seed)
        self.x, self.y = self._generate_data()

    def _generate_data(self) -> Tuple[pnp.ndarray, pnp.ndarray]:
        n = self.config.n_samples_per_class
        x0 = pnp.random.normal(loc=self.config.centroid_0, scale=self.config.cluster_std, size=(n, 2))
        x1 = pnp.random.normal(loc=self.config.centroid_1, scale=self.config.cluster_std, size=(n, 2))
        features = pnp.concatenate([x0, x1], axis=0)
        labels = pnp.concatenate([-pnp.ones(n), pnp.ones(n)], axis=0)
        return features, labels

    def get_torch_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.x, dtype=torch.float32),
            torch.tensor(self.y, dtype=torch.float32).unsqueeze(1),
        )


class VariationalQuantumClassifier:
    """Encapsulates the PennyLane QNode, parameters, and loss evaluation."""

    def __init__(self, config: VQCConfig) -> None:
        self.config = config
        self.dev = qml.device("default.qubit", wires=self.config.n_qubits)
        self._build_qnodes()

        # Initialize parameters
        pnp.random.seed(self.config.random_seed)
        self.w_init = 0.1 * pnp.random.randn(self.config.n_layers, self.config.n_qubits, requires_grad=True)
        self.b_init = pnp.array(0.0, requires_grad=True)

    def _build_qnodes(self) -> None:
        """Constructs QNodes for both PennyLane autograd and PyTorch interfaces."""
        
        @qml.qnode(self.dev, interface="autograd", diff_method="parameter-shift")
        def _circuit_autograd(x: pnp.ndarray, w: pnp.ndarray) -> float:
            qml.AngleEmbedding(x, wires=range(self.config.n_qubits), rotation="X")
            qml.BasicEntanglerLayers(w, wires=range(self.config.n_qubits), rotation=qml.RX)
            return qml.expval(qml.PauliZ(0))

        @qml.qnode(self.dev, interface="torch", diff_method="parameter-shift")
        def _circuit_torch(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            qml.AngleEmbedding(x, wires=range(self.config.n_qubits), rotation="X")
            qml.BasicEntanglerLayers(w, wires=range(self.config.n_qubits), rotation=qml.RX)
            return qml.expval(qml.PauliZ(0))

        self.circuit_autograd = _circuit_autograd
        self.circuit_torch = _circuit_torch

    def predict_sample(self, x: pnp.ndarray, w: pnp.ndarray, b: float) -> float:
        return self.circuit_autograd(x, w) + b

    def cost_function(
        self, params: Tuple[pnp.ndarray, pnp.ndarray], x: pnp.ndarray, y: pnp.ndarray
    ) -> float:
        """Computes batch MSE loss over the dataset."""
        w, b = params
        predictions = pnp.array([self.circuit_autograd(xi, w) + b for xi in x])
        return pnp.mean((predictions - y) ** 2)


class VQCOptimizationEngines:
    """Provides comparative training routines: native PennyLane optimizer vs. PyTorch Adam."""

    @staticmethod
    def train_pennylane_opt(
        model: VariationalQuantumClassifier,
        x: pnp.ndarray,
        y: pnp.ndarray,
    ) -> Tuple[pnp.ndarray, float, List[float]]:
        """Optimizes weights and bias using PennyLane's GradientDescentOptimizer."""
        w = pnp.copy(model.w_init)
        b = pnp.copy(model.b_init)
        optimizer = qml.GradientDescentOptimizer(stepsize=model.config.learning_rate)

        loss_history: List[float] = []

        for step in range(model.config.training_steps):
            (w, b), cost = optimizer.step_and_cost(
                lambda weights, bias: model.cost_function((weights, bias), x, y), w, b
            )
            loss_history.append(float(cost))
            if (step + 1) % 10 == 0 or step == 0:
                print(f"[PennyLane QGD] Step {step+1:02d} | MSE Loss: {cost:.6f}")

        return w, float(b), loss_history

    @staticmethod
    def train_torch_adam(
        model: VariationalQuantumClassifier,
        x_torch: torch.Tensor,
        y_torch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[float]]:
        """Optimizes the quantum model using PyTorch's Adam optimizer."""
        torch.manual_seed(model.config.random_seed)
        w = torch.nn.Parameter(torch.tensor(model.w_init, dtype=torch.float32, requires_grad=True))
        b = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32, requires_grad=True))
        optimizer = torch.optim.Adam([w, b], lr=model.config.learning_rate)
        criterion = torch.nn.MSELoss()

        loss_history: List[float] = []

        for step in range(model.config.training_steps):
            optimizer.zero_grad()
            # Predict sequentially or via vectorized qnode mapping
            preds = torch.stack([model.circuit_torch(xi, w) + b for xi in x_torch]).unsqueeze(1)
            loss = criterion(preds, y_torch)
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            loss_history.append(loss_val)
            if (step + 1) % 10 == 0 or step == 0:
                print(f"[PyTorch Adam]  Step {step+1:02d} | MSE Loss: {loss_val:.6f}")

        return w.detach(), b.detach(), loss_history


class QuantumVisualizer:
    """Exports circuit schematics, convergence curves, and non-linear decision regions."""

    def __init__(self, output_dir: str = "visualizations") -> None:
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def draw_circuit(self, model: VariationalQuantumClassifier) -> None:
        """Saves a publication-quality circuit schematic using matplotlib."""
        sample_x = pnp.array([0.5, -0.5])
        sample_w = model.w_init

        fig, ax = qml.draw_mpl(model.circuit_autograd)(sample_x, sample_w)
        fig.suptitle("Variational Quantum Classifier Architecture", fontsize=14, y=1.02)
        save_path = os.path.join(self.output_dir, "vqc_circuit.png")
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"Saved circuit diagram to {save_path}")

    def plot_benchmark_results(
        self,
        x: pnp.ndarray,
        y: pnp.ndarray,
        w_opt: pnp.ndarray,
        b_opt: float,
        model: VariationalQuantumClassifier,
        loss_history_pnl: List[float],
        loss_history_torch: List[float],
    ) -> None:
        """Plots training dynamics alongside the non-linear quantum decision boundary."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Panel 1: Loss curves
        axes[0].plot(loss_history_pnl, label="PennyLane GradientDescent", color="#1f77b4", lw=2)
        axes[0].plot(loss_history_torch, label="PyTorch Adam", color="#ff7f0e", lw=2, linestyle="--")
        axes[0].set_xlabel("Optimization Step")
        axes[0].set_ylabel("Mean Squared Error")
        axes[0].set_title("VQC Training Convergence")
        axes[0].grid(True, linestyle="--", alpha=0.5)
        axes[0].legend()

        # Panel 2: Continuous 2D quantum decision space
        grid_points = 35
        x_min, x_max = x[:, 0].min() - 0.8, x[:, 0].max() + 0.8
        y_min, y_max = x[:, 1].min() - 0.8, x[:, 1].max() + 0.8
        xx, yy = pnp.meshgrid(pnp.linspace(x_min, x_max, grid_points), pnp.linspace(y_min, y_max, grid_points))

        grid_inputs = pnp.c_[xx.ravel(), yy.ravel()]
        grid_preds = pnp.array([model.predict_sample(point, w_opt, b_opt) for point in grid_inputs])
        grid_preds = grid_preds.reshape(xx.shape)

        contour = axes[1].contourf(xx, yy, grid_preds, levels=20, cmap="coolwarm", alpha=0.7)
        fig.colorbar(contour, ax=axes[1], label=r"Prediction $f(\mathbf{x}) = \langle Z_0 \rangle + b$")

        # Zero-level decision contour: f(x) = 0
        axes[1].contour(xx, yy, grid_preds, levels=[0.0], colors="black", linewidths=2)

        # Scatter data points
        axes[1].scatter(x[y == -1, 0], x[y == -1, 1], color="blue", edgecolors="k", label="Class -1")
        axes[1].scatter(x[y == 1, 0], x[y == 1, 1], color="red", edgecolors="k", label="Class +1")
        axes[1].set_xlabel(r"$x_0$")
        axes[1].set_ylabel(r"$x_1$")
        axes[1].set_title("Quantum Decision Boundary & Prediction Landscape")
        axes[1].legend()

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, "vqc_classification_results.png")
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"Saved benchmark diagnostics to {save_path}")


def main() -> None:
    config = VQCConfig(
        n_qubits=2,
        n_layers=3,
        n_samples_per_class=35,
        learning_rate=0.08,
        training_steps=30,
        cluster_std=0.75,
    )

    # 1. Pipeline initialization
    pipeline = QuantumDataPipeline(config)
    vqc = VariationalQuantumClassifier(config)
    visualizer = QuantumVisualizer()

    # 2. Export circuit diagram
    visualizer.draw_circuit(vqc)

    # 3. Train via PennyLane native optimizer
    print("\n--- Running PennyLane GradientDescentOptimizer ---")
    w_pnl, b_pnl, loss_pnl = VQCOptimizationEngines.train_pennylane_opt(
        vqc, pipeline.x, pipeline.y
    )

    # 4. Train via PyTorch Adam
    print("\n--- Running PyTorch Adam Optimizer ---")
    x_torch, y_torch = pipeline.get_torch_tensors()
    w_torch, b_torch, loss_torch = VQCOptimizationEngines.train_torch_adam(
        vqc, x_torch, y_torch
    )

    # 5. Export comprehensive visualization panel
    visualizer.plot_benchmark_results(
        x=pipeline.x,
        y=pipeline.y,
        w_opt=w_pnl,
        b_opt=b_pnl,
        model=vqc,
        loss_history_pnl=loss_pnl,
        loss_history_torch=loss_torch,
    )


if __name__ == "__main__":
    main()
