# Differentiable Quantum-Classical Machine Learning

A modular benchmark suite illustrating the continuum from classical automatic differentiation to parameterized variational quantum algorithms using **PennyLane** and **PyTorch**.

---

## Theoretical Overview

```
Classical Pipeline:
x ∈ ℝ² ─────────────► [ Linear / MLP Layer ] ──────► ŷ = wᵀx + b ──────────► MSE Loss ──► Backprop (Autograd)

Quantum Pipeline (VQC):
x ∈ ℝ² ──► |x⟩ = ⨂ Rₓ(xⱼ)|0⟩ ──► [ U_ansatz(w) ] ──► ŷ = ⟨ψ|Z₀|ψ⟩ + b ──► MSE Loss ──► Parameter-Shift / Adam

```

### 1. Classical Autodiff (`classical_autodiff_benchmarks.py`)

Computes empirical risk minimization over a two-dimensional Gaussian mixture $y \in \{-1, +1\}$:

$$\min_{\mathbf{w}, b} \frac{1}{N} \sum_{i=1}^N \left( (\mathbf{w}^T \mathbf{x}_i + b) - y_i \right)^2$$

Evaluates the exact convex parameter landscape $\mathcal{L}(\mathbf{w} \mid b)$ and verifies convergence across:

* **PennyLane Autograd:** Analytical vector gradients via reverse-mode AD.
* **PyTorch Engine:** Closure-based gradient descent using the `torch.optim.Adam` optimizer.

### 2. Variational Quantum Classifier (`variational_quantum_classifier.py`)

Encodes classical features into an $n$-qubit Hilbert space $\mathcal{H} = (\mathbb{C}^2)^{\otimes n}$ via non-linear Pauli rotations:

$$\vert{}\mathbf{x}\rangle = \bigotimes_{j=0}^{n-1} R_X(x_j)\vert{}0\rangle$$

The state undergoes parameterized entanglement via $L$ layers of `BasicEntanglerLayers`:

$$U(\mathbf{w}) = \prod_{l=1}^L \left( \text{CNOT}_{0 \to 1} \prod_{j=0}^{n-1} R_X(w_{l, j}) \right)$$

Model predictions are obtained by measuring the expectation value of the Pauli-$Z$ observable on the first qubit:

$$f(\mathbf{x}; \mathbf{w}, b) = \langle \mathbf{x} \vert{} U^\dagger(\mathbf{w}) (Z \otimes I) U(\mathbf{w}) \vert{} \mathbf{x} \rangle + b$$

Analytical quantum gradients $\nabla_{\mathbf{w}} f$ are evaluated via the exact **parameter-shift rule**:

$$\frac{\partial f}{\partial w_k} = \frac{f\left(w_k + \frac{\pi}{2}\right) - f\left(w_k - \frac{\pi}{2}\right)}{2}$$

---

## Repository Structure

```
├── classical_autodiff_benchmarks.py    # Classical affine & MLP autodiff pipelines
├── variational_quantum_classifier.py   # Parameter-shift VQC & PyTorch quantum bridge
├── requirements.txt                    # Minimal environment dependencies
├── README.md                           # Technical documentation
└── visualizations/                     # Auto-generated diagnostics
    ├── optimization_summary.png        # Loss landscape, trajectory & linear hyperplane
    ├── vqc_circuit.png                 # Publication-grade QNode schematic
    └── vqc_classification_results.png  # Non-linear quantum decision contours & convergence

```

---

## Quickstart

### Environment Setup

```bash
git clone https://github.com/<your-username>/pennylane-classical-to-quantum.git
cd pennylane-classical-to-quantum

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

```

*`requirements.txt`:*

```text
pennylane>=0.35.0
torch>=2.0.0
matplotlib>=3.7.0
numpy>=1.24.0

```

### Running Locally

Execute the classical autodiff benchmark:

```bash
python classical_autodiff_benchmarks.py

```

Execute the variational quantum classifier:

```bash
python variational_quantum_classifier.py

```

### Running on Google Colab

Run either script directly inside a Colab notebook by adding the dependency installation to the top of the cell:

```python
!pip install pennylane torch matplotlib

```

To display figures inline instead of saving them to disk, replace `plt.close(fig)` with `plt.show()` inside the visualizer classes.

---

## Key Features

| Feature | `classical_autodiff_benchmarks.py` | `variational_quantum_classifier.py` |
| --- | --- | --- |
| **Model Type** | Affine Hyperplane / 1-Hidden MLP | Variational Quantum Circuit (VQC) |
| **State Space** | $\mathbf{x} \in \mathbb{R}^2$ | $\mathcal{H} = (\mathbb{C}^2)^{\otimes 2}$ |
| **Encoding** | Direct inner product $\langle \mathbf{w}, \mathbf{x} \rangle$ | Non-linear `AngleEmbedding` ($R_X$) |
| **Ansatz** | Vectorized matrix transformations | Entangling layers ($R_X + \text{CNOT}$) |
| **Differentiation** | Reverse-mode automatic differentiation | Parameter-shift rule (`diff_method="parameter-shift"`) |
| **Optimizers** | Autograd Vanilla GD / PyTorch Adam | PennyLane `GradientDescentOptimizer` / PyTorch Adam |
| **Diagnostics** | $\mathcal{L}(w_1, w_2)$ contour & trajectory | Circuit diagram (`qml.draw_mpl`) & 2D decision contours |
