# Numerical Heat Transfer: Analytical, FEM, and PINNs

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![LaTeX](https://img.shields.io/badge/LaTeX-008080?style=for-the-badge&logo=latex&logoColor=white)

A computational engineering project evaluating and comparing numerical solvers for 2D Heat Transfer equations using:
1.  **Analytical Methods**: Boundary value convergence evaluations.
2.  **Finite Element Method (FEM)**: Grid discretisation solutions.
3.  **Physics-Informed Neural Networks (PINNs)**: Neural solvers incorporating partial differential equations (PDEs) directly in the loss function.

## Key Features
*   **PINNs Solver in PyTorch**: Neural network training with spatial coordinates as inputs and temperature outputs matching boundary losses.
*   **FEM Numerical Solver**: NumPy/SciPy solver providing spatial temperature matrices.
*   **Comparison Visualisations**: Graph overlays showing accuracy residuals across the three paradigms.

## File Structure
```text
├── analytical.ipynb    # Boundary value solutions
├── fem.ipynb           # Finite Element Method numerical grid solver
├── pinns.ipynb         # PyTorch PINNs network and training pipeline
├── main.tex            # LaTeX project report source
├── main.pdf            # Compiled project report PDF
└── comparison_plot.png # Evaluation plot comparison
```
