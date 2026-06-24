# Saving and Loading KoopmanDL Prediction Weights

This explains how to save a trained `KoopmanDLSolver`'s weights and load them back to predict **without retraining**.

Companion files:

- `koopman_io.py` — provides `save_dl_solver` / `load_dl_solver`;
- `weights/duffing_dl_weights.npz`, `weights/vdp_dl_weights.npz` — pre-trained example weights (Duffing 3000 epochs, VdP 2000 epochs).

## What gets saved

A prediction only needs the trained dictionary network (`solver.dic`) plus `solver.eigenvalues`, `solver.eigenvectors`, and `solver.modes`. `save_dl_solver` bundles all of these (with metadata like `layer_sizes` and `n_psi_train`) into a single `.npz` file. Save/load is lossless — reloaded predictions match the originals exactly.

## Save after training

```python
from koopman_io import save_dl_solver

# layer_sizes / n_psi_train must match what you used to build PsiNN
save_dl_solver(
    solver,
    path="weights/duffing_dl_weights.npz",
    layer_sizes=[100, 100, 100],
    n_psi_train=22,
)
```

For vdp_demo, use `layer_sizes=[200, 200, 200], n_psi_train=40` and the matching path.

## Load and predict

```python
from koopman_io import load_dl_solver

solver = load_dl_solver("weights/duffing_dl_weights.npz")

x_pred = solver.predict(x0, traj_len=50)   # ready to predict, no rebuild needed
efunc = solver.eigenfunctions(x0)
```

## Retraining the model

The notebooks load the saved weights by default and skip the training cell. To retrain from scratch, turn off the skip-execution flag on the training cell and run it; then uncomment the `save_dl_solver(...)` cell to overwrite the weights in `weights/`.
