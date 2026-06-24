"""Save / load helpers for a trained KoopmanDLSolver.

A trained KoopmanDL model needs four things to make predictions:

* the trainable dictionary ``PsiNN`` (the neural-network weights), and
* the linear pieces computed from the Koopman matrix K:
  ``eigenvalues``, ``eigenvectors`` and ``modes``.

``save_dl_solver`` packs all of these into a single ``.npz`` file, and
``load_dl_solver`` rebuilds a ready-to-``predict`` solver from that file --
no retraining required.

Note: this code assumes the legacy Keras 2 backend, i.e. the environment
variable ``TF_USE_LEGACY_KERAS=1`` (the "Python (koopman)" Jupyter kernel
sets this automatically).
"""

import numpy as np

from koopmanlib.dictionary import PsiNN
from koopmanlib.solver import KoopmanDLSolver


def save_dl_solver(solver, path, layer_sizes, n_psi_train):
    """Save everything needed to reload `solver` for prediction.

    :param solver: a *trained* ``KoopmanDLSolver`` (i.e. ``solver.build(...)`` done)
    :param path: output path, e.g. ``"weights/duffing_dl_weights.npz"``
    :param layer_sizes: the ``layer_sizes`` used to build the ``PsiNN`` dictionary
    :param n_psi_train: the ``n_psi_train`` used to build the ``PsiNN`` dictionary
    """
    nn_weights = solver.dic.get_weights()        # list of numpy arrays (the NN params)
    arrays = {f"nn_{i}": w for i, w in enumerate(nn_weights)}
    arrays.update(
        n_nn_weights=np.array(len(nn_weights)),
        eigenvalues=np.asarray(solver.eigenvalues),
        eigenvectors=np.asarray(solver.eigenvectors),
        modes=np.asarray(solver.modes),
        K=np.asarray(solver.K),
        layer_sizes=np.asarray(layer_sizes),
        n_psi_train=np.asarray(n_psi_train),
        target_dim=np.asarray(solver.target_dim),
        reg=np.asarray(solver.reg),
    )
    np.savez(path, **arrays)
    return path


def load_dl_solver(path):
    """Rebuild a ``KoopmanDLSolver`` from a file written by ``save_dl_solver``.

    The returned solver is ready for ``solver.predict(x0, traj_len)`` and
    ``solver.eigenfunctions(x)``.
    """
    data = np.load(path, allow_pickle=True)

    layer_sizes = [int(x) for x in data["layer_sizes"]]
    n_psi_train = int(data["n_psi_train"])
    target_dim = int(data["target_dim"])
    reg = float(data["reg"])

    # Rebuild the dictionary with the SAME architecture, then load the weights.
    dic = PsiNN(layer_sizes=layer_sizes, n_psi_train=n_psi_train)
    dic(np.zeros((1, target_dim)))               # one call to create the variables
    n = int(data["n_nn_weights"])
    dic.set_weights([data[f"nn_{i}"] for i in range(n)])

    solver = KoopmanDLSolver(dic=dic, target_dim=target_dim, reg=reg)
    solver.eigenvalues = data["eigenvalues"]
    solver.eigenvectors = data["eigenvectors"]
    solver.eigenvectors_inv = np.linalg.inv(solver.eigenvectors)
    solver.modes = data["modes"]
    solver.K = data["K"]
    return solver
