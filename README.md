# Koopman Operator Analysis with Deep Learning

This package implements the data-driven extended dynamic mode decomposition (EDMD) with trainable dictionary [1].
This a generalization of the classical EDMD and is most effective for high dimensional or highly nonlinear problems
where *a priori* choice of dictionary functions are difficult. 
This package implements iterative algorithms to perform Koopman operator analysis, such as computing eigenfunctions, eigenvalues and modes,
using a deep neural network based parameterization of the Koopman dictionary functions.

## Installation

This project uses `python 3.8`. Set up the project for development using the following steps:

1. Create a virtual environment
    ```bash
    $virtualenv -p python3.8 ~/.virtualenvs/koopman
    ```
2. Activate the environment
    ```bash
    $source ~/.virtualenvs/koopman/bin/activate
    ```
3. Install requirements
    ```bash
    $pip install -r requirements.txt
    ```
4. Perform editable install for development
    ```bash
    $pip install .
    ```
5. Add this virtual environment to Jupyter by typing
    ```bash
    $python -m ipykernel install --user --name=koopman
    ```

## Generate Documentation

Generate sphinx documentation:

1. Go to docs directory
    ```shell
    $cd docs
    ```
2. Run the build
    ```shell
    $sphinx-build -b html source build
    ```
3. You can generate documents by
    ```shell
    $make latexpdf
    ```
    Otherwise, open `docs/build/index.html` with any browser to see the html.

## Test Basic Modules

1. Go to folder tests 
    ```shell
    $cd tests
    ```
2. Test
    * Test dictionaries
        ```shell
        $python test_dictionaries.py
        ```
    * Test solvers
        ```shell
        $python test_solvers.py
        ```

## Quickstart

We use Duffing equation and Van der Pol oscillator as examples to show how to use this package.

Look at [examples](./examples).

## Reference

[1] [Li, Q., Dietrich, F., Bollt, E. M., & Kevrekidis, I. G. (2017). Extended dynamic mode decomposition with dictionary learning: A data-driven adaptive spectral decomposition of the Koopman operator. Chaos: An Interdisciplinary Journal of Nonlinear Science, 27(10), 103111.](https://aip-scitation-org.libproxy1.nus.edu.sg/doi/full/10.1063/1.4993854).

## Contributors
* GUO Yue (NUS)
* LI Qianxiao (NUS)
