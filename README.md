# Improving predictions of Bayesian neural nets via local linearization

Accompanying code for the paper
> Immer, A.\*, Korzepa, M., Bauer, M.\*, [*Improving predictions of Bayesian neural nets via local linearization*](https://arxiv.org/abs/2008.08400), AISTATS 2021.

## Predictions with Laplace-GGN

We simply replace the _BNN_ predictive with a _GLM_ predictive for the Laplace-_GGN_ posterior approximation.
The code provides means to compute the Laplace-GGN posterior (diagonal, KFAC, full) and use it to make predictions.
In the following example, we construct the Laplace approximation from a trained model `model`.
The `'kron'` cov-type corresponds to KFAC.
The inferred `posterior` enables sampling from the posterior predictive for a batch of input data `X` using 
`posterior.predictive_samples_glm(X, n_samples=1000)`.

```python
from preds.likelihoods import CategoricalLh
from preds.laplace import Laplace

# infer posterior with Laplace-GGN
lh = CategoricalLh()  # likelihood 
prior_precision = 1.  # prior
posterior = Laplace(model, prior_precision, lh)
posterior.infer(train_loader, cov_type='kron', dampen_kron=False)  # or 'full', 'diag'

# GLM predictions
glm_samples = posterior.predictive_samples_glm(X, n_samples=1000)
# BNN predictions
bnn_samples = posterior.predictive_samples_bnn(X, n_samples=1000)
```

For a running and worked example, see the two examples on regression and classification:
- [classification example notebook](https://github.com/AlexImmer/BNN-predictions/blob/main/notebooks/Classification%20Predictive%20Example.ipynb)
- [regression example notebook](https://github.com/AlexImmer/BNN-predictions/blob/main/notebooks/Regression%20Predictive%20Example.ipynb)

The two examples train a neural network until convergence and construct variants of the Laplace-GGN posterior approximation.
The script plots the posterior predictive of the proposed GLM in comparison to the heavily underfitting BNN predictive.
The underfitting can only be resolved by artificially reducing the posterior variance; 
using a different prior does not help as it fails across the entire range of values.

### Regression example
The resulting plot compares the proposed GLM to the BNN predictive:

![image](https://user-images.githubusercontent.com/7715036/109063105-119ea800-76e9-11eb-8e3a-565d32699bdb.png)

### Classification example
In the following example, the BNN predictive underfits severely so that the contours are almost invisible:

![image](https://user-images.githubusercontent.com/7715036/109063214-33982a80-76e9-11eb-91c4-214a526c5fac.png)

## Setup

We use **python `>=3.7`**. 
To install the dependencies, run `pip install -r requirements.txt`.
To use a GPU, additional installations might be necessary (CUDA, etc.).
Then, install the `cml` package with `pip install .`.
Create the result directories `mkdir run_results` and `mkdir runs`.

```bash
# install requirements
pip install -r requirements.txt

# install the `preds` package
pip install .

# run tests (optional)
pip install pytest
pytest tests
```

