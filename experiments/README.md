# Running Experiments

## Setup

To run the experiments, the library needs to be set up as described [here](../README.md)

## Toy examples

See the notebooks in `/notebooks` and the example in the main README.

## UCI classification

All the commands can be produced by running `python uci_classification_commands.py` which will print out the commands.
These commands all need to be run and a `results/` directory needs to exist in this folder. 
For each experiment, a result-file will be saved which can be loaded with `pickle` and the performances for all the methods for all hyperparameters are saved.
See `classification.py` for the structure of the saved result file.
The runners will automatically use the GPU if available.

## Image classification

The image classification experiments are sequential and require CUDA support.
All the commands that need to be run are printed running `python img_classification_commands.py`.
1. We train several neural networks with different prior precisions as we do not want to artificially change the prior after training to convergence.
   The corresponding models are then saved in `/models`.
   See the file `imgclassification.py` for details on training.
2. We infer the posterior using the `Laplace` class which implements the Laplace-GGN posterior approximation and make predictions with the GLM and BNN predictives.
   See the function `main()` in `imginference.py`.
3. We use the best-performing models per predictive type (MAP, GLM-Kron, GLM-Diag, BNN-Kron, BNN-Diag) and make predictions on out-of-distribution data.
   See the function `ood()` in `imginference.py`.
4. We take the model with the best MAP performance and use the proposed GP inference.
   See the function `gp()` which computes the subset-of-data GP posterior predictive and measures performance and out-of-distribution detection.
   