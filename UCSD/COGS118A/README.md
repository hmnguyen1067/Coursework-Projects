# A replication analysis of the paper CNM06

This is the final project for COGS118A @ UCSD. It follows loosely on the procedures laid out in paper CNM06: randomly choose 5000 data samples for your training set with replacement, do 5-fold cross-validation on the training set to select the hyperparameters via a systematic gridsearch of the parameter space. After training, model performance will be measured on the test set with the best set of hyperparameter setting.

## Installation

I use `python=3.8.10` along with dependencies:

- `matplotlib=3.3.4`
- `numpy=1.20.0`
- `sklearn=0.23.1`
- `pandas=1.2.3`
- `seaborn=0.11.1`
