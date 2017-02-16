# Bayesian nonnegative matrix tri-factorisation: inference trade-offs
Submitted to ECML 2017.

Authors: **Thomas Brouwer**, **Jes Frellsen**, **Pietro Lio'**.

This project contains an implementation of the Bayesian non-negative matrix factorisation and tri-factorisation models presented in the paper **Fast Bayesian nonnegative matrix factorisation and tri-factorisation**. We furthermore provide all datasets used (including the preprocessing scripts), and Python scripts for experiments.

This project is structured as follows:

### /code/
Python code, for the models, cross-validation methods, and model selection.

#### /models/
Python classes for the BNMF and BNMTF models: Gibbs sampling, Variational Bayes, Iterated Conditional Modes, and non-probabilistic versions. Each class contains both the version with ARD, and without.
- **/distributions/** - Contains code for obtaining draws of the exponential, Gaussian, and Truncated Normal distributions. Also has code for computing the expectation and variance of these distributions.
- **/kmeans/** - Contains a class for performing K-means clustering on a matrix, when some of the values are unobserved. From my [other Github project](https://github.com/ThomasBrouwer/kmeans_missing).
- **bnmf_gibbs.py** - Implementation of Gibbs sampler for Bayesian non-negative matrix factorisation (BNMF), extended to take into account missing values. Initially introduced by Schmidt et al. 2009.
- **bnmf_vb.py** - Implementation of our variational Bayesian inference for BNMF.
- **nmf_icm.py** - Implementation of Iterated Conditional Modes NMF algorithm (MAP inference). Initially introduced by Schmidt et al. 2009.
- **nmf_np.py** - Implementation of non-probabilistic NMF (Lee and Seung 2001).
- **bnmtf_gibbs.py** - Implementation of our Gibbs sampler for Bayesian non-negative matrix tri-factorisation (BNMTF).
- **bnmtf_vb.py** - Implementation of our variational Bayesian inference for BNMTF.
- **nmtf_icm.py** - Implementation of Iterated Conditional Modes NMTF algorithm (MAP inference).
- **nmtf_np.py** - Implementation of non-probabilistic NMTF, introduced by Yoo and Choi 2009.

#### /grid_search/
Classes for doing cross-validation, and nested cross-validation, on the Bayesian NMF and NMTF models
- **matrix_cross_validation.py** - Class for finding the best value of K for any of the models (Gibbs, VB, ICM, NP), using cross-validation.
- **parallel_matrix_cross_validation.py** - Same as matrix_cross_validation.py but P folds are ran in parallel.
- **nested_matrix_cross_validation.py** - Class for measuring cross-validation performance, with nested cross-validation to choose K, used for non-probabilistic NMF and NMTF.
- **mask.py** - Contains methods for splitting data into training and test folds.

#### /data_toy/
Contains the toy data, and methods for generating toy data.
- **/bnmf/** - Generate toy data using **generate_bnmf.py**, giving files **U.txt**, **V.txt**, **R.txt**, **R_true.txt** (no noise), **M.txt**.
- **/bnmtf/** - Generate toy data using **generate_bnmtf.py**, giving files **F.txt**, **S.txt**, **G.txt**, **R.txt**, **R_true.txt** (no noise), **M.txt**.

#### /data_drug_sensitivity/
Contains the drug sensitivity datasets (GDSC IC50, CCLE IC50, CCLE EC50, CTRP EC50).
- **/gdsc/**, **/ctrp/**, **/ccle/** - The datasets. We obtained these from the "Bayesian Hybrid Matrix Factorisation for Data Integration" paper (Thomas Brouwer and Pietro Lio', 2017), using the complete datasets of each (before finding the overlap). More details in description.md.
- **/plots/** - Plots of the distribution of values in the four datasets.

#### /experiments/
- **/experiments_toy/** - Experiments on the toy data.
  - **/convergence/** - Measure convergence rate of the methods (against iterations and time) on the toy data.
  - **/noise/** - Measure the predictive performance on missing values for varying noise levels.
  - **/sparsity/** - Measure the predictive performance on missing values for varying sparsity levels.
- **/experiments_gdsc/** - Experiments on the Sanger GDSC IC50 dataset, as well as helper methods for loading in the data.
  - **/convergence/** - Measure convergence rate of the methods (against iterations and time) on the GDSC data.
  - **/grid_search/** - Measure the effectiveness of the line, grid, and greedy search model selection methods on the Sanger data.
  - **/cross_validation/** - 10-fold cross-validation experiments on the Sanger data.
  - **/sparsity/** - Measure the predictive performance on missing values for varying sparsity levels.

#### /plots/
The results and plots for the experiments are stored in this folder, along with scripts for making the plots.
- **/graphs_toy/** - Plots for the experiments on the toy data.
- **/graphs_Sanger/** - Plots for the experiments on the Sanger GDSC drug sensitivity data.
- **/missing_values/** - Scripts for plotting the varying missing values experiment outcomes.
- **/model_selection/** - Scripts for plotting the model selection experiment outcomes.
- **/noise/** - Scripts for plotting the varying noise experiment outcomes.
- **/convergence/** - Scripts for plotting the convergence (against iterations) on the toy and Sanger data.
- **/time_toy/** - Scripts for plotting the convergence (against time) on the toy data.
- **/time_Sanger/** - Scripts for plotting the convergence (against time) on the Sanger data.

#### /tests/
py.test unit tests for the code and classes in **/code/**. To run the tests, simply `cd` into the /tests/ folder, and run `pytest` in the command line.
