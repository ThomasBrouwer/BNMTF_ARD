# Comparative Study of Inference Methods for Bayesian Nonnegative Matrix Factorisation
This project contains an implementation of the Bayesian non-negative matrix factorisation and tri-factorisation models presented in the paper **Comparative Study of Inference Methods for Bayesian Nonnegative Matrix Factorisation**, published at the European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (ECML PKDD 2017). For both models we implement four different inference methods (Gibbs sampling, variational Bayesian inference, iterated conditional modes, and non-probabilistic inference), and for the Bayesian models we also provide automatic model selection (using the automatic relevance determination prior). We furthermore provide all datasets used (including the preprocessing scripts), and Python scripts for experiments.

![alt text](./images/gm_bnmf.pdf "Graphical model of the Bayesian nonnegative matrix factorisation model.")

#### Paper abstract
In this paper, we study the trade-offs of different inference approaches for Bayesian matrix factorisation methods, which are com- monly used for predicting missing values, and for finding patterns in the data. In particular, we consider Bayesian nonnegative variants of matrix factorisation and tri-factorisation, and compare non-probabilistic infer- ence, Gibbs sampling, variational Bayesian inference, and a maximum- a-posteriori approach. The variational approach is new for the Bayesian nonnegative models. We compare their convergence, and robustness to noise and sparsity of the data, on both synthetic and real-world datasets. Furthermore, we extend the models with the Bayesian automatic rele- vance determination prior, allowing the models to perform automatic model selection, and demonstrate its efficiency.

#### Corresponding authors
**Thomas Brouwer**, **Jes Frellsen**, **Pietro Lio'**. Contact: tab43@cam.ac.uk / thomas.a.brouwer@gmail.com.

## Installation 
If you wish to use the matrix factorisation models, or replicate the experiments, follow these steps. Please ensure you have Python 2.7 (3 is currently not supported). 
1. Clone the project to your computer, by running `git clone https://github.com/ThomasBrouwer/BNMTF.git` in your command line.
2. In your Python script, add the project to your system path using the following lines.  
   
   ``` 
   project_location = "/path/to/folder/containing/project/"
   import sys
   sys.path.append(project_location) 
   ```
   For example, if the path to the project is /johndoe/projects/BNMTF/, use `project_location = /johndoe/projects/`.
   If you intend to rerun some of the paper's experiments, make sure to change this variable appropriately in the scripts you are running.
3. You can now import the models in your code, e.g.
```
from BNMTF.code.models.nmf_np import NMF
model = NMF(R=numpy.ones((4,3)), M=ones((4,3)), K=2)
model.initialise()
model.train(iterations=10)
```

## Examples
You can find good examples of the models running on data in the [convergence experiment on the toy data](./experiments/experiments_toy/convergence/), e.g. [nonnegative matrix factorisation with Gibbs sampling](./experiments/experiments_toy/convergence/nmf_gibbs.py).

## Citation
If this project was useful for your research, please consider citing it,
> Thomas Brouwer, Jes Frellsen, and Pietro Lió (2017). Comparative Study of Inference Methods for Bayesian Nonnegative Matrix Factorisation. Proceedings of the European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (ECML PKDD 2017).
```
@inproceedings{Brouwer2017b,
	author = {Brouwer, Thomas and Frellsen, Jes and Li\'{o}, Pietro},
	booktitle = {Proceedings of the European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (ECML PKDD)},
	title = {{Comparative Study of Inference Methods for Bayesian Nonnegative Matrix Factorisation}},
	year = {2017}
}
```

## Project structure
Below you can find a description of the different folders and files available in this repository.

### /code/
Python code, for the models, cross-validation methods, and model selection.

**/models/**: Python classes for the BNMF and BNMTF models: Gibbs sampling, Variational Bayes, Iterated Conditional Modes, and non-probabilistic versions. Each class contains both the version with ARD, and without.
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

**/grid_search/**: Classes for doing cross-validation, and nested cross-validation, on the Bayesian NMF and NMTF models
- **matrix_cross_validation.py** - Class for finding the best value of K for any of the models (Gibbs, VB, ICM, NP), using cross-validation.
- **parallel_matrix_cross_validation.py** - Same as matrix_cross_validation.py but P folds are ran in parallel.
- **nested_matrix_cross_validation.py** - Class for measuring cross-validation performance, with nested cross-validation to choose K, used for non-probabilistic NMF and NMTF.
- **mask.py** - Contains methods for splitting data into training and test folds.

### /data/
Contains the datasets, as well as methods for loading them in.

**/toy/**: Contains the toy data, and methods for generating toy data.
- **/bnmf/** - Generate toy data using **generate_bnmf.py**, giving files **U.txt**, **V.txt**, **R.txt**, **R_true.txt** (no noise), **M.txt**.
- **/bnmtf/** - Generate toy data using **generate_bnmtf.py**, giving files **F.txt**, **S.txt**, **G.txt**, **R.txt**, **R_true.txt** (no noise), **M.txt**.

**/drug_sensitivity/**: Contains the drug sensitivity datasets (GDSC IC50, CCLE IC50, CCLE EC50, CTRP EC50). For more details see description.md.
- **/gdsc/**, **/ctrp/**, **/ccle/** - The datasets. We obtained these from the [Bayesian Hybrid Matrix Factorisation for Data Integration"](https://github.com/ThomasBrouwer/HMF) project ([Thomas Brouwer and Pietro Lio', 2017](https://arxiv.org/abs/1704.04962)), using the complete datasets of each (before finding the overlap). More details in description.md.
- **/plots/** - Plots of the distribution of values in the four datasets.
- **load_data.py** - Contains methods for loading in the four drug sensitivity datasets.

### /experiments/
- **/experiments_toy/** - Experiments on the toy data.
  - **/convergence/** - Measure convergence rate of the methods (against iterations and time) on the toy data.
  - **/noise/** - Measure the predictive performance on missing values for varying noise levels.
  - **/sparsity/** - Measure the predictive performance on missing values for varying sparsity levels.
- **/experiments_gdsc/** - Experiments on the Sanger GDSC IC50 dataset, as well as helper methods for loading in the data.
  - **/convergence/** - Measure convergence rate of the methods (against iterations and time) on the GDSC data.
  - **/grid_search/** - Measure the effectiveness of the line, grid, and greedy search model selection methods on the Sanger data.
  - **/cross_validation/** - 10-fold cross-validation experiments on the Sanger data. Also contains plots of performances.
  - **/sparsity/** - Measure the predictive performance on missing values for varying sparsity levels.
- **/experiments_ctrp/**, **/experiments_ctrp/**, **/experiments_ctrp/** - Similar to /experiments_gdsc/, but on the CTRP EC50, CCLE IC50, and CCLE EC50 datasets.

### /plots/
The results and plots for the experiments are stored in this folder, along with scripts for making the plots.
- **/graphs_toy/** - Plots for the experiments on the toy data.
- **/graphs_Sanger/** - Plots for the experiments on the Sanger GDSC drug sensitivity data.
- **/missing_values/** - Scripts for plotting the varying missing values experiment outcomes.
- **/model_selection/** - Scripts for plotting the model selection experiment outcomes.
- **/noise/** - Scripts for plotting the varying noise experiment outcomes.
- **/convergence/** - Scripts for plotting the convergence (against iterations) on the toy and Sanger data.
- **/time_toy/** - Scripts for plotting the convergence (against time) on the toy data.
- **/time_Sanger/** - Scripts for plotting the convergence (against time) on the Sanger data.

### /tests/
py.test unit tests for the code and classes in **/code/**. To run the tests, simply `cd` into the /tests/ folder, and run `pytest` in the command line.
