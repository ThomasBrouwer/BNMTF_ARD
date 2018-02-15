"""
Run the nested cross-validation for the Gibbs NMTF class, on the GDSC dataset.
"""

import sys, os
project_location = os.path.dirname(__file__)+"/../../../../../"
sys.path.append(project_location)

from BNMTF_ARD.code.models.bnmtf_gibbs import bnmtf_gibbs
from BNMTF_ARD.code.cross_validation.nested_matrix_cross_validation import MatrixNestedCrossValidation
from BNMTF_ARD.data.drug_sensitivity.load_data import load_gdsc_ic50

import itertools


''' Settings NMF. '''
ARD = False
lambdaF, lambdaS, lambdaG = 0.1, 0.1, 0.1
alphatau, betatau = 1., 1.
alpha0, beta0 = 1., 1.
hyperparams = { 'alphatau':alphatau, 'betatau':betatau, 'alpha0':alpha0, 'beta0':beta0, 'lambdaF':lambdaF, 'lambdaS':lambdaS, 'lambdaG':lambdaG }

train_config = {
    'iterations' : 200,
    'init_FG' : 'kmeans',
    'init_S' : 'random',
}
predict_config = {
    'burn_in' : 180,
    'thinning' : 2,
}


''' Settings nested cross-validation. '''
KL_range = [8,9,10]
K_range = [5,6,7,8]
L_range = [5,6,7,8]
no_folds = 10
no_threads = 5
parallel = False
output_file = "./results.txt"
files_nested_performances = ["./fold_%s.txt" % fold for fold in range(1,no_folds+1)]


''' Construct the parameter search. '''
parameter_search = [{'K':KL,'L':KL, 'ARD':ARD, 'hyperparameters':hyperparams} for KL in KL_range] # [{'K':K,'L':L, 'ARD':ARD, 'hyperparameters':hyperparams} for K,L in itertools.product(K_range,L_range)]


''' Load in the Sanger dataset. '''
R, M = load_gdsc_ic50()


''' Run the cross-validation framework. '''
nested_crossval = MatrixNestedCrossValidation(
    method=bnmtf_gibbs,
    R=R,
    M=M,
    K=no_folds,
    P=no_threads,
    parameter_search=parameter_search,
    train_config=train_config,
    predict_config=predict_config,
    file_performance=output_file,
    files_nested_performances=files_nested_performances,
)
nested_crossval.run(parallel=parallel)
