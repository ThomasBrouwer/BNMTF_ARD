"""
Run the nested cross-validation for the Gibbs NMF class, on the CCLE IC50 dataset.
"""

project_location = "/home/tab43/Documents/Projects/libraries/" # "/Users/thomasbrouwer/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

from BNMTF_ARD.code.models.bnmf_gibbs import bnmf_gibbs
from BNMTF_ARD.code.cross_validation.nested_matrix_cross_validation import MatrixNestedCrossValidation
from BNMTF_ARD.data.drug_sensitivity.load_data import load_ccle_ic50


''' Settings NMF. '''
ARD = False
lambdaU = 0.1
lambdaV = 0.1
alphatau, betatau = 1., 1.
alpha0, beta0 = 1., 1.
hyperparams = { 'alphatau':alphatau, 'betatau':betatau, 'alpha0':alpha0, 'beta0':beta0, 'lambdaU':lambdaU, 'lambdaV':lambdaV }

train_config = {
    'iterations' : 200,
    'init_UV' : 'random',
}
predict_config = {
    'burn_in' : 180,
    'thinning' : 2,
}


''' Settings nested cross-validation. '''
K_range = [1,2,3,4,5,6,7,8]
no_folds = 10
no_threads = 5
parallel = False
output_file = "./results.txt"
files_nested_performances = ["./fold_%s.txt" % fold for fold in range(1,no_folds+1)]


''' Construct the parameter search. '''
parameter_search = [{'K':K, 'ARD':ARD, 'hyperparameters':hyperparams} for K in K_range]


''' Load in the dataset. '''
R, M = load_ccle_ic50()


''' Run the cross-validation framework. '''
nested_crossval = MatrixNestedCrossValidation(
    method=bnmf_gibbs,
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
