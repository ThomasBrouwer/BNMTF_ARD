"""
Run the nested cross-validation for the Gibbs NMF class with ARD, on the CCLE EC50 dataset.
"""

project_location = "/Users/thomasbrouwer/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

from BNMTF_ARD.code.models.bnmf_gibbs import bnmf_gibbs
from BNMTF_ARD.code.cross_validation.matrix_single_cross_validation import MatrixSingleCrossValidation
from BNMTF_ARD.data.drug_sensitivity.load_data import load_ccle_ec50


''' Settings NMF. '''
K = 20
ARD = True
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
parameters = {'K':K, 'ARD':ARD, 'hyperparameters':hyperparams}

''' Settings nested cross-validation. '''
no_folds = 10
output_file = "./results.txt"


''' Load in the dataset. '''
R, M = load_ccle_ec50()


''' Run the cross-validation framework. '''
crossval = MatrixSingleCrossValidation(
    method=bnmf_gibbs,
    R=R,
    M=M,
    K=no_folds,
    parameters=parameters,
    train_config=train_config,
    predict_config=predict_config,
    file_performance=output_file,
)
crossval.run()
