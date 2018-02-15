"""
Run the nested cross-validation for the Gibbs NMTF class with ARD, on the CCLE IC50 dataset.
"""

import sys, os
project_location = os.path.dirname(__file__)+"/../../../../../"
sys.path.append(project_location)

from BNMTF_ARD.code.models.bnmtf_vb import bnmtf_vb
from BNMTF_ARD.code.cross_validation.matrix_single_cross_validation import MatrixSingleCrossValidation
from BNMTF_ARD.data.drug_sensitivity.load_data import load_ccle_ic50


''' Settings NMTF. '''
K, L = 10, 10
ARD = True

lambdaF, lambdaS, lambdaG = 0.1, 0.1, 0.1
alphatau, betatau = 1., 1.
alpha0, beta0 = 1., 1.
hyperparams = { 'alphatau':alphatau, 'betatau':betatau, 'alpha0':alpha0, 'beta0':beta0, 'lambdaF':lambdaF, 'lambdaS':lambdaS, 'lambdaG':lambdaG }

train_config = {
    'iterations' : 500,
    'init_FG' : 'kmeans',
    'init_S' : 'random',
}
predict_config = {}
parameters = {'K':K, 'L':L, 'ARD':ARD, 'hyperparameters':hyperparams}

''' Settings nested cross-validation. '''
no_folds = 10
output_file = "./results.txt"


''' Load in the dataset. '''
R, M = load_ccle_ic50()


''' Run the cross-validation framework. '''
crossval = MatrixSingleCrossValidation(
    method=bnmtf_vb,
    R=R,
    M=M,
    K=no_folds,
    parameters=parameters,
    train_config=train_config,
    predict_config=predict_config,
    file_performance=output_file,
)
crossval.run()
