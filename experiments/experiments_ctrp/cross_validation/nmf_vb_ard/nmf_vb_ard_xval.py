"""
Run the nested cross-validation for the VB NMF class with ARD, on the CTRP dataset.
"""

project_location = "/home/tab43/Documents/Projects/libraries/" # "/Users/thomasbrouwer/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

from BNMTF_ARD.code.models.bnmf_vb import bnmf_vb
from BNMTF_ARD.code.cross_validation.matrix_single_cross_validation import MatrixSingleCrossValidation
from BNMTF_ARD.data.drug_sensitivity.load_data import load_ctrp_ec50


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
predict_config = {}
parameters = {'K':K, 'ARD':ARD, 'hyperparameters':hyperparams}

''' Settings nested cross-validation. '''
no_folds = 10
output_file = "./results.txt"


''' Load in the dataset. '''
R, M = load_ctrp_ec50()


''' Run the cross-validation framework. '''
crossval = MatrixSingleCrossValidation(
    method=bnmf_vb,
    R=R,
    M=M,
    K=no_folds,
    parameters=parameters,
    train_config=train_config,
    predict_config=predict_config,
    file_performance=output_file,
)
crossval.run()
