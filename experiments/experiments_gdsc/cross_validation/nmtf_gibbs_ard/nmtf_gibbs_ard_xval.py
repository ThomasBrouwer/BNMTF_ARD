"""
Run the nested cross-validation for the Gibbs NMTF class with ARD, on the GDSC dataset.
"""

project_location = "/Users/thomasbrouwer/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

from BNMTF_ARD.code.models.bnmtf_gibbs import bnmtf_gibbs
from BNMTF_ARD.code.cross_validation.matrix_single_cross_validation import MatrixSingleCrossValidation
from BNMTF_ARD.data.drug_sensitivity.load_data import load_gdsc_ic50


''' Settings NMF. '''
K, L = 10, 10
ARD = True

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
parameters = {'K':K, 'L':L, 'ARD':ARD, 'hyperparameters':hyperparams}

''' Settings nested cross-validation. '''
no_folds = 10
output_file = "./results.txt"


''' Load in the Sanger dataset. '''
R, M = load_gdsc_ic50()


''' Run the cross-validation framework. '''
crossval = MatrixSingleCrossValidation(
    method=bnmtf_gibbs,
    R=R,
    M=M,
    K=no_folds,
    parameters=parameters,
    train_config=train_config,
    predict_config=predict_config,
    file_performance=output_file,
)
crossval.run()
