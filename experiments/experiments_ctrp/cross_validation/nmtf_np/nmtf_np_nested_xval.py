"""
Run the nested cross-validation for the NP NMTF class, on the CTRP dataset.
"""

project_location = "/Users/thomasbrouwer/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

from BNMTF_ARD.code.models.nmtf_np import nmtf_np
from BNMTF_ARD.code.cross_validation.nested_matrix_cross_validation import MatrixNestedCrossValidation
from BNMTF_ARD.data.drug_sensitivity.load_data import load_ctrp_ec50

import itertools


''' Settings NMTF. '''
train_config = {
    'iterations' : 300, #1000,
    'init_FG' : 'kmeans',
    'init_S' : 'exponential',
    'expo_prior' : 0.1
}
predict_config = {}


''' Settings nested cross-validation. '''
KL_range = [3,4]
#K_range = [5,6,7,8]
#L_range = [5,6,7,8]
no_folds = 10
no_threads = 5
parallel = False
output_file = "./results.txt"
files_nested_performances = ["./fold_%s.txt" % fold for fold in range(1,no_folds+1)]


''' Construct the parameter search. '''
parameter_search = [{'K':KL,'L':KL} for KL in KL_range] # [{'K':K,'L':L} for K,L in itertools.product(K_range,L_range)]


''' Load in the dataset. '''
R, M = load_ctrp_ec50()


''' Run the cross-validation framework. '''
nested_crossval = MatrixNestedCrossValidation(
    method=nmtf_np,
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
