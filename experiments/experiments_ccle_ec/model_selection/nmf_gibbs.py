'''
Test the performance of Gibbs sampling for recovering the CCLE EC dataset, where we 
vary the number of factors. Run cross-validation for each value of K.
'''

project_location = "/Users/thomasbrouwer/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

from BNMTF_ARD.code.models.bnmf_gibbs import bnmf_gibbs
from BNMTF_ARD.data.drug_sensitivity.load_data import load_ccle_ec50
from BNMTF_ARD.code.cross_validation.mask import compute_folds_attempts

import matplotlib.pyplot as plt


''' Experiment settings. '''
no_folds = 10
values_K = [1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,25,30,40]

output_folder = project_location+"BNMTF_ARD/experiments/experiments_ccle_ec/model_selection/results/"
output_file = output_folder+'nmf_gibbs.txt'

metrics = ['MSE', 'R^2', 'Rp']


''' Model settings. '''
iterations = 200
burn_in = 180
thinning = 2

init_UV = 'random'
ARD = False

lambdaU, lambdaV = 0.1, 0.1
alphatau, betatau = 1., 1.
alpha0, beta0 = 1., 1.
hyperparams = { 'alphatau':alphatau, 'betatau':betatau, 'alpha0':alpha0, 'beta0':beta0, 'lambdaU':lambdaU, 'lambdaV':lambdaV }


''' Load in data. '''
R, M = load_ccle_ec50()
I, J = M.shape


''' Generate matrices M - one list of M's for each value of K. '''
M_attempts = 1000
all_Ms_training_and_test = [
    compute_folds_attempts(I=I,J=J,no_folds=no_folds,attempts=M_attempts,M=M)
    for K in values_K
]


''' We now run the Gibbs sampler on each of the M's for each fraction. '''
all_performances = {metric:[] for metric in metrics} 
average_performances = {metric:[] for metric in metrics} # averaged over repeats
for K,(Ms_train,Ms_test) in zip(values_K,all_Ms_training_and_test):
    print "Trying K=%s." % K
    
    # Run the algorithm <repeats> times and store all the performances
    for metric in metrics:
        all_performances[metric].append([])
    for fold,(M_train,M_test) in enumerate(zip(Ms_train,Ms_test)):
        print "Fold %s of K=%s." % (fold+1, K)
        
        BNMF = bnmf_gibbs(R,M_train,K,ARD,hyperparams)
        BNMF.initialise(init_UV)
        BNMF.run(iterations)
    
        # Measure the performances
        performances = BNMF.predict(M_test,burn_in,thinning)
        for metric in metrics:
            # Add this metric's performance to the list of <repeat> performances for this fraction
            all_performances[metric][-1].append(performances[metric])
            
    # Compute the average across attempts
    for metric in metrics:
        average_performances[metric].append(sum(all_performances[metric][-1])/no_folds)
    
 
''' Print and store the performances. '''
print "values_K = %s \nall_performances = %s \naverage_performances = %s" % \
    (values_K,all_performances,average_performances)
open(output_file,'w').write("%s" % all_performances)


''' Plot the average performances. '''
for metric in ['MSE']:
    plt.figure()
    x = values_K
    y = average_performances[metric]
    plt.plot(x,y)
    plt.xlabel("K")
    plt.ylabel(metric)