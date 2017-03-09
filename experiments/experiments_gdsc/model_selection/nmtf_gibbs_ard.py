'''
Test the performance of Gibbs sampling with ARD for recovering the GDSC dataset, 
where we vary the number of factors. Run cross-validation for each value of K and L.
'''

project_location = "/home/tab43/Documents/Projects/libraries/" # "/Users/thomasbrouwer/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

from BNMTF_ARD.code.models.bnmtf_gibbs import bnmtf_gibbs
from BNMTF_ARD.data.drug_sensitivity.load_data import load_gdsc_ic50
from BNMTF_ARD.code.cross_validation.mask import compute_folds_attempts

import matplotlib.pyplot as plt


''' Experiment settings. '''
no_folds = 10
values_KL = [1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,25,30,40]

output_folder = project_location+"BNMTF_ARD/experiments/experiments_gdsc/model_selection/results/"
output_file = output_folder+'nmtf_gibbs_ard.txt'

metrics = ['MSE', 'R^2', 'Rp']


''' Model settings. '''
iterations = 200
burn_in = 180
thinning = 2

init_FG, init_S = 'kmeans', 'random'
ARD = True

lambdaF, lambdaS, lambdaG = 0.1, 0.1, 0.1
alphatau, betatau = 1., 1.
alpha0, beta0 = 1., 1.
hyperparams = { 'alphatau':alphatau, 'betatau':betatau, 'alpha0':alpha0, 'beta0':beta0, 'lambdaF':lambdaF, 'lambdaS':lambdaS, 'lambdaG':lambdaG }


''' Load in data. '''
R, M = load_gdsc_ic50()
I, J = M.shape


''' Generate matrices M - one list of M's for each value of K. '''
M_attempts = 1000
all_Ms_training_and_test = [
    compute_folds_attempts(I=I,J=J,no_folds=no_folds,attempts=M_attempts,M=M)
    for KL in values_KL
]


''' We now run the Gibbs sampler on each of the M's for each fraction. '''
all_performances = {metric:[] for metric in metrics} 
average_performances = {metric:[] for metric in metrics} # averaged over repeats
for KL,(Ms_train,Ms_test) in zip(values_KL,all_Ms_training_and_test):
    print "Trying K,L=%s." % KL
    
    # Run the algorithm <repeats> times and store all the performances
    for metric in metrics:
        all_performances[metric].append([])
    for fold,(M_train,M_test) in enumerate(zip(Ms_train,Ms_test)):
        print "Fold %s of K,L=%s." % (fold+1, KL)
        
        BNMTF = bnmtf_gibbs(R,M_train,KL,KL,ARD,hyperparams)
        BNMTF.initialise(init_FG=init_FG, init_S=init_S)
        BNMTF.run(iterations)
    
        # Measure the performances
        performances = BNMTF.predict(M_test,burn_in,thinning)
        for metric in metrics:
            # Add this metric's performance to the list of <repeat> performances for this fraction
            all_performances[metric][-1].append(performances[metric])
            
    # Compute the average across attempts
    for metric in metrics:
        average_performances[metric].append(sum(all_performances[metric][-1])/no_folds)
    
 
''' Print and store the performances. '''
print "values_KL = %s \nall_performances = %s \naverage_performances = %s" % \
    (values_KL,all_performances,average_performances)
open(output_file,'w').write("%s" % all_performances)


''' Plot the average performances. '''
for metric in ['MSE']:
    plt.figure()
    x = values_KL
    y = average_performances[metric]
    plt.plot(x,y)
    plt.xlabel("K,L")
    plt.ylabel(metric)