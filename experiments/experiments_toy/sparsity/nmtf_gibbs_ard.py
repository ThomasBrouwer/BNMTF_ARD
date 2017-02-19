'''
Test the performance of Gibbs sampling for recovering a toy dataset, where we 
vary the fraction of entries that are missing.
We repeat this 10 times per fraction and average that.
'''

project_location = "/Users/thomasbrouwer/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

from BNMTF_ARD.code.models.bnmtf_gibbs import bnmtf_gibbs
from BNMTF_ARD.code.cross_validation.mask import calc_inverse_M
from BNMTF_ARD.code.cross_validation.mask import try_generate_M

import numpy, matplotlib.pyplot as plt


''' Experiment settings. '''
repeats = 10
fractions_unknown = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95] #[ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 ]

input_folder = project_location+"BNMTF_ARD/data/toy/bnmtf/"
output_folder = project_location+"BNMTF_ARD/experiments/experiments_toy/sparsity/results/"
output_file = output_folder+'nmtf_gibbs_ard.txt'

metrics = ['MSE', 'R^2', 'Rp']


''' Model settings. '''
iterations = 1000
burn_in = 800
thinning = 2

init_FG, init_S = 'kmeans', 'random'
I, J, K, L = 100, 80, 5, 5
ARD = True

lambdaF, lambdaS, lambdaG = 0.1, 0.1, 0.1
alphatau, betatau = 1., 1.
alpha0, beta0 = 1., 1.
hyperparams = { 'alphatau':alphatau, 'betatau':betatau, 'alpha0':alpha0, 'beta0':beta0, 'lambdaF':lambdaF, 'lambdaS':lambdaS, 'lambdaG':lambdaG }


''' Load in data. '''
R = numpy.loadtxt(input_folder+"R.txt")


''' Generate matrices M - one list of M's for each fraction. '''
M_attempts = 100
all_Ms = [ 
    [try_generate_M(I,J,fraction,M_attempts)[0] for r in range(0,repeats)]
    for fraction in fractions_unknown
]
all_Ms_test = [ [calc_inverse_M(M) for M in Ms] for Ms in all_Ms ]


''' Make sure each M has no empty rows or columns. '''
def check_empty_rows_columns(M,fraction):
    sums_columns = M.sum(axis=0)
    sums_rows = M.sum(axis=1)
    for i,c in enumerate(sums_rows):
        assert c != 0, "Fully unobserved row in M, row %s. Fraction %s." % (i,fraction)
    for j,c in enumerate(sums_columns):
        assert c != 0, "Fully unobserved column in M, column %s. Fraction %s." % (j,fraction)
        
for Ms,fraction in zip(all_Ms,fractions_unknown):
    for M in Ms:
        check_empty_rows_columns(M,fraction)


''' We now run the Gibbs sampler on each of the M's for each fraction. '''
all_performances = {metric:[] for metric in metrics} 
average_performances = {metric:[] for metric in metrics} # averaged over repeats
for (fraction,Ms,Ms_test) in zip(fractions_unknown,all_Ms,all_Ms_test):
    print "Trying fraction %s." % fraction
    
    # Run the algorithm <repeats> times and store all the performances
    for metric in metrics:
        all_performances[metric].append([])
    for (repeat,M,M_test) in zip(range(0,repeats),Ms,Ms_test):
        print "Repeat %s of fraction %s." % (repeat+1, fraction)
    
        BNMTF = bnmtf_gibbs(R,M,K,L,ARD,hyperparams)
        BNMTF.initialise(init_FG=init_FG, init_S=init_S)
        BNMTF.run(iterations)
    
        # Measure the performances
        performances = BNMTF.predict(M_test,burn_in,thinning)
        for metric in metrics:
            # Add this metric's performance to the list of <repeat> performances for this fraction
            all_performances[metric][-1].append(performances[metric])
            
    # Compute the average across attempts
    for metric in metrics:
        average_performances[metric].append(sum(all_performances[metric][-1])/repeats)
    
 
''' Print and store the performances. '''
print "repeats=%s \nfractions_unknown = %s \nall_performances = %s \naverage_performances = %s" % \
    (repeats,fractions_unknown,all_performances,average_performances)
open(output_file,'w').write("%s" % all_performances)


''' Plot the average performances. '''
for metric in ['MSE']:
    plt.figure()
    x = fractions_unknown
    y = average_performances[metric]
    plt.plot(x,y)
    plt.xlabel("Fraction missing")
    plt.ylabel(metric)