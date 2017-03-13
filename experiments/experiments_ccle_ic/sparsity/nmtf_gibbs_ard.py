'''
Test the performance of Gibbs sampling with ARD for recovering the CCLE IC50 dataset, 
where we vary the fraction of entries that are missing.
We repeat this 10 times per fraction and average that.
'''

project_location = "/home/tab43/Documents/Projects/libraries/" # "/Users/thomasbrouwer/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

from BNMTF_ARD.code.models.bnmtf_gibbs import bnmtf_gibbs
from BNMTF_ARD.data.drug_sensitivity.load_data import load_ccle_ic50
from BNMTF_ARD.code.cross_validation.mask import try_generate_M
from BNMTF_ARD.code.cross_validation.mask import calc_inverse_M

import matplotlib.pyplot as plt


''' Experiment settings. '''
repeats = 10
fractions_unknown = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]

output_folder = './results/'
output_file = output_folder+'nmtf_gibbs_ard.txt'

metrics = ['MSE', 'R^2', 'Rp']


''' Model settings. '''
iterations = 200
burn_in = 180
thinning = 2

init_FG, init_S = 'kmeans', 'random'
K, L = 7, 7
ARD = True

lambdaF, lambdaS, lambdaG = 0.1, 0.1, 0.1
alphatau, betatau = 1., 1.
alpha0, beta0 = 1., 1.
hyperparams = { 'alphatau':alphatau, 'betatau':betatau, 'alpha0':alpha0, 'beta0':beta0, 'lambdaF':lambdaF, 'lambdaS':lambdaS, 'lambdaG':lambdaG }


''' Load in data. '''
R, M = load_ccle_ic50()
I, J = M.shape


''' Generate matrices M - one list of M's for each fraction. '''
M_attempts = 1000
all_Ms = [ 
    [try_generate_M(I=I,J=J,fraction=fraction,attempts=M_attempts,M=M)[0] for r in range(repeats)]
    for fraction in fractions_unknown
]
all_Ms_test = [ [calc_inverse_M(M_train, M_combined=M) for M_train in Ms] for Ms in all_Ms ]


''' Make sure each M has no empty rows or columns. '''
def check_empty_rows_columns(matrix,fraction):
    sums_columns = matrix.sum(axis=0)
    sums_rows = matrix.sum(axis=1)
    for i,c in enumerate(sums_rows):
        assert c != 0, "Fully unobserved row in matrix, row %s. Fraction %s." % (i,fraction)
    for j,c in enumerate(sums_columns):
        assert c != 0, "Fully unobserved column in matrix, column %s. Fraction %s." % (j,fraction)
        
for Ms,fraction in zip(all_Ms,fractions_unknown):
    for matrix in Ms:
        check_empty_rows_columns(matrix,fraction)


''' We now run the Gibbs sampler on each of the M's for each fraction. '''
all_performances = {metric:[] for metric in metrics} 
average_performances = {metric:[] for metric in metrics} # averaged over repeats
for (fraction,Ms,Ms_test) in zip(fractions_unknown,all_Ms,all_Ms_test):
    print "Trying fraction %s." % fraction
    
    # Run the algorithm <repeats> times and store all the performances
    for metric in metrics:
        all_performances[metric].append([])
    for (repeat,M_train,M_test) in zip(range(0,repeats),Ms,Ms_test):
        print "Repeat %s of fraction %s." % (repeat+1, fraction)
    
        BNMTF = bnmtf_gibbs(R,M_train,K,L,ARD,hyperparams)
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