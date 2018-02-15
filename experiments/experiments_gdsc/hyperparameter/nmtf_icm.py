'''
Test the performance of NMTF ICM for recovering the GDSC dataset, where we vary 
the value of the hyperparameter lambda, and the fraction of unobserved entries. 
Run 10-fold cross-validation for each value of lambda and fraction.
'''

import sys, os
project_location = os.path.dirname(__file__)+"/../../../../"
sys.path.append(project_location)

from BNMTF_ARD.code.models.nmtf_icm import nmtf_icm
from BNMTF_ARD.data.drug_sensitivity.load_data import load_gdsc_ic50
from BNMTF_ARD.code.cross_validation.mask import try_generate_M

import matplotlib.pyplot as plt


''' Experiment settings. '''
values_lambda = [0.0001, 0.001, 0.01, 0.1, 1., 10., 100.]
fractions_unknown = [0.2, 0.5, 0.8]

output_folder = project_location+"BNMTF_ARD/experiments/experiments_gdsc/hyperparameter/results/"
output_file = output_folder+'nmtf_icm.txt'


''' Model settings. '''
iterations, burn_in, thinning = 500, 450, 2
no_folds = 10
K, L = 10, 10
init_FG, init_S = 'kmeans', 'random'
ARD = False

alphatau, betatau = 1., 1.
alpha0, beta0 = 1., 1.
hyperparams = { 'alphatau':alphatau, 'betatau':betatau, 'alpha0':alpha0, 'beta0':beta0, 'lambdaF':0., 'lambdaS':0., 'lambdaG':0. }


''' Load in data. '''
R, M = load_gdsc_ic50()
I, J = M.shape


''' Generate 10 M_train and M_test matrices for each value of lambda, and each
    value of fraction_unknown. '''
M_attempts = 1000
all_Ms_fraction_lambda = [
    [
        [
            try_generate_M(I=I, J=J, fraction=fraction, attempts=M_attempts, M=M)
            for f in range(no_folds)
        ]
        for lamb in values_lambda
    ]
    for fraction in fractions_unknown
]


''' We now run the Gibbs sampler on each of the M's for each fraction. '''
all_performances = { 
    fraction: { lamb: [] for lamb in values_lambda } for fraction in fractions_unknown }
average_performances = {
    fraction: { lamb: 0. for lamb in values_lambda } for fraction in fractions_unknown }

for fraction, (all_Ms_lambda) in zip(fractions_unknown, all_Ms_fraction_lambda):
    print "Trying fraction unknown=%s." % fraction
    
    for lamb, Ms_train_and_test in zip(values_lambda, all_Ms_lambda):
        print "Trying lambdaUV=%s." % lamb
        hyperparams['lambdaF'], hyperparams['lambdaS'], hyperparams['lambdaG'] = lamb, lamb, lamb
                 
        for fold, (M_train, M_test) in enumerate(Ms_train_and_test):
            print "Fold %s of fraction unknown=%s, lambda=%s." % (fold+1, fraction, lamb)
        
            BNMTF = nmtf_icm(R,M_train,K,L,ARD,hyperparams)
            BNMTF.initialise(init_FG=init_FG, init_S=init_S)
            BNMTF.run(iterations)
            
            performance = BNMTF.predict(M_test,burn_in,thinning)['MSE']
            all_performances[fraction][lamb].append(performance)
            
        average_performances[fraction][lamb] = sum(all_performances[fraction][lamb]) / float(no_folds)
            

''' Print and store the performances. '''
print "fractions_unknown = %s \nvalues_lambda = %s \nall_performances = %s \naverage_performances = %s" % \
    (fractions_unknown, values_lambda, all_performances, average_performances)
open(output_file,'w').write("%s" % all_performances)


''' Plot the average performances. '''
plt.figure()
for fraction in fractions_unknown:
    x = values_lambda
    y = [average_performances[fraction][lamb] for lamb in values_lambda]
    plt.plot(x, y, label='Fraction %s' % fraction)
    plt.xlabel('lambdaF, lambdaS, lambdaG')
    plt.ylabel('MSE')
    plt.xscale("log")
plt.legend()