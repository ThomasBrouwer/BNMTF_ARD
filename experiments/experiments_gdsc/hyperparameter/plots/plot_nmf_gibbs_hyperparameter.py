"""
Plot the performances of NMF Gibbs for different hyperparameter values, for 
three different sparsity levels.
"""

import matplotlib.pyplot as plt
import numpy


''' Plot settings. '''
MSE_min, MSE_max = 650, 850

values_lambda = [0.0001, 0.001, 0.01, 0.1, 1., 10., 100.]
fractions_unknown = [0.2, 0.5, 0.8]

folder_plots = "./"
folder_results = "./../results/"
plot_file = folder_plots+"nmf_gibbs_hyperparameter.png"


''' Load in the performances. '''
performances = eval(open(folder_results+'nmf_gibbs.txt','r').read())
average_performances = {
    fraction: [
        numpy.mean(performances[fraction][lamb]) 
        for lamb in values_lambda
    ]
    for fraction in fractions_unknown
}


''' Plot the performances - one line per fraction. '''
fig = plt.figure(figsize=(1.9,1.5))
fig.subplots_adjust(left=0.17, right=0.96, bottom=0.18, top=0.97)

plt.xlabel('lambdaU, lambdaV', fontsize=8, labelpad=1)
plt.xscale("log")
plt.ylabel('MSE', fontsize=8, labelpad=0)
plt.ylim(MSE_min, MSE_max)

for fraction in fractions_unknown:
    x = values_lambda
    y = [average_performances[fraction][lamb] for lamb in values_lambda]
    plt.plot(x, y, label='Fraction %s' % fraction)
plt.legend()
plt.savefig(plot_file, dpi=600)