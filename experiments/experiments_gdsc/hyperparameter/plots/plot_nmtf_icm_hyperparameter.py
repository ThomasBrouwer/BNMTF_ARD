"""
Plot the performances of NMTF ICM for different hyperparameter values, for 
three different sparsity levels.
"""

import matplotlib.pyplot as plt
import numpy


''' Plot settings. '''
MSE_min, MSE_max = 600, 1400

values_lambda = [0.0001, 0.001, 0.01, 0.1, 1., 10., 100.]
fractions_unknown = [0.2, 0.5, 0.8]

folder_plots = "./"
folder_results = "./../results/"
plot_file = folder_plots+"nmtf_icm_hyperparameter.png"


''' Load in the performances. '''
performances = eval(open(folder_results+'nmtf_icm.txt','r').read())
average_performances = {
    fraction: [
        numpy.mean(performances[fraction][lamb]) 
        for lamb in values_lambda
    ]
    for fraction in fractions_unknown
}


''' Plot the performances - one line per fraction. '''
fig = plt.figure(figsize=(2.5,1.9))
fig.subplots_adjust(left=0.17, right=0.98, bottom=0.17, top=0.98)

plt.xlabel('lambdaF, lambdaS, lambdaG', fontsize=8, labelpad=1)
plt.xscale("log")
plt.xticks(fontsize=6)
plt.ylabel('MSE', fontsize=8, labelpad=1)
plt.yticks(range(0,MSE_max+1,200),fontsize=6)
plt.ylim(MSE_min, MSE_max)

for fraction in fractions_unknown:
    x = values_lambda
    y = average_performances[fraction]
    plt.plot(x, y, label='Fraction %s' % fraction)
plt.savefig(plot_file, dpi=600)