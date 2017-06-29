"""
Plot the performances of the different NMTF algorithms for the sparsity levels.
"""

import matplotlib.pyplot as plt
import numpy


''' Plot settings. '''
metrics = ['MSE']#['MSE','R^2','Rp']
MSE_min, MSE_max = 700, 1100
fractions_unknown = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

folder_plots = "./"
folder_results = "./../results/"
plot_file = folder_plots+"mse_nmtf_sparsity.png"


''' Load in the performances. '''
vb_performances = eval(open(folder_results+'nmtf_vb.txt','r').read())
gibbs_performances = eval(open(folder_results+'nmtf_gibbs.txt','r').read())
icm_performances = eval(open(folder_results+'nmtf_icm.txt','r').read())
np_performances = eval(open(folder_results+'nmtf_np.txt','r').read())


''' Assemble the average performances and method names. '''
methods = ['VB-NMTF', 'G-NMTF', 'ICM-NMTF', 'NP-NMTF']
performances = [
    vb_performances,
    gibbs_performances,
    icm_performances,
    np_performances,
]
colours = ['r','b','g','c']

for metric in metrics:
    fig = plt.figure(figsize=(1.9,1.5))
    fig.subplots_adjust(left=0.20, right=0.96, bottom=0.18, top=0.97)
    #plt.title("Performances (%s) for different fractions of missing values" % metric)
    plt.xlabel("Fraction missing", fontsize=8, labelpad=1)
    plt.ylabel(metric, fontsize=8, labelpad=0)
    
    for (method, all_perf, colour) in zip(methods,performances,colours):
        x, y = fractions_unknown, numpy.mean(all_perf[metric],axis=1)
        plt.plot(x,y,linestyle='-', marker='o', label=method, c=colour, markersize=3)
    
    plt.xlim(0.2, 1.)
    plt.xticks(numpy.arange(0.2, 1.1, 0.2), fontsize=6)
    if metric == 'MSE':
        plt.ylim(MSE_min,MSE_max)
        plt.yticks(range(MSE_min,MSE_max+1,100), fontsize=6)
    else:
        plt.ylim(0.5,1.05)
     
    plt.savefig(plot_file, dpi=600)