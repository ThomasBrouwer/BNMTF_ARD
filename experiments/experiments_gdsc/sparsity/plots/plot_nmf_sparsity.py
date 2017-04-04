"""
Plot the performances of the different NMF algorithms for the sparsity levels.
"""

import matplotlib.pyplot as plt
import numpy


''' Plot settings. '''
metrics = ['MSE']#['MSE','R^2','Rp']
MSE_min, MSE_max = 600, 1500
fractions_unknown = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

folder_plots = "./"
folder_results = "./../results/"
plot_file = folder_plots+"mse_nmf_sparsity.png"


''' Load in the performances. '''
vb_performances = eval(open(folder_results+'nmf_vb.txt','r').read())
gibbs_performances = eval(open(folder_results+'nmf_gibbs.txt','r').read())
icm_performances = eval(open(folder_results+'nmf_icm.txt','r').read())
np_performances = eval(open(folder_results+'nmf_np.txt','r').read())


''' Assemble the average performances and method names. '''
methods = ['VB-NMF', 'G-NMF', 'ICM-NMF', 'NP-NMF']
performances = [
    vb_performances,
    gibbs_performances,
    icm_performances,
    np_performances,
]
colours = ['r','b','g','c']

for metric in metrics:
    fig = plt.figure(figsize=(1.9,1.5))
    fig.subplots_adjust(left=0.19, right=0.95, bottom=0.18, top=0.97)
    #plt.title("Performances (%s) for different fractions of missing values" % metric)
    plt.xlabel("Fraction missing", fontsize=8, labelpad=1)
    plt.ylabel(metric, fontsize=8, labelpad=-1)
    if metric == 'MSE':
        plt.yticks(range(0,MSE_max+1,2),fontsize=6)
    else:
        plt.yticks(fontsize=6)
    plt.xticks(fontsize=6)
    
    for (method, all_perf, colour) in zip(methods,performances,colours):
        x, y = fractions_unknown, numpy.mean(all_perf[metric],axis=1)
        plt.plot(x,y,linestyle='-', marker='o', label=method, c=colour, markersize=3)
    
    plt.xlim(0.1, 1.0)
    if metric == 'MSE':
        plt.ylim(MSE_min,MSE_max)
        plt.yticks(range(MSE_min,MSE_max+1,100))
    else:
        plt.ylim(0.5,1.05)
     
    plt.savefig(plot_file, dpi=600)