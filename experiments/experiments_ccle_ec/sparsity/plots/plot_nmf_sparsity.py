"""
Plot the performances of the different NMF algorithms for the sparsity levels.
"""

import matplotlib.pyplot as plt
import numpy


''' Plot settings. '''
metrics = ['MSE']#['MSE','R^2','Rp']
MSE_min, MSE_max = 7, 11
fractions_unknown = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
xmin, xmax = 0.35, 0.8

folder_plots = "./"
folder_results = "./../results/"
plot_file = folder_plots+"mse_nmf_sparsity.png"


''' Load in the performances. '''
def eval_handle_nan(fin):
    string = open(fin,'r').readline()
    old, new = "nan", "100" #"numpy.nan"
    string = string.replace(old, new)
    return eval(string)
    
vb_performances = eval(open(folder_results+'nmf_vb.txt','r').read())
gibbs_performances = eval(open(folder_results+'nmf_gibbs.txt','r').read())
icm_performances = eval(open(folder_results+'nmf_icm.txt','r').read())
np_performances = eval_handle_nan(folder_results+'nmf_np.txt')


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
    fig.subplots_adjust(left=0.14, right=0.96, bottom=0.18, top=0.97)
    #plt.title("Performances (%s) for different fractions of missing values" % metric)
    plt.xlabel("Fraction missing", fontsize=8, labelpad=1)
    plt.ylabel(metric, fontsize=8, labelpad=-1)
    
    for (method, all_perf, colour) in zip(methods,performances,colours):
        x, y = fractions_unknown, numpy.mean(all_perf[metric],axis=1)
        plt.plot(x,y,linestyle='-', marker='o', label=method, c=colour, markersize=3)
    
    plt.xlim(xmin, xmax)
    plt.xticks(numpy.arange(0.4, 0.85, 0.1), fontsize=6)
    if metric == 'MSE':
        plt.ylim(MSE_min,MSE_max)
        plt.yticks(range(MSE_min,MSE_max+1,1), fontsize=6)
    else:
        plt.ylim(0.5,1.05)
     
    plt.savefig(plot_file, dpi=600)