"""
Plot the performances of the different NMF algorithms for the sparsity levels.
"""

import matplotlib.pyplot as plt
import numpy


''' Plot settings. '''
metrics = ['MSE']#['MSE','R^2','Rp']
MSE_min, MSE_max = 600, 1500
fractions_unknown = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]

folder_plots = "./"
folder_results = "./../results/"
plot_file = folder_plots+"mse_nmf_ard_sparsity.png"


''' Load in the performances. '''
def eval_handle_nan(fin):
    string = open(fin,'r').readline()
    old, new = "nan", "numpy.nan"
    string = string.replace(old, new)
    return eval(string)
    
vb_performances = eval(open(folder_results+'nmf_vb.txt','r').read())
vb_ard_performances = eval(open(folder_results+'nmf_vb_ard.txt','r').read())
gibbs_performances = eval(open(folder_results+'nmf_gibbs.txt','r').read())
gibbs_ard_performances = eval(open(folder_results+'nmf_gibbs_ard.txt','r').read())
icm_performances = eval(open(folder_results+'nmf_icm.txt','r').read())
icm_ard_performances = eval_handle_nan(folder_results+'nmf_icm_ard.txt')
np_performances = eval(open(folder_results+'nmf_np.txt','r').read())


''' Assemble the average performances and method names. '''
methods = ['VB-NM(T)F', 'VB-NM(T)F (ARD)', 'G-NM(T)F', 'G-NM(T)F (ARD)', 'ICM-NM(T)F', 'ICM-NM(T)F (ARD)', 'NP-NM(T)F']
performances = [
    vb_performances,
    vb_ard_performances,
    gibbs_performances,
    gibbs_ard_performances,
    icm_performances,
    icm_ard_performances,
    np_performances,
]
colours = ['r','m','b','y','g','k','c']

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
    
    plt.xlim(0.0,1.)
    if metric == 'MSE':
        plt.ylim(MSE_min,MSE_max)
        plt.yticks(range(MSE_min,MSE_max+1,100))
    else:
        plt.ylim(0.5,1.05)
     
    plt.savefig(plot_file, dpi=600)