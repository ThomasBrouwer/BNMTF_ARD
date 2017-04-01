"""
Plot the performances of the normal vs ARD model for NMF ICM.
"""

import matplotlib.pyplot as plt
import numpy


''' Plot settings. '''
metrics = ['MSE']#['MSE','R^2','Rp']
MSE_min, MSE_max = 650, 850
values_K = [1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,25,30,40]

folder_plots = "./"
folder_results = "./../results/"
plot_file = folder_plots+"nmtf_icm_model_selection_2.png"

colour = 'g' #['r','b','g','c'] -> VB, Gibbs, ICM, NP


''' Load in the performances. '''
gibbs_performances = eval(open(folder_results+'nmtf_icm_2.txt','r').read())
gibbs_ard_performances = eval(open(folder_results+'nmtf_icm_ard_2.txt','r').read())


for metric in metrics:
    fig = plt.figure(figsize=(1.9,1.5))
    fig.subplots_adjust(left=0.17, right=0.96, bottom=0.18, top=0.97)
    plt.xlabel("K", fontsize=8, labelpad=1)
    plt.ylabel(metric, fontsize=8, labelpad=-1)
    
    x, y1, y2 = values_K, numpy.mean(gibbs_ard_performances[metric],axis=1), numpy.mean(gibbs_performances[metric],axis=1)
    plt.plot(x,y1,linestyle='-', marker='o', c=colour, markersize=3)
    plt.plot(x,y2,linestyle='--', marker='x', c=colour, markersize=3)
    
    plt.xticks(fontsize=6)
    if metric == 'MSE':
        plt.ylim(MSE_min,MSE_max)
        plt.yticks(range(MSE_min,MSE_max+1,100),fontsize=6)
    else:
        plt.ylim(0.5,1.05,fontsize=6)
     
    plt.savefig(plot_file, dpi=600)