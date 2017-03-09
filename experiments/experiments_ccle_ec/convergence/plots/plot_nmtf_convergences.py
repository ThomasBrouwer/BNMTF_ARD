"""
Plot the convergence of the many different NMF algorithms on the toy data.
"""

import matplotlib.pyplot as plt


''' Plot settings. '''
metrics = ['MSE']#,'R^2','Rp']
MSE_min, MSE_max = 500, 1000
iterations = range(1,300+1)

folder_plots = "./"
folder_results = "./../results/"
plot_file = folder_plots+"mse_nmtf_convergences_gdsc.png"


''' Load in the performances. '''
vb_all_performances = eval(open(folder_results+'nmtf_vb_all_performances.txt','r').read())
gibbs_all_performances = eval(open(folder_results+'nmtf_gibbs_all_performances.txt','r').read())
icm_all_performances = eval(open(folder_results+'nmtf_icm_all_performances.txt','r').read())
np_all_performances = eval(open(folder_results+'nmtf_np_all_performances.txt','r').read())


''' Assemble the average performances and method names. '''
all_performances = [
    vb_all_performances,
    gibbs_all_performances,
    icm_all_performances,
    np_all_performances
]
colours = ['r','b','g','c']


''' Plot the performances for the metrics specified. '''
for metric in metrics:
    fig = plt.figure(figsize=(1.9,1.5))
    fig.subplots_adjust(left=0.14, right=0.95, bottom=0.17, top=0.95)
    plt.xlabel("Iterations", fontsize=8, labelpad=0)
    plt.ylabel(metric, fontsize=8, labelpad=-1)
    plt.yticks(range(0,MSE_max+1),fontsize=6)
    plt.xticks(fontsize=6)
    
    x = iterations
    for performances, colour in zip(all_performances,colours):
        y = performances[metric][0:len(iterations)]
        plt.plot(x,y,linestyle='-', marker=None, c=colour)
        
    if metric == 'MSE':
        plt.yticks(range(0,MSE_max+1,100))
        plt.ylim(MSE_min,MSE_max)
    elif metric == 'R^2' or metric == 'Rp':
        plt.ylim(0,1)
        
    plt.savefig(plot_file, dpi=600)
      