"""
Plot the performances of the many different NMF algorithms on the noise test.
Also include the ARD models.
"""

import matplotlib.pyplot as plt, numpy


''' Plot settings. '''
metrics = ['MSE'] #['MSE','R^2','Rp']
noise_ratios = [ 0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5 ]

N = len(noise_ratios) # number of bars
ind = 1.6*numpy.arange(N) # x locations groups. We have 7 so need 8*width
offset = 0 # offset for first bar
width = 0.2 # width of bars
MSE_max = 40

folder_plots = "./"
folder_results = "./../results/"
plot_file = folder_plots+"mse_nmf_ard_noise.png"
legend_file = folder_plots+"legend.png"


''' Load in the performances. '''
vb_performances = eval(open(folder_results+'nmf_vb.txt','r').read())
vb_ard_performances = eval(open(folder_results+'nmf_vb_ard.txt','r').read())
gibbs_performances = eval(open(folder_results+'nmf_gibbs.txt','r').read())
gibbs_ard_performances = eval(open(folder_results+'nmf_gibbs_ard.txt','r').read())
icm_performances = eval(open(folder_results+'nmf_icm.txt','r').read())
icm_ard_performances = eval(open(folder_results+'nmf_icm_ard.txt','r').read())
np_performances = eval(open(folder_results+'nmf_np.txt','r').read())


''' Assemble the average performances and method names. '''
methods = ['VB-NMF', 'VB-NMF (ARD)', 'G-NMF', 'G-NMF (ARD)', 'ICM-NMF', 'ICM-NMF (ARD)', 'NP-NMF']
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


''' Plot the performances for the metrics specified. '''
for metric in metrics:
    fig = plt.figure(figsize=(3,1.5))
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.17, top=0.95)
    plt.xlabel("Noise to signal ratio", fontsize=8, labelpad=1)
    plt.ylabel(metric, fontsize=8, labelpad=-1)
    plt.yticks(range(0,MSE_max+1,5),fontsize=6)
    plt.xticks(fontsize=6)
    
    for (method, all_perf, colour) in zip(methods,performances,colours):
        x, y = noise_ratios, numpy.mean(all_perf[metric],axis=1)
        plt.bar(ind+offset, y, width, label=method, color=colour)
        offset += width
        
    plt.ylim(0,MSE_max)
    plt.xlim(0,11.3)
    plt.xticks(ind + 3.5*width, x)
    
    plt.savefig(plot_file, dpi=600)
