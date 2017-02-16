"""
Plot the performances of the many different NMF algorithms on the noise test.
"""

import matplotlib.pyplot as plt, numpy


''' Plot settings. '''
metrics = ['MSE'] #['MSE','R^2','Rp']
noise_ratios = [ 0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5 ]

N = len(noise_ratios) # number of bars
ind = numpy.arange(N) # x locations groups
offset = 0 # offset for first bar
width = 0.2 # width of bars
MSE_max = 40

folder_plots = "./"
folder_results = "./../results/"
plot_file = folder_plots+"mse_nmf_noise.png"


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


''' Plot the performances for the metrics specified. '''
for metric in metrics:
    fig = plt.figure(figsize=(1.9,1.5))
    fig.add_subplot(111)
    fig.subplots_adjust(left=0.15, right=0.99, bottom=0.17, top=0.95)
    plt.xlabel("Noise to signal ratio", fontsize=8, labelpad=1)
    plt.ylabel(metric, fontsize=8, labelpad=-1)
    plt.yticks(range(0,MSE_max+1,5),fontsize=6)
    plt.xticks(fontsize=6)
    
    for (method, all_perf, colour) in zip(methods,performances,colours):
        x, y = noise_ratios, numpy.mean(all_perf[metric],axis=1)
        plt.bar(ind+offset, y, width, label=method, color=colour)
        offset += width
        
    plt.ylim(0,MSE_max)
    plt.xticks(numpy.arange(N) + 2*width, x)
    
    plt.savefig(plot_file, dpi=600)
