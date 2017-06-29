"""
Plot the performances of the different NMF algorithms for the sparsity levels.
"""

import matplotlib.pyplot as plt
import numpy


''' Plot settings. '''
metrics = ['MSE']#['MSE','R^2','Rp']
MSE_min, MSE_max = 0, 15
fractions_unknown = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

folder_plots = "./"
folder_results = "./../results/"
plot_file = folder_plots+"mse_nmf_ard_sparsity.png"
legend_file = folder_plots+"legend_ard.png"


''' Load in the performances. '''
def eval_handle_nan(fin):
    string = open(fin,'r').readline()
    old, new = "nan", "10000" #"numpy.nan"
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
    fig.subplots_adjust(left=0.14, right=0.95, bottom=0.18, top=0.97)
    #plt.title("Performances (%s) for different fractions of missing values" % metric)
    plt.xlabel("Fraction missing", fontsize=8, labelpad=1)
    plt.ylabel(metric, fontsize=8, labelpad=0)
    if metric == 'MSE':
        plt.yticks(range(0,MSE_max+1,2),fontsize=6)
    else:
        plt.yticks(fontsize=6)
    plt.xticks(numpy.arange(fractions_unknown[0]-0.05, fractions_unknown[-1]+0.051, 0.2), fontsize=6)
    
    for (method, all_perf, colour) in zip(methods,performances,colours):
        x, y = fractions_unknown, numpy.mean(all_perf[metric],axis=1)
        plt.plot(x,y,linestyle='-', marker='o', label=method, c=colour, markersize=3)
    
    plt.xlim(0.0,1.)
    if metric == 'MSE':
        plt.ylim(MSE_min,MSE_max)
    else:
        plt.ylim(0.5,1.05)
     
    plt.savefig(plot_file, dpi=600)
      
    # Set up the legend outside
    font_size_legend, number_of_columns, legend_box_line_width, legend_line_width = 12, 4, 1, 2
    ax = fig.add_subplot(111)
    legend_fig = plt.figure(figsize=(8.2,0.6))
    legend = legend_fig.legend(*ax.get_legend_handles_labels(), loc='center', prop={'size':font_size_legend}, ncol=number_of_columns)
    legend.get_frame().set_linewidth(legend_box_line_width)
    plt.setp(legend.get_lines(),linewidth=legend_line_width)
        
    plt.savefig(legend_file, dpi=600)