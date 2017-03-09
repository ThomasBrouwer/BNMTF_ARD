"""
Plot the convergence of the many different NMF algorithms on the toy data.
"""

import matplotlib.pyplot as plt


''' Plot settings. '''
metrics = ['MSE']#,'R^2','Rp']
MSE_max = 3
iterations = range(1,200+1)

folder_plots = "./"
folder_results = "./../results/"
plot_file = folder_plots+"mse_nmf_convergences.png"
legend_file = folder_plots+"legend.png"


''' Load in the performances. '''
vb_all_performances = eval(open(folder_results+'nmf_vb_all_performances.txt','r').read())
gibbs_all_performances = eval(open(folder_results+'nmf_gibbs_all_performances.txt','r').read())
icm_all_performances = eval(open(folder_results+'nmf_icm_all_performances.txt','r').read())
np_all_performances = eval(open(folder_results+'nmf_np_all_performances.txt','r').read())


''' Assemble the average performances and method names. '''
methods = ['VB-NM(T)F', 'G-NM(T)F', 'ICM-NM(T)F', 'NP-NM(T)F']
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
    fig.subplots_adjust(left=0.12, right=0.95, bottom=0.17, top=0.95)
    plt.xlabel("Iterations", fontsize=8, labelpad=0)
    plt.ylabel(metric, fontsize=8, labelpad=-1)
    plt.yticks(range(0,MSE_max+1),fontsize=6)
    plt.xticks(fontsize=6)
    
    x = iterations
    for method, performances, colour in zip(methods,all_performances,colours):
        y = performances[metric][0:len(iterations)]
        plt.plot(x,y,linestyle='-', marker=None, label=method, c=colour)
        
    if metric == 'MSE':
        plt.ylim(0,MSE_max)
    elif metric == 'R^2' or metric == 'Rp':
        plt.ylim(0,1)
        
    plt.savefig(plot_file, dpi=600)
      
    # Set up the legend outside
    font_size_legend, number_of_columns, legend_box_line_width, legend_line_width = 12, 4, 1, 2
    ax = fig.add_subplot(111)
    legend_fig = plt.figure(figsize=(6.7,0.4))
    legend = legend_fig.legend(*ax.get_legend_handles_labels(), loc='center', prop={'size':font_size_legend}, ncol=number_of_columns)
    legend.get_frame().set_linewidth(legend_box_line_width)
    plt.setp(legend.get_lines(),linewidth=legend_line_width)
        
    plt.savefig(legend_file, dpi=600)
    