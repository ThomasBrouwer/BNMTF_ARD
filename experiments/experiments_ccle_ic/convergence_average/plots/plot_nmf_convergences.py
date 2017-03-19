"""
Plot the convergence of the many different NMF algorithms on the toy data.
"""

import matplotlib.pyplot as plt


''' Plot settings. '''
MSE_min, MSE_max = 0, 5
iterations = range(1,300+1)

folder_plots = "./"
folder_results = "./../results/"
plot_file = folder_plots+"mse_nmf_convergences_ccle_ic.png"
legend_file = folder_plots+"legend.png"


''' Load in the performances. '''
def eval_handle_nan(fin):
    string = open(fin,'r').readline()
    old, new = "nan", "numpy.nan"
    string = string.replace(old, new)
    return eval(string)
    
vb_all_performances = eval(open(folder_results+'nmf_vb_all_performances.txt','r').read())
gibbs_all_performances = eval(open(folder_results+'nmf_gibbs_all_performances.txt','r').read())
icm_all_performances = eval_handle_nan(folder_results+'nmf_icm_all_performances.txt')
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


''' Plot the performances. '''
fig = plt.figure(figsize=(1.9,1.5))
fig.subplots_adjust(left=0.13, right=0.95, bottom=0.17, top=0.95)
plt.xlabel("Iterations", fontsize=8, labelpad=0)
plt.ylabel("MSE", fontsize=8, labelpad=0)
plt.yticks(range(0,MSE_max+1),fontsize=6)
plt.xticks(fontsize=6)

x = iterations
for method, performances, colour in zip(methods,all_performances,colours):
    y = performances[0:len(iterations)]
    plt.plot(x,y,linestyle='-', marker=None, label=method, c=colour)

plt.yticks(range(0,MSE_max+1,1))
plt.ylim(MSE_min,MSE_max)
    
plt.savefig(plot_file, dpi=600)
