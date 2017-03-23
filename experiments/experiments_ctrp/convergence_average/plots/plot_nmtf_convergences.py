"""
Plot the convergence of the many different NMF algorithms on the toy data.
"""

import matplotlib.pyplot as plt


''' Plot settings. '''
MSE_min, MSE_max = 650, 800
iterations = range(1,200+1)

folder_plots = "./"
folder_results = "./../results/"
plot_file = folder_plots+"mse_nmtf_convergences_ctrp.png"


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
fig = plt.figure(figsize=(1.9,1.5))
fig.subplots_adjust(left=0.17, right=0.95, bottom=0.17, top=0.95)
plt.xlabel("Iterations", fontsize=8, labelpad=0)
plt.ylabel("MSE", fontsize=8, labelpad=-1)
plt.yticks(range(0,MSE_max+1),fontsize=6)
plt.xticks(fontsize=6)

x = iterations
for performances, colour in zip(all_performances,colours):
    y = performances[0:len(iterations)]
    plt.plot(x,y,linestyle='-', marker=None, c=colour)

plt.yticks(range(0,MSE_max+1,50))
plt.ylim(MSE_min,MSE_max)
    
plt.savefig(plot_file, dpi=600)
  