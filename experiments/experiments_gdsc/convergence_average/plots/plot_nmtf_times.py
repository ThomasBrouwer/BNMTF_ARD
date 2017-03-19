"""
Plot the convergence of the many different NMF algorithms on the toy data.
"""

import matplotlib.pyplot as plt


''' Plot settings. '''
MSE_min, MSE_max = 500, 900
time_max = 80

folder_plots = "./"
folder_results = "./../results/"
plot_file = folder_plots+"mse_nmtf_times_gdsc.png"


''' Load in the performances. '''
vb_all_performances = eval(open(folder_results+'nmtf_vb_all_performances.txt','r').read())
gibbs_all_performances = eval(open(folder_results+'nmtf_gibbs_all_performances.txt','r').read())
icm_all_performances = eval(open(folder_results+'nmtf_icm_all_performances.txt','r').read())
np_all_performances = eval(open(folder_results+'nmtf_np_all_performances.txt','r').read())


''' Load in the times. '''
vb_all_times = eval(open(folder_results+'nmtf_vb_all_times.txt','r').read())
gibbs_all_times = eval(open(folder_results+'nmtf_gibbs_all_times.txt','r').read())
icm_all_times = eval(open(folder_results+'nmtf_icm_all_times.txt','r').read())
np_all_times = eval(open(folder_results+'nmtf_np_all_times.txt','r').read())


''' Assemble the average performances and method names. '''
methods = ['VB-NM(T)F', 'G-NM(T)F', 'ICM-NM(T)F', 'NP-NM(T)F']
all_performances = [
    vb_all_performances,
    gibbs_all_performances,
    icm_all_performances,
    np_all_performances,
]
all_times = [
    vb_all_times,
    gibbs_all_times,
    icm_all_times,
    np_all_times,
]
colours = ['r','b','g','c']


''' Plot the performances. '''
fig = plt.figure(figsize=(1.9,1.5))
fig.subplots_adjust(left=0.18, right=0.95, bottom=0.17, top=0.95)
plt.xlabel("Time (s)", fontsize=8, labelpad=0)
plt.ylabel("MSE", fontsize=8, labelpad=0)
plt.yticks((0,MSE_max+1),fontsize=6)
plt.xticks(fontsize=6)

for performances, times, colour in zip(all_performances,all_times,colours):
    x, y = times, performances
    plt.plot(x,y,linestyle='-', marker=None, c=colour)
    
plt.xlim(0,time_max)
plt.yticks(range(0,MSE_max+1,100))
plt.ylim(MSE_min,MSE_max)
    
plt.savefig(plot_file, dpi=600)
  