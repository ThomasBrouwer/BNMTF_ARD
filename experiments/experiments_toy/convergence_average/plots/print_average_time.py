"""
Print the average time taken per iteration for each of the methods.
"""

''' Load in the times. '''
folder_results = "./../results/"

vb_nmf_all_times = eval(open(folder_results+'nmf_vb_all_times.txt','r').read())
gibbs_nmf_all_times = eval(open(folder_results+'nmf_gibbs_all_times.txt','r').read())
icm_nmf_all_times = eval(open(folder_results+'nmf_icm_all_times.txt','r').read())
np_nmf_all_times = eval(open(folder_results+'nmf_np_all_times.txt','r').read())

vb_nmtf_all_times = eval(open(folder_results+'nmtf_vb_all_times.txt','r').read())
gibbs_nmtf_all_times = eval(open(folder_results+'nmtf_gibbs_all_times.txt','r').read())
icm_nmtf_all_times = eval(open(folder_results+'nmtf_icm_all_times.txt','r').read())
np_nmtf_all_times = eval(open(folder_results+'nmtf_np_all_times.txt','r').read())


''' Compute average time per iteration. '''
time_nmf_vb = vb_nmf_all_times[-1] / len(vb_nmf_all_times)
time_nmf_gibbs = gibbs_nmf_all_times[-1] / len(gibbs_nmf_all_times)
time_nmf_icm = icm_nmf_all_times[-1] / len(icm_nmf_all_times)
time_nmf_np = np_nmf_all_times[-1] / len(np_nmf_all_times)

time_nmtf_vb = vb_nmtf_all_times[-1] / len(vb_nmtf_all_times)
time_nmtf_gibbs = gibbs_nmtf_all_times[-1] / len(gibbs_nmtf_all_times)
time_nmtf_icm = icm_nmtf_all_times[-1] / len(icm_nmtf_all_times)
time_nmtf_np = np_nmtf_all_times[-1] / len(np_nmtf_all_times)


''' Print average time per iteration. '''
print('NMF VB: %s.' % time_nmf_vb)
print('NMF Gibbs: %s.' % time_nmf_gibbs)
print('NMF ICM: %s.' % time_nmf_icm)
print('NMF NP: %s.' % time_nmf_np)

print('NMTF VB: %s.' % time_nmtf_vb)
print('NMTF Gibbs: %s.' % time_nmtf_gibbs)
print('NMTF ICM: %s.' % time_nmtf_icm)
print('NMTF NP: %s.' % time_nmtf_np)