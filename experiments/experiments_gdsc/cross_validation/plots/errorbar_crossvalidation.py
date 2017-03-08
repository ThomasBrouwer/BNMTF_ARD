'''
Script for plotting the average cross-validation results (MSE), with std, on 
the drug sensitivity datasets, for all the methods.
'''

import matplotlib.pyplot as plt
import numpy


''' Method names and performances. '''
method_names = [
    'VB NMF',     'ARD VB NMF',     # 'NMF VB',     'NMF VB ARD',   
    'Gibbs NMF',  'ARD Gibbs NMF',  # 'NMF Gibbs',  'NMF Gibbs ARD', 
    'ICM NMF',    'ARD ICM NMF',    # 'NMF ICM',    'NMF ICM ARD',  
    'NP NMF',                       # 'NMF NP',
    'VB NMTF',    'ARD VB NMTF',    # 'NMTF VB',    'NMTF VB ARD',  
    'Gibbs NMTF', 'ARD Gibbs NMTF', # 'NMTF Gibbs', 'NMTF Gibbs ARD',
    'ICM NMTF',   'ARD ICM NMTF',   # 'NMTF ICM',   'NMTF ICM ARD', 
    'NP NMTF',                      # 'NMTF NP'
]

all_performances_gdsc = [
    # NMF VB
    [685.16092497288184, 710.32299686966451, 703.93385350459334, 729.37119838253295, 694.19032120098836, 717.24799410224693, 709.31261131796532, 703.91053316274144, 690.59053883679485, 704.88806250648452], 
    # NMF VB ARD
    [713.19655462934008, 721.31043774315253, 700.61687793361943, 730.50785633227781, 690.44435527527401, 729.89396636405286, 709.25486450030576, 695.87389520876559, 695.26225316353748, 708.63755041867626],
    # NMF Gibbs
    [716.29910393484386, 715.52845829907073, 703.52578259341101, 709.20707278487555, 687.70064858928674, 693.97202369941476, 706.8950503017204, 715.72301076004032, 711.38252527375471, 706.97860147371944],
    # NMF Gibbs ARD
    [694.90346124983796, 715.52250464628071, 704.55305176478714, 714.98213938693993, 725.41797648404577, 714.44915577933818, 712.5503226263927, 698.81184946087126, 698.07307876833954, 706.26266295425569],
    # NMF ICM
    [729.40046288239785, 728.31355919207101, 691.86118903511647, 715.96727421267792, 720.09430651949663, 706.40449970111422, 712.78407670764443, 737.772158316218, 706.05196536035635, 710.02270641901021],
    # NMF ICM ARD
    [719.7824077508526, 702.05393641620196, 716.98226544179477, 721.75595599822327, 707.45450551121269, 704.35809979582973, 711.96780103652907, 725.73787364118584, 708.53767382589353, 699.44186822909819],
    # NMF NP
    [730.41701997684834, 718.47042584753819, 691.32336665982211, 714.46213589158765, 738.65912996317377, 748.30003701209398, 769.34017968955698, 737.41429097546097, 732.87961285661277, 713.64457275090967],
    # NMTF VB
    [709.66736129233709, 765.88214504613325, 746.70025243132341, 772.27745449563452, 745.41642208861413, 746.63416043672657, 778.08389152148902, 748.30816777944619, 714.12790211075958, 743.14504817463092],
    # NMTF VB ARD
    [722.72999079614908, 698.21931545764801, 737.52356589414251, 704.58457565636195, 703.6715710102967, 719.81797714446282, 713.82496427351339, 721.18206084627707, 716.59301318608493, 704.87671708595326],
    # NMTF Gibbs
    [697.72264470755874, 688.59963116138488, 723.68395059119223, 713.2404399255239, 683.51643895492123, 718.74357305822946, 732.69484500181056, 720.24949905023777, 714.61859728166576, 724.46695154334247],
    # NMTF Gibbs ARD
    [722.32214004260163, 715.16731589342203, 707.82099472214759, 707.16863642869293, 733.6826319505052, 690.37086651359289, 707.11946096310135, 708.39868693728738, 722.03579555089505, 709.30341080247854],
    # NMTF ICM
    [732.69212036935778, 703.59373803276731, 721.57547589519618, 715.59296851929923, 735.58494884486913, 714.06157279712954, 709.41793906913722, 692.15251416044828, 733.59594782945101, 722.39377489936044],
    # NMTF ICM ARD
    [749.09358456994858, 728.48867483636559, 751.71321779677794, 710.14552165194539, 720.68132672814818, 718.33961046515481, 744.25440644549565, 746.00363914072398, 750.51510607673333, 754.19888301307026],
    # NMTF NP
    [690.25343057989005, 736.40038711059503, 764.11082576499348, 704.70177120527262, 732.15532917291273, 742.61597904822986, 716.52571979549577, 713.71528194366965, 733.0727564068369, 728.82412387744012],
]
all_performances_ctrp = all_performances_gdsc
all_performances_ccle_ic = all_performances_gdsc
all_performances_ccle_ec = all_performances_gdsc

colours = ['r','m','b','y','g','k','c']


''' Plot settings. '''
figsize = (8.0, 6.0)
nrows, ncols = 4, 1
left, right, bottom, top = 0.1, 0.99, 0.25, 0.98
plot_file = "./errorbar_crossvalidation.png"

gdsc_MSE_min,    gdsc_MSE_max,    gdsc_step =    650, 800, 50
ctrp_MSE_min,    ctrp_MSE_max,    ctrp_step =    650, 800, 50
ccle_ic_MSE_min, ccle_ic_MSE_max, ccle_ic_step = 650, 800, 50
ccle_ec_MSE_min, ccle_ec_MSE_max, ccle_ec_step = 650, 800, 50


''' Make the plot. '''
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)

# Plot the boxplots
x = method_names
all_eb = []
for i, all_performances in enumerate([all_performances_gdsc, all_performances_ctrp, 
                                      all_performances_ccle_ic, all_performances_ccle_ec]):
    x, y, yerr = range(1,len(method_names)+1), [numpy.mean(p) for p in all_performances], [numpy.std(p) for p in all_performances]
    eb = axes[i].errorbar(x=x, y=y, yerr=yerr)
    all_eb.append(eb)

# Set up the y axes - label and limits
for i, (dataset, MSE_min, MSE_max, MSE_step) in enumerate(zip(
    ['GDSC', 'CTRP', 'CCLE IC50', 'CCLE EC50'],
    [gdsc_MSE_min, ctrp_MSE_min, ccle_ic_MSE_min, ccle_ec_MSE_min],
    [gdsc_MSE_max, ctrp_MSE_max, ccle_ic_MSE_max, ccle_ec_MSE_max],
    [gdsc_step, ctrp_step, ccle_ic_step, ccle_ec_step])):
    
    axes[i].set_ylabel(dataset)
    axes[i].set_yticks(range(MSE_min, MSE_max+1, MSE_step))
    axes[i].set_ylim(MSE_min, MSE_max)
    axes[i].yaxis.set_ticks_position('none')

# Turn off x-axis labels for top 3 plots, and xticks for bottom one
for x in range(2+1):
    axes[x].get_xaxis().set_visible(False)
axes[3].xaxis.set_ticks_position('none')

# Transform xlabels into method names
plt.xticks(range(1,len(method_names)+1), rotation=90)

# Make the lines from dashed into straight line
for eb in all_eb:
    plt.setp(bp['whiskers'], linestyle='-')

# Save plot
plt.savefig(plot_file, dpi=600)