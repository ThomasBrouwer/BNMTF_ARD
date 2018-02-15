'''
Methods for plotting the distribution of the drug sensitivity datasets.
'''

import sys, os
project_location = os.path.dirname(__file__)+"/../../../../"
sys.path.append(project_location)

from BNMTF_ARD.data.drug_sensitivity.load_data import load_gdsc_ic50
from BNMTF_ARD.data.drug_sensitivity.load_data import load_ctrp_ec50
from BNMTF_ARD.data.drug_sensitivity.load_data import load_ccle_ic50
from BNMTF_ARD.data.drug_sensitivity.load_data import load_ccle_ec50

import itertools
import matplotlib.pyplot as plt


''' Load in the data. '''
R_gdsc, M_gdsc = load_gdsc_ic50()
R_ctrp, M_ctrp = load_ctrp_ec50()
R_ccle_ic, M_ccle_ic = load_ccle_ic50()
R_ccle_ec, M_ccle_ec = load_ccle_ec50()

def extract_values(R, M):
    I, J = R.shape
    return [R[i,j] for i,j in itertools.product(range(I),range(J)) if M[i,j]]

values_plotnames_bins = [
    (extract_values(R_gdsc, M_gdsc), 'distribution_gdsc_ic50.pdf', [v-0.5 for v in range(0,100+10,5)]),
    (extract_values(R_ctrp, M_ctrp), 'distribution_ctrp_ec50.pdf', [v-0.5 for v in range(0,100+10,5)]),
    (extract_values(R_ccle_ic, M_ccle_ic), 'distribution_ccle_ic50.pdf', [v-0.5 for v in range(0,8+2)]),
    (extract_values(R_ccle_ec, M_ccle_ec), 'distribution_ccle_ec50.pdf', [v-0.5 for v in range(0,10+2)]),
]


''' Make the plots. '''
for values, plotname, bins in values_plotnames_bins:
    fig = plt.figure(figsize=(2, 1.5))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.1, top=0.99)
    plt.hist(values, bins=bins)
    
    plt.xticks(fontsize=8)
    plt.yticks([], fontsize=8)
    plt.xlim(bins[0], bins[-1])
    
    plt.savefig(plotname, dpi=600)