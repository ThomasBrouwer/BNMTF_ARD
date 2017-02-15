'''
Methods for loading in the drug sensitivity datasets.

Rows are cell lines, drugs are columns.

Initially unobserved values are nan. We replace them by 0, and set those mask
entries to 0.

SUMMARY: n_cl, n_drugs, n_entries, fraction_obs
GDSC:    707,  139,     79262,     0.806549103009
CTRP:    887,  545,     387130,    0.800823309165
CCLE IC: 504,  24,      11670,     0.964781746032
CCLE EC: 504,  24,      7626,      0.630456349206

'''
import numpy
import itertools

folder_gdsc_ic50 = './GDSC/processed_all/'
file_gdsc_ic50 = folder_gdsc_ic50+'ic50.txt'

folder_ctrp_ec50 = './CTRP/processed_all/'
file_ctrp_ec50 = folder_ctrp_ec50+'ec50.txt'

folder_ccle_ic50 = './CCLE/processed_all/'
file_ccle_ic50 = folder_ccle_ic50+'ic50.txt'

folder_ccle_ec50 = './CCLE/processed_all/'
file_ccle_ec50 = folder_ccle_ec50+'ec50.txt'

DELIM = '\t'

def load_data_create_mask(location):
    ''' Load in .txt file, and set mask entries for nan to 0. '''
    R = numpy.loadtxt(location, dtype=float, delimiter=DELIM)
    I,J = R.shape
    M = numpy.ones((I,J))
    for i,j in itertools.product(range(I),range(J)):
        if numpy.isnan(R[i,j]):
            R[i,j], M[i,j] = 0., 0.            
    return (R, M)

def load_gdsc_ic50(location=file_gdsc_ic50):
    ''' Return (R_gdsc, M_gdsc). '''
    return load_data_create_mask(location)

def load_ctrp_ec50(location=file_ctrp_ec50):
    ''' Return (R_ctrp, M_ctrp). '''
    return load_data_create_mask(location)

def load_ccle_ic50(location=file_ccle_ic50):
    ''' Return (R_ccle_ic50, M_ccle_ic50). '''
    return load_data_create_mask(location)

def load_ccle_ec50(location=file_ccle_ec50):
    ''' Return (R_ccle_ec50, M_ccle_ec50). '''
    return load_data_create_mask(location)


'''
R_gdsc, M_gdsc = load_gdsc_ic50()
R_ctrp, M_ctrp = load_ctrp_ec50()
R_ccle_ic, M_ccle_ic = load_ccle_ic50()
R_ccle_ec, M_ccle_ec = load_ccle_ec50()
'''