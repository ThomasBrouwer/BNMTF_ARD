"""
Generate a toy dataset for the BNMTF ARD model, using the model's assumptions.
We use alpha0 = beta0 = 1e-14.
"""

project_location = "/home/tab43/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

from BNMTF.code.distributions.exponential import exponential_draw
from BNMTF.code.distributions.normal import normal_draw
from BNMTF.code.distributions.gamma import gamma_draw
from ml_helpers.code.mask import generate_M

import numpy, itertools, matplotlib.pyplot as plt

def generate_dataset(I,J,K,L,alpha0,beta0,tau):
    ''' Generate the datasets: (lambdaF,F,lambdaS,S,lambdaG,G,tau,R_true,R). '''
    # Generate lambdaF, lambdaS, lambdaG   
    lambdaF = numpy.array([gamma_draw(alpha0,beta0) for k in range(0,K)])
    lambdaS = numpy.array([[gamma_draw(alpha0,beta0) for l in range(0,L)] for k in range(0,K)])
    lambdaG = numpy.array([gamma_draw(alpha0,beta0) for l in range(0,L)])
        
    # Generate F, S, G
    F = numpy.array([[exponential_draw(lambdaF[k]) for k in range(0,K)] for i in range(0,I)])
    S = numpy.array([[exponential_draw(lambdaS[k,l]) for l in range(0,L)] for k in range(0,K)])
    G = numpy.array([[exponential_draw(lambdaG[l]) for l in range(0,L)] for j in range(0,J)])
        
    # Generate R
    R_true = numpy.dot(F,numpy.dot(S,G.T))
    R = add_noise(R_true,tau) 
        
    return (lambdaF,F,lambdaS,S,lambdaG,G,tau,R_true,R)
    
def add_noise(R_true,tau):
    if numpy.isinf(tau):
        return numpy.copy(R_true)
    
    (I,J) = R_true.shape
    R = numpy.zeros((I,J))
    for i,j in itertools.product(xrange(0,I),xrange(0,J)):
        R[i,j] = normal_draw(R_true[i,j],tau)
    return R
    
def try_generate_M(I,J,fraction_unknown,attempts):
    for attempt in range(1,attempts+1):
        try:
            M = generate_M(I,J,fraction_unknown)
            sums_columns = M.sum(axis=0)
            sums_rows = M.sum(axis=1)
            for i,c in enumerate(sums_rows):
                assert c != 0, "Fully unobserved row in M, row %s. Fraction %s." % (i,fraction_unknown)
            for j,c in enumerate(sums_columns):
                assert c != 0, "Fully unobserved column in M, column %s. Fraction %s." % (j,fraction_unknown)
            print "Took %s attempts to generate M." % attempt
            return M
        except AssertionError:
            pass
    raise Exception("Tried to generate M %s times, with I=%s, J=%s, fraction=%s, but failed." % (attempts,I,J,fraction_unknown))
      
##########

if __name__ == "__main__":
    output_folder = project_location+"BNMTF_ARD/experiments/generate_toy/bnmtf_ard/"

    I,J,K,L = 100, 80, 5, 5
    fraction_unknown = 0.1
    
    alpha0, beta0 = 1., 1. #0.1, 0.1 #1e-14, 1e-14
    alphaR, betaR = 1., 1.
    tau = alphaR / betaR
    
    (lambdaF,F,lambdaS,S,lambdaG,G,tau,R_true,R) = generate_dataset(I,J,K,L,alpha0,beta0,tau)
    
    # Try to generate M
    M = try_generate_M(I,J,fraction_unknown,attempts=1000)
    
    # Store all matrices in text files
    numpy.savetxt(open(output_folder+"lambdaF.txt",'w'),lambdaF)
    numpy.savetxt(open(output_folder+"F.txt",'w'),F)
    numpy.savetxt(open(output_folder+"lambdaS.txt",'w'),lambdaS)
    numpy.savetxt(open(output_folder+"S.txt",'w'),S)
    numpy.savetxt(open(output_folder+"lambdaG.txt",'w'),lambdaG)
    numpy.savetxt(open(output_folder+"G.txt",'w'),G)
    numpy.savetxt(open(output_folder+"R_true.txt",'w'),R_true)
    numpy.savetxt(open(output_folder+"R.txt",'w'),R)
    numpy.savetxt(open(output_folder+"M.txt",'w'),M)
    
    print "Mean R: %s. Variance R: %s. Min R: %s. Max R: %s." % (numpy.mean(R),numpy.var(R),R.min(),R.max())
    fig = plt.figure()
    plt.hist(R.flatten(),bins=range(0,int(R.max())+1,10))
    plt.show()