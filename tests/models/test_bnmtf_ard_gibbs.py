"""
Tests for the BNMTF+ARD Gibbs sampler.
"""

import sys
sys.path.append("/home/tab43/Documents/Projects/libraries/")
from BNMTF_ARD.code.models.bnmtf_ard_gibbs import bnmtf_ard_gibbs

import numpy, math, pytest, itertools


""" Test constructor """
def test_init():
    # Test getting an exception when R and M are different sizes, and when R is not a 2D array.
    R1 = numpy.ones(3)
    M = numpy.ones((2,3))
    I,J,K,L = 5,3,1,2
    alpha0, beta0 = 4., 2.
    alphaR, betaR = 3., 1.    
    priors = { 'alpha0':alpha0, 'beta0':beta0, 'alphaR':alphaR, 'betaR':betaR }
    
    with pytest.raises(AssertionError) as error:
        bnmtf_ard_gibbs(R1,M,K,L,priors)
    assert str(error.value) == "Input matrix R is not a two-dimensional array, but instead 1-dimensional."
    
    R2 = numpy.ones((4,3,2))
    with pytest.raises(AssertionError) as error:
        bnmtf_ard_gibbs(R2,M,K,L,priors)
    assert str(error.value) == "Input matrix R is not a two-dimensional array, but instead 3-dimensional."
    
    R3 = numpy.ones((3,2))
    with pytest.raises(AssertionError) as error:
        bnmtf_ard_gibbs(R3,M,K,L,priors)
    assert str(error.value) == "Input matrix R is not of the same size as the indicator matrix M: (3, 2) and (2, 3) respectively."
    
    # Test getting an exception if a row or column is entirely unknown
    R4 = numpy.ones((2,3))
    M1 = [[1,1,1],[0,0,0]]
    M2 = [[1,1,0],[1,0,0]]
    
    with pytest.raises(AssertionError) as error:
        bnmtf_ard_gibbs(R4,M1,K,L,priors)
    assert str(error.value) == "Fully unobserved row in R, row 1."
    with pytest.raises(AssertionError) as error:
        bnmtf_ard_gibbs(R4,M2,K,L,priors)
    assert str(error.value) == "Fully unobserved column in R, column 2."
    
    # Finally, a successful case
    I,J,K,L = 3,2,2,2
    R5 = 2*numpy.ones((I,J))
    M = numpy.ones((I,J))
    BNMTF = bnmtf_ard_gibbs(R5,M,K,L,priors)
    
    assert numpy.array_equal(BNMTF.R,R5)
    assert numpy.array_equal(BNMTF.M,M)
    assert BNMTF.I == I
    assert BNMTF.J == J
    assert BNMTF.K == K
    assert BNMTF.L == L
    assert BNMTF.size_Omega == I*J
    assert BNMTF.alphaR == alphaR
    assert BNMTF.betaR == betaR
    assert BNMTF.alpha0 == alpha0
    assert BNMTF.beta0 == beta0
    
    
""" Test initialing parameters """
def test_initialise():
    I,J,K,L = 5,3,2,4
    R = numpy.ones((I,J))
    M = numpy.ones((I,J))
    
    alpha0, beta0 = 4., 2.
    alphaR, betaR = 3., 1.    
    priors = { 'alpha0':alpha0, 'beta0':beta0, 'alphaR':alphaR, 'betaR':betaR }
    
    # First do a random initialisation - we can then only check whether values are correctly initialised
    init = 'random'
    BNMTF = bnmtf_ard_gibbs(R,M,K,L,priors)
    BNMTF.initialise(init)
    
    assert BNMTF.tau >= 0.0
    for k in xrange(0,K):
        assert BNMTF.lambdaF[k] >= 0.0
        for i in xrange(0,I):
            assert BNMTF.F[i,k] >= 0.0
    for l in xrange(0,L):
        assert BNMTF.lambdaG[l] >= 0.0
        for j in xrange(0,J):
            assert BNMTF.G[j,l] >= 0.0
    for k,l in itertools.product(xrange(0,K),xrange(0,L)):
        assert BNMTF.lambdaS[k,l] >= 0.0
        assert BNMTF.S[k,l] >= 0.0
        
    # Expectation
    init = 'exp'
    BNMTF = bnmtf_ard_gibbs(R,M,K,L,priors)
    BNMTF.initialise(init)
    
    assert BNMTF.tau >= 0.0
    for k in xrange(0,K):
        assert BNMTF.lambdaF[k] >= 0.0
    for l in xrange(0,L):
        assert BNMTF.lambdaG[l] >= 0.0
    for k,l in itertools.product(xrange(0,K),xrange(0,L)):
        assert BNMTF.lambdaS[k,l] >= 0.0
    
    for i,k in itertools.product(xrange(0,I),xrange(0,K)):
        assert BNMTF.F[i,k] == 1. / BNMTF.lambdaF[k]
    for k,l in itertools.product(xrange(0,K),xrange(0,L)):
        assert BNMTF.S[k,l] == 1. / BNMTF.lambdaS[k,l]
    for j,l in itertools.product(xrange(0,J),xrange(0,L)):
        assert BNMTF.G[j,l] == 1. / BNMTF.lambdaG[l]
    
    # Initialisation of F and G using Kmeans
    init = 'kmeans'
    BNMTF = bnmtf_ard_gibbs(R,M,K,L,priors)
    BNMTF.initialise(init)
    
    assert BNMTF.tau >= 0.0
    
    for k in xrange(0,K):
        assert BNMTF.lambdaF[k] >= 0.0
    for l in xrange(0,L):
        assert BNMTF.lambdaG[l] >= 0.0
    for k,l in itertools.product(xrange(0,K),xrange(0,L)):
        assert BNMTF.lambdaS[k,l] >= 0.0
        
    for i,k in itertools.product(xrange(0,I),xrange(0,K)):
        assert BNMTF.F[i,k] == 0.2 or BNMTF.F[i,k] == 1.2
    for j,l in itertools.product(xrange(0,J),xrange(0,L)):
        assert BNMTF.G[j,l] == 0.2 or BNMTF.G[j,l] == 1.2
    for k,l in itertools.product(xrange(0,K),xrange(0,L)):
        assert BNMTF.S[k,l] >= 0.0
    
    
""" Test computing values for alpha, beta, mu, tau. """
I,J,K,L = 5,3,2,4
R = numpy.ones((I,J))
M = numpy.ones((I,J))
M[0,0], M[2,2], M[3,1] = 0, 0, 0

alpha0, beta0 = 2., 4.
alphaR, betaR = 3., 1.    
priors = { 'alpha0':alpha0, 'beta0':beta0, 'alphaR':alphaR, 'betaR':betaR }
init = 'exp'
# F = 1/2, S = 1/3, G = 1/5
F, S, G = 1./2.*numpy.ones((I,K)), 1./3.*numpy.ones((K,L)), 1./5.*numpy.ones((J,L))
# R - FSG.T = [[1]] - [[4/15]] = [[11/15]]

def test_alpha_s():
    BNMTF = bnmtf_ard_gibbs(R,M,K,L,priors)
    BNMTF.initialise(init)
    BNMTF.tau = 3.
    alphaR_s = alphaR + 6.
    assert BNMTF.alphaR_s() == alphaR_s

def test_beta_s():
    BNMTF = bnmtf_ard_gibbs(R,M,K,L,priors)
    BNMTF.initialise(init)
    BNMTF.tau, BNMTF.F, BNMTF.S, BNMTF.G = 3., F, S, G
    betaR_s = betaR + .5*(12*(11./15.)**2) #F*S = [[1/6+1/6=1/3,..]], F*S*G^T = [[1/15*4=4/15,..]]
    assert abs(BNMTF.betaR_s() - betaR_s) < 0.00000000000001
    
def test_tauF():
    BNMTF = bnmtf_ard_gibbs(R,M,K,L,priors)
    BNMTF.initialise(init)
    BNMTF.tau, BNMTF.F, BNMTF.S, BNMTF.G = 3., F, S, G
    # S*G.T = [[4/15]], (S*G.T)^2 = [[16/225]], sum_j S*G.T = [[32/225,32/225],[48/225,48/225],[32/225,32/225],[32/225,32/225],[48/225,48/225]]
    tauF = 3.*numpy.array([[32./225.,32./225.],[48./225.,48./225.],[32./225.,32./225.],[32./225.,32./225.],[48./225.,48./225.]])
    for i,k in itertools.product(xrange(0,I),xrange(0,K)):
        assert abs(BNMTF.tauF(k)[i] - tauF[i,k]) < 0.000000000000001
        
def test_muF():
    BNMTF = bnmtf_ard_gibbs(R,M,K,L,priors)
    BNMTF.initialise(init)
    BNMTF.tau, BNMTF.F, BNMTF.S, BNMTF.G = 3., F, S, G
    tauF = 3.*numpy.array([[32./225.,32./225.],[48./225.,48./225.],[32./225.,32./225.],[32./225.,32./225.],[48./225.,48./225.]])
    # Rij - Fi*S*Gj + Fik(Sk*Gj) = 11/15 + 1/2 * 4/15 = 13/15
    # (Rij - Fi*S*Gj + Fik(Sk*Gj)) * (Sk*Gj) = 13/15 * 4/15 = 52/225
    muF = 1./tauF * ( 3. * numpy.array([[2*(52./225.),2*(52./225.)],[3*(52./225.),3*(52./225.)],[2*(52./225.),2*(52./225.)],[2*(52./225.),2*(52./225.)],[3*(52./225.),3*(52./225.)]]) - alpha0 / beta0 )
    for i,k in itertools.product(xrange(0,I),xrange(0,K)):
        assert abs(BNMTF.muF(tauF[:,k],k)[i] - muF[i,k]) < 0.000000000000001
        
def test_alphaF():
    BNMTF = bnmtf_ard_gibbs(R,M,K,L,priors)
    BNMTF.initialise(init)
    alphaF = [alpha0 + I, alpha0 + I]
    assert numpy.array_equal(alphaF, [BNMTF.alphaF(0),BNMTF.alphaF(1)])
        
def test_betaF():
    BNMTF = bnmtf_ard_gibbs(R,M,K,L,priors)
    BNMTF.initialise(init)
    BNMTF.F, BNMTF.S, BNMTF.G = F, S, G
    betaF = [beta0 + I*1./2., beta0 + I*1./2.]
    assert numpy.array_equal(betaF, [BNMTF.betaF(0),BNMTF.betaF(1)])
    
def test_tauS():
    BNMTF = bnmtf_ard_gibbs(R,M,K,L,priors)
    BNMTF.initialise(init)
    BNMTF.tau, BNMTF.F, BNMTF.S, BNMTF.G = 3., F, S, G
    # F outer G = [[1/10]], (F outer G)^2 = [[1/100]], sum (F outer G)^2 = [[12/100]]
    tauS = 3.*numpy.array([[3./25.,3./25.,3./25.,3./25.],[3./25.,3./25.,3./25.,3./25.]])
    for k,l in itertools.product(xrange(0,K),xrange(0,L)):
        assert abs(BNMTF.tauS(k,l) - tauS[k,l]) < 0.000000000000001
    
def test_muS():
    BNMTF = bnmtf_ard_gibbs(R,M,K,L,priors)
    BNMTF.initialise(init)
    BNMTF.tau, BNMTF.F, BNMTF.S, BNMTF.G = 3., F, S, G
    tauS = 3.*numpy.array([[3./25.,3./25.,3./25.,3./25.],[3./25.,3./25.,3./25.,3./25.]])
    # Rij - Fi*S*Gj + Fik*Skl*Gjk = 11/15 + 1/2*1/3*1/5 = 23/30
    # (Rij - Fi*S*Gj + Fik*Skl*Gjk) * Fik*Gjk = 23/30 * 1/10 = 23/300
    muS = 1./tauS * ( 3. * numpy.array([[12*23./300.,12*23./300.,12*23./300.,12*23./300.],[12*23./300.,12*23./300.,12*23./300.,12*23./300.]]) - alpha0 / beta0 )
    for k,l in itertools.product(xrange(0,K),xrange(0,L)):
        assert abs(BNMTF.muS(tauS[k,l],k,l) - muS[k,l]) < 0.000000000000001
        
def test_alphaS():
    BNMTF = bnmtf_ard_gibbs(R,M,K,L,priors)
    BNMTF.initialise(init)
    alphaS = numpy.array([[alpha0 + 1. for l in range(0,L)] for k in range(0,K)])
    for k,l in itertools.product(xrange(0,K),xrange(0,L)):
        assert numpy.array_equal(alphaS[k,l], BNMTF.alphaS(k,l))
        
def test_betaS():
    BNMTF = bnmtf_ard_gibbs(R,M,K,L,priors)
    BNMTF.initialise(init)
    BNMTF.F, BNMTF.S, BNMTF.G = F, S, G
    betaS = numpy.array([[beta0 + S[k,l] for l in range(0,L)] for k in range(0,K)])
    for k,l in itertools.product(xrange(0,K),xrange(0,L)):
        assert numpy.array_equal(betaS[k,l], BNMTF.betaS(k,l))
        
def test_tauG():
    BNMTF = bnmtf_ard_gibbs(R,M,K,L,priors)
    BNMTF.initialise(init)
    BNMTF.tau, BNMTF.F, BNMTF.S, BNMTF.G = 3., F, S, G
    # F*S = [[1/3]], (F*S)^2 = [[1/9]], sum_i F*S = [[4/9]]
    tauG = 3.*numpy.array([[4./9.,4./9.,4./9.,4./9.],[4./9.,4./9.,4./9.,4./9.],[4./9.,4./9.,4./9.,4./9.]])
    for j,l in itertools.product(xrange(0,J),xrange(0,L)):
        assert BNMTF.tauG(l)[j] == tauG[j,l]
    
def test_muG():
    BNMTF = bnmtf_ard_gibbs(R,M,K,L,priors)
    BNMTF.initialise(init)
    BNMTF.tau, BNMTF.F, BNMTF.S, BNMTF.G = 3., F, S, G
    tauG = 3.*numpy.array([[4./9.,4./9.,4./9.,4./9.],[4./9.,4./9.,4./9.,4./9.],[4./9.,4./9.,4./9.,4./9.]])
    # Rij - Fi*S*Gj + Gjl*(Fi*Sl)) = 11/15 + 1/5 * 1/3 = 12/15 = 4/5
    # (Rij - Fi*S*Gj + Gjl*(Fi*Sl)) * (Fi*Sl) = 4/5 * 1/3 = 4/15
    muG = 1./tauG * ( 3. * numpy.array([[4.*4./15.,4.*4./15.,4.*4./15.,4.*4./15.],[4.*4./15.,4.*4./15.,4.*4./15.,4.*4./15.],[4.*4./15.,4.*4./15.,4.*4./15.,4.*4./15.]]) - alpha0 / beta0 )
    for j,l in itertools.product(xrange(0,J),xrange(0,L)):
        assert abs(BNMTF.muG(tauG[:,l],l)[j] - muG[j,l]) < 0.000000000000001
        
def test_alphaG():
    BNMTF = bnmtf_ard_gibbs(R,M,K,L,priors)
    BNMTF.initialise(init)
    alphaG = [alpha0 + J, alpha0 + J, alpha0 + J, alpha0 + J]
    assert numpy.array_equal(alphaG, [BNMTF.alphaG(0),BNMTF.alphaG(1),BNMTF.alphaG(2),BNMTF.alphaG(3)])
        
def test_betaG():
    BNMTF = bnmtf_ard_gibbs(R,M,K,L,priors)
    BNMTF.initialise(init)
    BNMTF.F, BNMTF.S, BNMTF.G = F, S, G
    betaG = [beta0 + J*1./5., beta0 + J*1./5., beta0 + J*1./5., beta0 + J*1./5.]
    assert numpy.array_equal(betaG, [BNMTF.betaG(0),BNMTF.betaG(1),BNMTF.betaG(2),BNMTF.betaG(3)])
      
      
""" Test some iterations, and that the values have changed in F, S, G. """
def test_run():
    I,J,K,L = 10,5,3,2
    R = numpy.ones((I,J))
    M = numpy.ones((I,J))
    M[0,0], M[2,2], M[3,1] = 0, 0, 0
    
    alpha0, beta0 = 4., 2.
    alphaR, betaR = 3., 1.    
    priors = { 'alpha0':alpha0, 'beta0':beta0, 'alphaR':alphaR, 'betaR':betaR }
    init = 'exp'
    iterations = 15
    F, S, G = numpy.ones((I,K))/2., numpy.ones((K,L))/3., numpy.ones((J,L))/4.
    
    BNMTF = bnmtf_ard_gibbs(R,M,K,L,priors)
    BNMTF.initialise(init)
    BNMTF.F, BNMTF.S, BNMTF.G = numpy.copy(F), numpy.copy(S), numpy.copy(G)
    
    BNMTF.run(iterations)
    
    assert BNMTF.all_F.shape == (iterations,I,K)
    assert BNMTF.all_lambdaF.shape == (iterations,K)
    assert BNMTF.all_S.shape == (iterations,K,L)
    assert BNMTF.all_lambdaS.shape == (iterations,K,L)
    assert BNMTF.all_G.shape == (iterations,J,L)
    assert BNMTF.all_lambdaG.shape == (iterations,L)
    assert BNMTF.all_tau.shape == (iterations,)
    
    for i,k in itertools.product(xrange(0,I),xrange(0,K)):
        assert BNMTF.all_F[0,i,k] != F[i,k]
    for k,l in itertools.product(xrange(0,K),xrange(0,L)):
        assert BNMTF.all_S[0,k,l] != S[k,l]
    for j,l in itertools.product(xrange(0,J),xrange(0,L)):
        assert BNMTF.all_G[0,j,l] != G[j,l]
    
    for it in range(1,iterations):
        assert BNMTF.all_tau[it] != alphaR / float(betaR)
        for k in xrange(0,K):
            assert BNMTF.all_lambdaF[0,k] != alpha0 / float(beta0)
            for i in xrange(0,I):
                assert BNMTF.all_F[it,i,k] != F[i,k]
        for l in xrange(0,L):
            assert BNMTF.all_lambdaG[0,l] != alpha0 / float(beta0)
            for j in xrange(0,J):
                assert BNMTF.all_G[it,j,l] != G[j,l]
        for k,l in itertools.product(xrange(0,K),xrange(0,L)):
            BNMTF.all_lambdaS[it,k,l] != alpha0 / float(beta0)
            BNMTF.all_S[it,k,l] != S[k,l]
    
    
""" Test approximating the expectations for F, S, G, tau """
def test_approx_expectation():
    burn_in = 2
    thinning = 3 # so index 2,5,8 -> m=3,m=6,m=9
    (I,J,K,L) = (5,3,2,4)
    
    Fs = [numpy.ones((I,K)) * 3*m**2 for m in range(1,10+1)] 
    lambdaFs = [numpy.ones(K) * 3*m**2 for m in range(1,10+1)] 
    Ss = [numpy.ones((K,L)) * 2*m**2 for m in range(1,10+1)]
    lambdaSs = [numpy.ones((K,L)) * 2*m**2 for m in range(1,10+1)]
    Gs = [numpy.ones((J,L)) * 1*m**2 for m in range(1,10+1)]
    lambdaGs = [numpy.ones(L) * 1*m**2 for m in range(1,10+1)] #first is 1's, second is 4's, third is 9's, etc.
    taus = [m**2 for m in range(1,10+1)]
    
    expected_exp_tau = (9.+36.+81.)/3.
    expected_exp_F = numpy.array([[9.+36.+81.,9.+36.+81.],[9.+36.+81.,9.+36.+81.],[9.+36.+81.,9.+36.+81.],[9.+36.+81.,9.+36.+81.],[9.+36.+81.,9.+36.+81.]])
    expected_exp_lambdaF = numpy.array([9.+36.+81.,9.+36.+81.])
    expected_exp_S = numpy.array([[(9.+36.+81.)*(2./3.),(9.+36.+81.)*(2./3.),(9.+36.+81.)*(2./3.),(9.+36.+81.)*(2./3.)],[(9.+36.+81.)*(2./3.),(9.+36.+81.)*(2./3.),(9.+36.+81.)*(2./3.),(9.+36.+81.)*(2./3.)]])
    expected_exp_lambdaS = numpy.array([[(9.+36.+81.)*(2./3.),(9.+36.+81.)*(2./3.),(9.+36.+81.)*(2./3.),(9.+36.+81.)*(2./3.)],[(9.+36.+81.)*(2./3.),(9.+36.+81.)*(2./3.),(9.+36.+81.)*(2./3.),(9.+36.+81.)*(2./3.)]])
    expected_exp_G = numpy.array([[(9.+36.+81.)*(1./3.),(9.+36.+81.)*(1./3.),(9.+36.+81.)*(1./3.),(9.+36.+81.)*(1./3.)],[(9.+36.+81.)*(1./3.),(9.+36.+81.)*(1./3.),(9.+36.+81.)*(1./3.),(9.+36.+81.)*(1./3.)],[(9.+36.+81.)*(1./3.),(9.+36.+81.)*(1./3.),(9.+36.+81.)*(1./3.),(9.+36.+81.)*(1./3.)]])
    expected_exp_lambdaG = numpy.array([(9.+36.+81.)*(1./3.),(9.+36.+81.)*(1./3.),(9.+36.+81.)*(1./3.),(9.+36.+81.)*(1./3.)])
    
    R = numpy.ones((I,J))
    M = numpy.ones((I,J))
    alpha0, beta0 = 4., 2.
    alphaR, betaR = 3., 1.    
    priors = { 'alpha0':alpha0, 'beta0':beta0, 'alphaR':alphaR, 'betaR':betaR }
    
    BNMTF = bnmtf_ard_gibbs(R,M,K,L,priors)
    BNMTF.all_tau, BNMTF.all_F, BNMTF.all_S, BNMTF.all_G = taus, Fs, Ss, Gs
    BNMTF.all_lambdaF, BNMTF.all_lambdaS, BNMTF.all_lambdaG = lambdaFs, lambdaSs, lambdaGs
    
    (exp_tau, exp_F, exp_S, exp_G, exp_lambdaF, exp_lambdaS, exp_lambdaG) = BNMTF.approx_expectation(burn_in,thinning)
    
    assert expected_exp_tau == exp_tau
    assert numpy.array_equal(expected_exp_F,exp_F)
    assert numpy.array_equal(expected_exp_lambdaF,exp_lambdaF)
    assert numpy.array_equal(expected_exp_S,exp_S)
    assert numpy.array_equal(expected_exp_lambdaS,exp_lambdaS)
    assert numpy.array_equal(expected_exp_G,exp_G)
    assert numpy.array_equal(expected_exp_lambdaG,exp_lambdaG)

    
""" Test computing the performance of the predictions using the expectations """
def test_predict():
    burn_in = 2
    thinning = 3 # so index 2,5,8 -> m=3,m=6,m=9
    (I,J,K,L) = (5,3,2,4)
    
    Fs = [numpy.ones((I,K)) * 3*m**2 for m in range(1,10+1)] 
    Fs[2][0,0] = 24 #instead of 27 - to ensure we do not get 0 variance in our predictions
    lambdaFs = [numpy.ones(K) * 3*m**2 for m in range(1,10+1)] 
    Ss = [numpy.ones((K,L)) * 2*m**2 for m in range(1,10+1)]
    lambdaSs = [numpy.ones((K,L)) * 2*m**2 for m in range(1,10+1)]
    Gs = [numpy.ones((J,L)) * 1*m**2 for m in range(1,10+1)]
    lambdaGs = [numpy.ones(L) * 1*m**2 for m in range(1,10+1)] #first is 1's, second is 4's, third is 9's, etc.
    taus = [m**2 for m in range(1,10+1)]
    
    R = numpy.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15]],dtype=float)
    M = numpy.ones((I,J))
    alpha0, beta0 = 4., 2.
    alphaR, betaR = 3., 1.    
    priors = { 'alpha0':alpha0, 'beta0':beta0, 'alphaR':alphaR, 'betaR':betaR }
    
    #expected_exp_F = numpy.array([[125.,126.],[126.,126.],[126.,126.],[126.,126.],[126.,126.]])
    #expected_exp_S = numpy.array([[84.,84.,84.,84.],[84.,84.,84.,84.]])
    #expected_exp_G = numpy.array([[42.,42.,42.,42.],[42.,42.,42.,42.],[42.,42.,42.,42.]])
    #R_pred = numpy.array([[ 3542112.,  3542112.,  3542112.],[ 3556224.,  3556224.,  3556224.],[ 3556224.,  3556224.,  3556224.],[ 3556224.,  3556224.,  3556224.],[ 3556224.,  3556224.,  3556224.]])
       
    M_test = numpy.array([[0,0,1],[0,1,0],[0,0,0],[1,1,0],[0,0,0]]) #R->3,5,10,11, R_pred->3542112,3556224,3556224,3556224
    MSE = ((3.-3542112.)**2 + (5.-3556224.)**2 + (10.-3556224.)**2 + (11.-3556224.)**2) / 4.
    R2 = 1. - ((3.-3542112.)**2 + (5.-3556224.)**2 + (10.-3556224.)**2 + (11.-3556224.)**2) / (4.25**2+2.25**2+2.75**2+3.75**2) #mean=7.25
    Rp = 357. / ( math.sqrt(44.75) * math.sqrt(5292.) ) #mean=7.25,var=44.75, mean_pred=3552696,var_pred=5292, corr=(-4.25*-63 + -2.25*21 + 2.75*21 + 3.75*21)
    
    BNMTF = bnmtf_ard_gibbs(R,M,K,L,priors)
    BNMTF.all_tau, BNMTF.all_F, BNMTF.all_S, BNMTF.all_G = taus, Fs, Ss, Gs
    BNMTF.all_lambdaF, BNMTF.all_lambdaS, BNMTF.all_lambdaG = lambdaFs, lambdaSs, lambdaGs
    
    performances = BNMTF.predict(M_test,burn_in,thinning)
    
    assert performances['MSE'] == MSE
    assert performances['R^2'] == R2
    assert performances['Rp'] == Rp
    
    
""" Test the evaluation measures MSE, R^2, Rp """
def test_compute_statistics():
    R = numpy.array([[1,2],[3,4]],dtype=float)
    M = numpy.array([[1,1],[0,1]])
    I, J, K, L = 2, 2, 3, 4
    
    alpha0, beta0 = 4., 2.
    alphaR, betaR = 3., 1.    
    priors = { 'alpha0':alpha0, 'beta0':beta0, 'alphaR':alphaR, 'betaR':betaR }
    
    BNMTF = bnmtf_ard_gibbs(R,M,K,L,priors)
    
    R_pred = numpy.array([[500,550],[1220,1342]],dtype=float)
    M_pred = numpy.array([[0,0],[1,1]])
    
    MSE_pred = (1217**2 + 1338**2) / 2.0
    R2_pred = 1. - (1217**2+1338**2)/(0.5**2+0.5**2) #mean=3.5
    Rp_pred = 61. / ( math.sqrt(.5) * math.sqrt(7442.) ) #mean=3.5,var=0.5,mean_pred=1281,var_pred=7442,cov=61
    
    assert MSE_pred == BNMTF.compute_MSE(M_pred,R,R_pred)
    assert R2_pred == BNMTF.compute_R2(M_pred,R,R_pred)
    assert Rp_pred == BNMTF.compute_Rp(M_pred,R,R_pred)
    
    
""" Test the model quality measures. """
def test_log_likelihood():
    R = numpy.array([[1,2],[3,4]],dtype=float)
    M = numpy.array([[1,1],[0,1]])
    I, J, K, L = 2, 2, 3, 4
    
    alpha0, beta0 = 4., 2.
    alphaR, betaR = 3., 1.    
    priors = { 'alpha0':alpha0, 'beta0':beta0, 'alphaR':alphaR, 'betaR':betaR }
    
    iterations = 10
    burnin, thinning = 4, 2
    BNMTF = bnmtf_ard_gibbs(R,M,K,L,priors)
    
    BNMTF.all_F = [numpy.ones((I,K)) for i in range(0,iterations)]
    BNMTF.all_lambdaF = [numpy.ones(K) for i in range(0,iterations)]
    BNMTF.all_S = [2*numpy.ones((K,L)) for i in range(0,iterations)]
    BNMTF.all_lambdaS = [2*numpy.ones((K,L)) for i in range(0,iterations)]
    BNMTF.all_G = [3*numpy.ones((J,L)) for i in range(0,iterations)]
    BNMTF.all_lambdaG = [3*numpy.ones(L) for i in range(0,iterations)]
    BNMTF.all_tau = [3. for i in range(0,iterations)]
    # expF*expS*expG.T = [[72.]]
    
    log_likelihood = 3./2.*(math.log(3)-math.log(2*math.pi)) - 3./2. * (71**2 + 70**2 + 68**2)
    AIC = -2*log_likelihood + 2*(2*3+3*4+2*4)
    BIC = -2*log_likelihood + (2*3+3*4+2*4)*math.log(3)
    MSE = (71**2+70**2+68**2)/3.
    
    assert log_likelihood == BNMTF.quality('loglikelihood',burnin,thinning)
    assert AIC == BNMTF.quality('AIC',burnin,thinning)
    assert BIC == BNMTF.quality('BIC',burnin,thinning)
    assert MSE == BNMTF.quality('MSE',burnin,thinning)
    with pytest.raises(AssertionError) as error:
        BNMTF.quality('FAIL',burnin,thinning)
    assert str(error.value) == "Unrecognised metric for model quality: FAIL."