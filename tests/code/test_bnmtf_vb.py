"""
Tests for the BNMTF Variational Bayes algorithm.
"""

import sys, os
project_location = os.path.dirname(__file__)+"/../../../"
sys.path.append(project_location)

import numpy, math, pytest, itertools
from BNMTF_ARD.code.models.bnmtf_vb import bnmtf_vb


""" Test constructor """
def test_init():
    # Test getting an exception when R and M are different sizes, and when R is not a 2D array.
    R1 = numpy.ones(3)
    M = numpy.ones((2,3))
    I,J,K,L = 5,3,1,2
    lambdaF = numpy.ones((I,K))
    lambdaS = numpy.ones((K,L))
    lambdaG = numpy.ones((J,L))
    alphatau, betatau = 3, 1    
    hyperparams = { 'alphatau':alphatau, 'betatau':betatau, 'lambdaF':lambdaF, 'lambdaS':lambdaS, 'lambdaG':lambdaG }
    
    with pytest.raises(AssertionError) as error:
        bnmtf_vb(R1,M,K,L,False,hyperparams)
    assert str(error.value) == "Input matrix R is not a two-dimensional array, but instead 1-dimensional."
    
    R2 = numpy.ones((4,3,2))
    with pytest.raises(AssertionError) as error:
        bnmtf_vb(R2,M,K,L,False,hyperparams)
    assert str(error.value) == "Input matrix R is not a two-dimensional array, but instead 3-dimensional."
    
    R3 = numpy.ones((3,2))
    with pytest.raises(AssertionError) as error:
        bnmtf_vb(R3,M,K,L,False,hyperparams)
    assert str(error.value) == "Input matrix R is not of the same size as the indicator matrix M: (3, 2) and (2, 3) respectively."
    
    # Similarly for lambdaF, lambdaS, lambdaG
    I,J,K,L = 2,3,1,2
    R4 = numpy.ones((2,3))
    lambdaF = numpy.ones((2+1,1))
    hyperparams = { 'alphatau':alphatau, 'betatau':betatau, 'lambdaF':lambdaF, 'lambdaS':lambdaS, 'lambdaG':lambdaG }
    with pytest.raises(AssertionError) as error:
        bnmtf_vb(R4,M,K,L,False,hyperparams)
    assert str(error.value) == "Prior matrix lambdaF has the wrong shape: (3, 1) instead of (2, 1)."
    
    lambdaF = numpy.ones((2,1))
    lambdaS = numpy.ones((1+1,2+1))
    hyperparams = { 'alphatau':alphatau, 'betatau':betatau, 'lambdaF':lambdaF, 'lambdaS':lambdaS, 'lambdaG':lambdaG }
    with pytest.raises(AssertionError) as error:
        bnmtf_vb(R4,M,K,L,False,hyperparams)
    assert str(error.value) == "Prior matrix lambdaS has the wrong shape: (2, 3) instead of (1, 2)."
    
    lambdaS = numpy.ones((1,2))
    lambdaG = numpy.ones((3,2+1))
    hyperparams = { 'alphatau':alphatau, 'betatau':betatau, 'lambdaF':lambdaF, 'lambdaS':lambdaS, 'lambdaG':lambdaG }
    with pytest.raises(AssertionError) as error:
        bnmtf_vb(R4,M,K,L,False,hyperparams)
    assert str(error.value) == "Prior matrix lambdaG has the wrong shape: (3, 3) instead of (3, 2)."
    
    # Test getting an exception if a row or column is entirely unknown
    lambdaF = numpy.ones((I,K))
    lambdaS = numpy.ones((K,L))
    lambdaG = numpy.ones((J,L))
    M1 = [[1,1,1],[0,0,0]]
    M2 = [[1,1,0],[1,0,0]]
    hyperparams = { 'alphatau':alphatau, 'betatau':betatau, 'lambdaF':lambdaF, 'lambdaS':lambdaS, 'lambdaG':lambdaG }
    
    with pytest.raises(AssertionError) as error:
        bnmtf_vb(R4,M1,K,L,False,hyperparams)
    assert str(error.value) == "Fully unobserved row in R, row 1."
    with pytest.raises(AssertionError) as error:
        bnmtf_vb(R4,M2,K,L,False,hyperparams)
    assert str(error.value) == "Fully unobserved column in R, column 2."
    
    # Finally, a successful case
    I,J,K,L = 3,2,2,2
    R5 = 2*numpy.ones((I,J))
    lambdaF = numpy.ones((I,K))
    lambdaS = numpy.ones((K,L))
    lambdaG = numpy.ones((J,L))
    M = numpy.ones((I,J))
    hyperparams = { 'alphatau':alphatau, 'betatau':betatau, 'lambdaF':lambdaF, 'lambdaS':lambdaS, 'lambdaG':lambdaG }
    BNMTF = bnmtf_vb(R5,M,K,L,False,hyperparams)
    
    assert numpy.array_equal(BNMTF.R,R5)
    assert numpy.array_equal(BNMTF.M,M)
    assert BNMTF.I == I
    assert BNMTF.J == J
    assert BNMTF.K == K
    assert BNMTF.L == L
    assert BNMTF.size_Omega == I*J
    assert BNMTF.alphatau == alphatau
    assert BNMTF.betatau == betatau
    assert numpy.array_equal(BNMTF.lambdaF,lambdaF)
    assert numpy.array_equal(BNMTF.lambdaS,lambdaS)
    assert numpy.array_equal(BNMTF.lambdaG,lambdaG)
    
    # Test when lambdaF S G are integers
    I,J,K,L = 3,2,2,2
    R5 = 2*numpy.ones((I,J))
    lambdaF = 3
    lambdaS = 4
    lambdaG = 5
    M = numpy.ones((I,J))
    hyperparams = { 'alphatau':alphatau, 'betatau':betatau, 'lambdaF':lambdaF, 'lambdaS':lambdaS, 'lambdaG':lambdaG }
    BNMTF = bnmtf_vb(R5,M,K,L,False,hyperparams)
    
    assert numpy.array_equal(BNMTF.R,R5)
    assert numpy.array_equal(BNMTF.M,M)
    assert BNMTF.I == I
    assert BNMTF.J == J
    assert BNMTF.K == K
    assert BNMTF.L == L
    assert BNMTF.size_Omega == I*J
    assert BNMTF.alphatau == alphatau
    assert BNMTF.betatau == betatau
    assert numpy.array_equal(BNMTF.lambdaF,3*numpy.ones((I,K)))
    assert numpy.array_equal(BNMTF.lambdaS,4*numpy.ones((K,L)))
    assert numpy.array_equal(BNMTF.lambdaG,5*numpy.ones((J,L)))
    
    # Finally, test case with ARD
    I,J,K,L = 3,2,2,2
    R5 = 2*numpy.ones((I,J))
    lambdaS = numpy.ones((K,L))
    alpha0, beta0 = 6, 2
    M = numpy.ones((I,J))
    hyperparams = { 'alphatau':alphatau, 'betatau':betatau, 'alpha0':alpha0, 'beta0':beta0, 'lambdaS':lambdaS }
    BNMTF = bnmtf_vb(R5,M,K,L,True,hyperparams)
    
    assert numpy.array_equal(BNMTF.R,R5)
    assert numpy.array_equal(BNMTF.M,M)
    assert BNMTF.I == I
    assert BNMTF.J == J
    assert BNMTF.K == K
    assert BNMTF.L == L
    assert BNMTF.size_Omega == I*J
    assert BNMTF.alphatau == alphatau
    assert BNMTF.betatau == betatau
    assert BNMTF.alpha0 == alpha0
    assert BNMTF.beta0 == beta0
    assert numpy.array_equal(BNMTF.lambdaS,lambdaS)
    
    
""" Test initialing parameters """
def test_initialise():
    I,J,K,L = 5,3,2,4
    R = numpy.ones((I,J))
    M = numpy.ones((I,J))
    
    lambdaF = 2*numpy.ones((I,K))
    lambdaS = 3*numpy.ones((K,L))
    lambdaG = 4*numpy.ones((J,L))
    alphatau, betatau = 3, 1
    alpha0, beta0 = 6, 2
    hyperparams = { 'alphatau':alphatau, 'betatau':betatau, 'alpha0':alpha0, 'beta0':beta0, 'lambdaF':lambdaF, 'lambdaS':lambdaS, 'lambdaG':lambdaG }
    
    # Initialisation using expectation
    init_S, init_FG = 'exp', 'exp'
    BNMTF = bnmtf_vb(R,M,K,L,False,hyperparams)
    BNMTF.initialise(init_FG,init_S)
    
    for i,k in itertools.product(xrange(0,I),xrange(0,K)):
        assert BNMTF.tau_F[i,k] == 1.
        assert BNMTF.mu_F[i,k] == 1./lambdaF[i,k]
    for k,l in itertools.product(xrange(0,K),xrange(0,L)):
        assert BNMTF.tau_S[k,l] == 1.
        assert BNMTF.mu_S[k,l] == 1./lambdaS[k,l]
    for j,l in itertools.product(xrange(0,J),xrange(0,L)):
        assert BNMTF.tau_G[j,l] == 1.
        assert BNMTF.mu_G[j,l] == 1./lambdaG[j,l]
    assert BNMTF.alpha_s == alphatau + I*J/2.
    assert BNMTF.beta_s == betatau + BNMTF.exp_square_diff()/2.
        
    assert BNMTF.exp_tau == (alphatau + I*J/2.)/(betatau + BNMTF.exp_square_diff()/2.)
    
    for i,k in itertools.product(xrange(0,I),xrange(0,K)):
        assert abs(BNMTF.exp_F[i,k] - (0.5 + 0.352065 / (1-0.3085))) < 0.0001
    for k,l in itertools.product(xrange(0,K),xrange(0,L)):
        assert abs(BNMTF.exp_S[k,l] - (1./3. + 0.377383 / (1-0.3694))) < 0.0001
    for j,l in itertools.product(xrange(0,J),xrange(0,L)):
        assert abs(BNMTF.exp_G[j,l] - (1./4. + 0.386668 / (1-0.4013))) < 0.0001
    
    # Initialisation of S using random draws from prior
    init_S, init_FG = 'random', 'exp'
    BNMTF = bnmtf_vb(R,M,K,L,False,hyperparams)
    BNMTF.initialise(init_FG,init_S)
    
    for i,k in itertools.product(xrange(0,I),xrange(0,K)):
        assert BNMTF.tau_F[i,k] == 1.
        assert BNMTF.mu_F[i,k] == 1./lambdaF[i,k]
    for k,l in itertools.product(xrange(0,K),xrange(0,L)):
        assert BNMTF.tau_S[k,l] == 1.
        assert BNMTF.mu_S[k,l] != 1./lambdaS[k,l] # test whether we overwrote the expectation
    for j,l in itertools.product(xrange(0,J),xrange(0,L)):
        assert BNMTF.tau_G[j,l] == 1.
        assert BNMTF.mu_G[j,l] == 1./lambdaG[j,l]
    
    # Initialisation of F and G using random draws from prior
    init_S, init_FG = 'exp', 'random'
    BNMTF = bnmtf_vb(R,M,K,L,False,hyperparams)
    BNMTF.initialise(init_FG,init_S)
    
    for i,k in itertools.product(xrange(0,I),xrange(0,K)):
        assert BNMTF.tau_F[i,k] == 1.
        assert BNMTF.mu_F[i,k] != 1./lambdaF[i,k] # test whether we overwrote the expectation
    for k,l in itertools.product(xrange(0,K),xrange(0,L)):
        assert BNMTF.tau_S[k,l] == 1.
        assert BNMTF.mu_S[k,l] == 1./lambdaS[k,l]
    for j,l in itertools.product(xrange(0,J),xrange(0,L)):
        assert BNMTF.tau_G[j,l] == 1.
        assert BNMTF.mu_G[j,l] != 1./lambdaG[j,l]
        
    # Initialisation of F and G using Kmeans
    init_S, init_FG = 'exp', 'kmeans'
    BNMTF = bnmtf_vb(R,M,K,L,False,hyperparams)
    BNMTF.initialise(init_FG,init_S)
    
    for i,k in itertools.product(xrange(0,I),xrange(0,K)):
        assert BNMTF.tau_F[i,k] == 1.
        assert BNMTF.mu_F[i,k] == 0 or BNMTF.mu_F[i,k] == 1 
    for j,l in itertools.product(xrange(0,J),xrange(0,L)):
        assert BNMTF.tau_G[j,l] == 1.
        assert BNMTF.mu_G[j,l] == 0 or BNMTF.mu_G[j,l] == 1 
    for k,l in itertools.product(xrange(0,K),xrange(0,L)):
        assert BNMTF.tau_S[k,l] == 1.
        assert BNMTF.mu_S[k,l] == 1./lambdaS[k,l]
        
    # Initialise with ARD
    init_S, init_FG = 'exp', 'exp'
    BNMTF = bnmtf_vb(R,M,K,L,True,hyperparams)
    BNMTF.initialise(init_FG,init_S)
    
    for k in range(K):
        assert BNMTF.alphaFk_s[k] == alpha0
        assert BNMTF.betaFk_s[k] == beta0
        assert BNMTF.exp_lambdaFk[k] == alpha0 / float(beta0)
    for l in range(L):
        assert BNMTF.alphaGl_s[l] == alpha0
        assert BNMTF.betaGl_s[l] == beta0
        assert BNMTF.exp_lambdaGl[l] == alpha0 / float(beta0)
    
    for i,k in itertools.product(range(I),range(K)):
        assert BNMTF.tau_F[i,k] == 1.
        assert BNMTF.mu_F[i,k] == 1./(alpha0 / float(beta0))
    for k,l in itertools.product(xrange(0,K),xrange(0,L)):
        assert BNMTF.tau_S[k,l] == 1.
        assert BNMTF.mu_S[k,l] == 1./lambdaS[k,l]
    for j,l in itertools.product(xrange(0,J),xrange(0,L)):
        assert BNMTF.tau_G[j,l] == 1.
        assert BNMTF.mu_G[j,l] == 1./(alpha0 / float(beta0))
        
    assert BNMTF.alpha_s == alphatau + I*J/2.
    assert BNMTF.beta_s == betatau + BNMTF.exp_square_diff()/2.    
    assert BNMTF.exp_tau == (alphatau + I*J/2.)/(betatau + BNMTF.exp_square_diff()/2.)
    
        
""" Test computing the ELBO. """
def test_elbo():
    I,J,K,L = 5,3,2,4
    R = numpy.ones((I,J))
    M = numpy.ones((I,J))
    M[0,0], M[2,2], M[3,1] = 0, 0, 0 # size Omega = 12
    
    lambdaF = 2*numpy.ones((I,K))
    lambdaS = 3*numpy.ones((K,L))
    lambdaG = 4*numpy.ones((J,L))
    alphatau, betatau = 3, 1
    hyperparams = { 'alphatau':alphatau, 'betatau':betatau, 'lambdaF':lambdaF, 'lambdaS':lambdaS, 'lambdaG':lambdaG }
  
    expF = 5*numpy.ones((I,K))
    expS = 6*numpy.ones((K,L))
    expG = 7*numpy.ones((J,L))
    varF = 11*numpy.ones((I,K))
    varS = 12*numpy.ones((K,L))
    varG = 13*numpy.ones((J,L))
    exptau = 8.
    explogtau = 9.
    
    muF = 14*numpy.ones((I,K))
    muS = 15*numpy.ones((K,L))
    muG = 16*numpy.ones((J,L))
    tauF = numpy.ones((I,K))/100.
    tauS = numpy.ones((K,L))/101.
    tauG = numpy.ones((J,L))/102.
    alpha_s = 20.
    beta_s = 21.
    
    # expF * expS * expG.T = [[1680]]
    # (R - expF*expS*expG.T)^2 = 12*1679^2 = 33828492
    # Var[F*S*G.T] = 12*K*L*((11+5^2)*(12+6^2)*(13+7^2)-5^2*6^2*7^2
    #                        + 11*6*7*((4-1)*6*7) + 13*5*6*((2-1)*5*6))
    #              = 12*2*4*(63036 + 58212 + 11700) = 12763008
    
    # -muF*sqrt(tauF) = -14*math.sqrt(1./100.) = -1.4
    # -muS*sqrt(tauS) = -15*math.sqrt(1./101.) = -1.4925557853149838
    # -muG*sqrt(tauG) = -16*math.sqrt(1./102.) = -1.5842360687626789
    # cdf(-1.4) = 0.080756659233771066
    # cdf(-1.4925557853149838) = 0.067776752211548219
    # cdf(-1.5842360687626789) = 0.056570004076003155
    
    ELBO = 12./2.*(explogtau - math.log(2*math.pi)) - 8./2.*(33828492+12763008) \
         + 5*2*(math.log(2.) - 2.*5.) + 2*4*(math.log(3.) - 3.*6.) + 3*4*(math.log(4.) - 4.*7.) \
         + 3.*numpy.log(1.) - numpy.log(math.gamma(3.)) + 2.*9. - 1.*8. \
         - 20.*numpy.log(21.) + numpy.log(math.gamma(20.)) - 19.*9. + 21.*8. \
         - 0.5*5*2*math.log(1./100.) + 0.5*5*2*math.log(2*math.pi) + 5*2*math.log(1.-0.080756659233771066) \
         + 0.5*5*2*1./100.*(11.+81.) \
         - 0.5*4*2*math.log(1./101.) + 0.5*4*2*math.log(2*math.pi) + 4*2*math.log(1.-0.067776752211548219) \
         + 0.5*4*2*1./101.*(12.+81.) \
         - 0.5*4*3*math.log(1./102.) + 0.5*4*3*math.log(2*math.pi) + 4*3*math.log(1.-0.056570004076003155) \
         + 0.5*4*3*1./102.*(13.+81.)
         
    BNMTF = bnmtf_vb(R,M,K,L,False,hyperparams)
    BNMTF.exp_F = expF
    BNMTF.exp_S = expS
    BNMTF.exp_G = expG
    BNMTF.var_F = varF
    BNMTF.var_S = varS
    BNMTF.var_G = varG
    BNMTF.exp_tau = exptau
    BNMTF.exp_logtau = explogtau
    BNMTF.mu_F = muF
    BNMTF.mu_S = muS
    BNMTF.mu_G = muG
    BNMTF.tau_F = tauF
    BNMTF.tau_S = tauS
    BNMTF.tau_G = tauG
    BNMTF.alpha_s = alpha_s
    BNMTF.beta_s = beta_s
    assert abs(BNMTF.elbo() - ELBO) < 0.000001
    
        
""" Test updating parameters U, V, tau """          
I,J,K,L = 5,3,2,4
R = numpy.ones((I,J))
M = numpy.ones((I,J))
M[0,0], M[2,2], M[3,1] = 0, 0, 0 # size Omega = 12

lambdaF = 2*numpy.ones((I,K))
lambdaS = 3*numpy.ones((K,L))
lambdaG = 4*numpy.ones((J,L))
alphatau, betatau = 3, 1
alpha0, beta0 = 6, 2
hyperparams = { 'alphatau':alphatau, 'betatau':betatau, 'alpha0':alpha0, 'beta0':beta0, 'lambdaF':lambdaF, 'lambdaS':lambdaS, 'lambdaG':lambdaG }
    
init_S, init_FG = 'exp', 'exp'

def test_exp_square_diff():
    BNMTF = bnmtf_vb(R,M,K,L,False,hyperparams)
    BNMTF.exp_F = 1./lambdaF #[[1./2.]]
    BNMTF.exp_S = 1./lambdaS #[[1./3.]]
    BNMTF.exp_G = 1./lambdaG #[[1./4.]]
    BNMTF.var_F = numpy.ones((I,K))*2 #[[2.]]
    BNMTF.var_S = numpy.ones((K,L))*3 #[[3.]]
    BNMTF.var_G = numpy.ones((J,L))*4 #[[4.]]
    # expF * expS * expV.T = [[1./3.]]. (varF+expF^2)=2.25, (varS+expS^2)=3.+1./9., (varG+expG^2)=4.0625
    # 12.*(4./9.) + 12.*(2*4*(2.25*(3.+1./9.)*4.0625-1./4.*1./9.*1./16. + 2./3./4.*((4-1)/3./4.) +4./2./3.*((2-1)/2./3.) ))
    exp_square_diff = 2749+5./6. # 
    assert abs(BNMTF.exp_square_diff() - exp_square_diff) < 0.000000000001

def test_update_tau():
    BNMTF = bnmtf_vb(R,M,K,L,False,hyperparams)
    BNMTF.exp_F = 1./lambdaF #[[1./2.]]
    BNMTF.exp_S = 1./lambdaS #[[1./3.]]
    BNMTF.exp_G = 1./lambdaG #[[1./4.]]
    BNMTF.var_F = numpy.ones((I,K))*2 #[[2.]]
    BNMTF.var_S = numpy.ones((K,L))*3 #[[3.]]
    BNMTF.var_G = numpy.ones((J,L))*4 #[[4.]]
    BNMTF.update_tau()
    assert BNMTF.alpha_s == alphatau + 12./2.
    assert abs(BNMTF.beta_s - (betatau + (2749+5./6.)/2.)) < 0.000000000001
    
def test_update_lambdaFk():
    BNMTF = bnmtf_vb(R,M,K,L,True,hyperparams)
    BNMTF.alphaFk_s, BNMTF.betaFk_s = numpy.zeros(K), numpy.zeros(K)
    BNMTF.exp_F = 1./lambdaF #[[1./2.]]
    for k in range(K):
        BNMTF.update_lambdaFk(k)
        assert BNMTF.alphaFk_s[k] == alpha0 + I
        assert BNMTF.betaFk_s[k] == (beta0 + I*1./2.)
    
def test_update_lambdaGl():
    BNMTF = bnmtf_vb(R,M,K,L,True,hyperparams)
    BNMTF.alphaGl_s, BNMTF.betaGl_s = numpy.zeros(L), numpy.zeros(L)
    BNMTF.exp_G = 1./lambdaG #[[1./4.]]
    for l in range(L):
        BNMTF.update_lambdaGl(l)
        assert BNMTF.alphaGl_s[l] == alpha0 + J
        assert BNMTF.betaGl_s[l] == (beta0 + J*1./4.)
    
def test_update_F():
    for k in range(K):
        BNMTF = bnmtf_vb(R,M,K,L,False,hyperparams)
        BNMTF.mu_F = numpy.zeros((I,K))
        BNMTF.tau_F = numpy.zeros((I,K))
        BNMTF.exp_F = 1./lambdaF #[[1./2.]]
        BNMTF.exp_S = 1./lambdaS #[[1./3.]]
        BNMTF.exp_G = 1./lambdaG #[[1./4.]]
        BNMTF.var_F = numpy.ones((I,K))*2 #[[2.]]
        BNMTF.var_S = numpy.ones((K,L))*3 #[[3.]]
        BNMTF.var_G = numpy.ones((J,L))*4 #[[4.]]
        BNMTF.exp_tau = 3.
        BNMTF.update_F(k)
        
        for i in range(0,I):
            tauFik = 3. * sum([
                sum([BNMTF.exp_S[k,l]*BNMTF.exp_G[j,l] for l in range(L)])**2 \
                + sum([(BNMTF.var_S[k,l]+BNMTF.exp_S[k,l]**2)*(BNMTF.var_G[j,l]+BNMTF.exp_G[j,l]**2) - BNMTF.exp_S[k,l]**2*BNMTF.exp_G[j,l]**2 for l in range(L)])
            for j in range(0,J) if M[i,j]])
            muFik = 1./tauFik * (-lambdaF[i,k] + BNMTF.exp_tau * sum([
                sum([BNMTF.exp_S[k,l]*BNMTF.exp_G[j,l] for l in range(L)]) * \
                (R[i,j] - sum([BNMTF.exp_F[i,kp]*BNMTF.exp_S[kp,l]*BNMTF.exp_G[j,l] for kp,l in itertools.product(range(K),range(L)) if kp != k]))
                - sum([
                    BNMTF.exp_S[k,l] * BNMTF.var_G[j,l] * sum([BNMTF.exp_F[i,kp] * BNMTF.exp_S[kp,l] for kp in range(K) if kp != k])
                for l in range(L)])
            for j in range(J) if M[i,j]]))
                
            assert BNMTF.tau_F[i,k] == tauFik
            assert abs(BNMTF.mu_F[i,k] - muFik) < 0.00000000000000001
    
def test_update_S():
    for k,l in itertools.product(range(K),range(L)):
        BNMTF = bnmtf_vb(R,M,K,L,False,hyperparams)
        BNMTF.mu_S = numpy.zeros((K,L))
        BNMTF.tau_S = numpy.zeros((K,L))
        BNMTF.exp_F = 1./lambdaF #[[1./2.]]
        BNMTF.exp_S = 1./lambdaS #[[1./3.]]
        BNMTF.exp_G = 1./lambdaG #[[1./4.]]
        BNMTF.var_F = numpy.ones((I,K))*2 #[[2.]]
        BNMTF.var_S = numpy.ones((K,L))*3 #[[3.]]
        BNMTF.var_G = numpy.ones((J,L))*4 #[[4.]]
        BNMTF.exp_tau = 3.
        BNMTF.update_S(k,l)
        
        tauSkl = 3. * sum([
            BNMTF.exp_F[i,k]**2 * BNMTF.exp_G[j,l]**2 \
            + (BNMTF.var_F[i,k]+BNMTF.exp_F[i,k]**2)*(BNMTF.var_G[j,l]+BNMTF.exp_G[j,l]**2) - BNMTF.exp_F[i,k]**2*BNMTF.exp_G[j,l]**2
        for i,j in itertools.product(range(I),range(J)) if M[i,j]])
        muSkl = 1./tauSkl * (-lambdaS[k,l] + BNMTF.exp_tau * sum([
            BNMTF.exp_F[i,k]*BNMTF.exp_G[j,l]*(R[i,j] - sum([BNMTF.exp_F[i,kp]*BNMTF.exp_S[kp,lp]*BNMTF.exp_G[j,lp] for kp,lp in itertools.product(range(K),range(L)) if (kp != k or lp != l)]))
            - BNMTF.var_F[i,k] * BNMTF.exp_G[j,l] * sum([BNMTF.exp_S[k,lp] * BNMTF.exp_G[j,lp] for lp in range(L) if lp != l])
            - BNMTF.exp_F[i,k] * BNMTF.var_G[j,l] * sum([BNMTF.exp_F[i,kp] * BNMTF.exp_S[kp,l] for kp in range(K) if kp != k])
        for i,j in itertools.product(range(I),range(J)) if M[i,j]]))
        
        assert BNMTF.tau_S[k,l] == tauSkl
        assert abs(BNMTF.mu_S[k,l] - muSkl) < 0.0000000000000001
    
def test_update_G():
    for l in range(0,L):
        BNMTF = bnmtf_vb(R,M,K,L,False,hyperparams)
        BNMTF.mu_G = numpy.zeros((J,L))
        BNMTF.tau_G = numpy.zeros((J,L))
        BNMTF.exp_F = 1./lambdaF #[[1./2.]]
        BNMTF.exp_S = 1./lambdaS #[[1./3.]]
        BNMTF.exp_G = 1./lambdaG #[[1./4.]]
        BNMTF.var_F = numpy.ones((I,K))*2 #[[2.]]
        BNMTF.var_S = numpy.ones((K,L))*3 #[[3.]]
        BNMTF.var_G = numpy.ones((J,L))*4 #[[4.]]
        BNMTF.exp_tau = 3.
        BNMTF.update_G(l)
        
        for j in range(J):
            tauGjl = 3. * sum([
                sum([BNMTF.exp_F[i,k]*BNMTF.exp_S[k,l] for k in range(K)])**2 \
                + sum([(BNMTF.var_S[k,l]+BNMTF.exp_S[k,l]**2)*(BNMTF.var_F[i,k]+BNMTF.exp_F[i,k]**2) - BNMTF.exp_S[k,l]**2*BNMTF.exp_F[i,k]**2 for k in range(K)])
            for i in range(I) if M[i,j]])
            muGjl = 1./tauGjl * (-lambdaG[j,l] + BNMTF.exp_tau * sum([
                sum([BNMTF.exp_F[i,k]*BNMTF.exp_S[k,l] for k in range(K)]) * \
                (R[i,j] - sum([BNMTF.exp_F[i,k]*BNMTF.exp_S[k,lp]*BNMTF.exp_G[j,lp] for k,lp in itertools.product(range(K),range(L)) if lp != l]))
                - sum([
                    BNMTF.var_F[i,k] * BNMTF.exp_S[k,l] * sum([BNMTF.exp_S[k,lp] * BNMTF.exp_G[j,lp] for lp in range(L) if lp != l])
                for k in range(K)])
            for i in range(I) if M[i,j]]))
            
            assert BNMTF.tau_G[j,l] == tauGjl
            assert BNMTF.mu_G[j,l] == muGjl


""" Test computing expectation, variance F, S, G, tau """  
def test_update_exp_tau():
    BNMTF = bnmtf_vb(R,M,K,L,False,hyperparams)
    BNMTF.initialise()  
    BNMTF.exp_F = 1./lambdaF #[[1./2.]]
    BNMTF.exp_S = 1./lambdaS #[[1./3.]]
    BNMTF.exp_G = 1./lambdaG #[[1./4.]]
    BNMTF.var_F = numpy.ones((I,K))*2 #[[2.]]
    BNMTF.var_S = numpy.ones((K,L))*3 #[[3.]]
    BNMTF.var_G = numpy.ones((J,L))*4 #[[4.]]
    BNMTF.update_tau()
    BNMTF.update_exp_tau()
    
    assert abs(BNMTF.exp_tau - 9./1375.91666667) < 0.0000000000001
    assert abs(BNMTF.exp_logtau - (2.14064147795560999 - math.log(1375.91666667))) < 0.00000000001   
   
def test_update_exp_lambdaFk():
    BNMTF = bnmtf_vb(R,M,K,L,True,hyperparams)
    BNMTF.initialise(init_FG='exp',init_S='exp')  
    for k in range(K):
        assert abs(BNMTF.exp_lambdaFk[k] - (alpha0+I)/(beta0+I*beta0/float(alpha0))) < 0.000000000001
        assert BNMTF.exp_loglambdaFk[k] == 1.0129704878718551
        
def test_update_exp_lambdaGl():
    BNMTF = bnmtf_vb(R,M,K,L,True,hyperparams)
    BNMTF.initialise(init_FG='exp',init_S='exp')  
    for l in range(L):
        assert abs(BNMTF.exp_lambdaGl[l] - (alpha0+J)/(beta0+J*beta0/float(alpha0))) < 0.000000000001
        assert BNMTF.exp_loglambdaGl[l] == 1.0129704878718551
        
def test_update_exp_F():
    for k in range(K):
        BNMTF = bnmtf_vb(R,M,K,L,False,hyperparams)
        BNMTF.initialise(init_FG,init_S)
        BNMTF.tau_F = 4*numpy.ones((I,K))  # muF = [[0.5]], tauF = [[4.]]
        BNMTF.update_exp_F(k) #-mu*sqrt(tau) = -0.5*2 = -1. lambda(1) = 0.241971 / (1-0.1587) = 0.2876155949126352. gamma = 0.37033832534958433
        for i in range(I):        
            assert abs(BNMTF.exp_F[i,k] - (0.5 + 1./2. * 0.2876155949126352)) < 0.00001
            assert abs(BNMTF.var_F[i,k] - 1./4.*(1.-0.37033832534958433)) < 0.00001

def test_update_exp_S():
    for k,l in itertools.product(range(K),range(L)):
        BNMTF = bnmtf_vb(R,M,K,L,False,hyperparams)
        BNMTF.initialise(init_FG,init_S) 
        BNMTF.tau_S = 4*numpy.ones((K,L)) # muS = [[1./3.]], tauS = [[4.]]
        BNMTF.update_exp_S(k,l) #-mu*sqrt(tau) = -2./3., lambda(..) = 0.319448 / (1-0.2525) = 0.4273551839464883, gamma = 
        assert abs(BNMTF.exp_S[k,l] - (1./3. + 1./2. * 0.4273551839464883)) < 0.00001
        assert abs(BNMTF.var_S[k,l] - 1./4.*(1. - 0.4675359092102624)) < 0.00001

def test_update_exp_G():
    for l in range(L):
        BNMTF = bnmtf_vb(R,M,K,L,False,hyperparams)
        BNMTF.initialise(init_FG,init_S) 
        BNMTF.tau_G = 4*numpy.ones((J,L)) # muG = [[1./4.]], tauG = [[4.]]
        BNMTF.update_exp_G(l) #-mu*sqrt(tau) = -0.5., lambda(..) = 0.352065 / (1-0.3085) = 0.5091323210412148, gamma = 0.5137818808494219
        for j in range(0,J):        
            assert abs(BNMTF.exp_G[j,l] - (1./4. + 1./2. * 0.5091323210412148)) < 0.0001
            assert abs(BNMTF.var_G[j,l] - 1./4.*(1. - 0.5137818808494219)) < 0.0001
    

""" Test two iterations of run(), and that all values have changed. """
def test_run():
    I,J,K,L = 10,5,3,2
    R = numpy.ones((I,J))
    M = numpy.ones((I,J))
    M[0,0], M[2,2], M[3,1] = 0, 0, 0
    R[0,1], R[0,2] = 2., 3.
    
    lambdaS = 3*numpy.ones((K,L))
    alphatau, betatau = 3, 1
    alpha0, beta0 = 6, 2
    hyperparams = { 'alphatau':alphatau, 'betatau':betatau, 'alpha0':alpha0, 'beta0':beta0, 'lambdaS': lambdaS }
    
    iterations = 15
    
    BNMTF = bnmtf_vb(R,M,K,L,True,hyperparams)
    BNMTF.initialise(init_FG='exp', init_S='exp')
    BNMTF.run(iterations)
    
    for k in range(K):
        assert BNMTF.alphaFk_s[k] != alpha0
        assert BNMTF.betaFk_s[k] != beta0
        assert BNMTF.exp_lambdaFk[k] != alpha0/float(beta0)
    for l in range(L):
        assert BNMTF.alphaGl_s[l] != alpha0
        assert BNMTF.betaGl_s[l] != beta0
        assert BNMTF.exp_lambdaGl[l] != alpha0/float(beta0)
    for i,k in itertools.product(range(I),range(K)):
        assert BNMTF.mu_F[i,k] != 1./(alpha0 / float(beta0))
        assert BNMTF.tau_F[i,k] != 1.
        assert BNMTF.exp_F[i,k] != numpy.inf and not math.isnan(BNMTF.exp_F[i,k])
        assert BNMTF.tau_F[i,k] != numpy.inf and not math.isnan(BNMTF.tau_F[i,k])
    for k,l in itertools.product(range(K),range(L)):
        assert BNMTF.mu_S[k,l] != 1./lambdaS[k,l]
        assert BNMTF.tau_S[k,l] != 1.
        assert BNMTF.exp_S[k,l] != numpy.inf and not math.isnan(BNMTF.exp_S[k,l])
        assert BNMTF.tau_S[k,l] != numpy.inf and not math.isnan(BNMTF.tau_S[k,l])
    for j,l in itertools.product(range(J),range(L)):
        assert BNMTF.mu_G[j,l] != 1./(alpha0 / float(beta0))
        assert BNMTF.tau_G[j,l] != 1.
        assert BNMTF.exp_G[j,l] != numpy.inf and not math.isnan(BNMTF.exp_G[j,l])
        assert BNMTF.tau_G[j,l] != numpy.inf and not math.isnan(BNMTF.tau_G[j,l])
    assert BNMTF.exp_tau != alphatau/float(betatau)

""" Test computing the performance of the predictions using the expectations """
def test_predict():
    (I,J,K) = (5,3,2)
    R = numpy.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15]],dtype=float)
    M = numpy.ones((I,J))
    K = 3
    
    lambdaF = 2*numpy.ones((I,K))
    lambdaS = 3*numpy.ones((K,L))
    lambdaG = 5*numpy.ones((J,L))
    alphatau, betatau = 3, 1
    hyperparams = { 'alphatau':alphatau, 'betatau':betatau, 'lambdaF':lambdaF, 'lambdaS':lambdaS, 'lambdaG':lambdaG }
    
    expF = numpy.array([[125.,126.],[126.,126.],[126.,126.],[126.,126.],[126.,126.]])
    expS = numpy.array([[84.,84.,84.,84.],[84.,84.,84.,84.]])
    expG = numpy.array([[42.,42.,42.,42.],[42.,42.,42.,42.],[42.,42.,42.,42.]])
    
    M_test = numpy.array([[0,0,1],[0,1,0],[0,0,0],[1,1,0],[0,0,0]]) #R->3,5,10,11, R_pred->3542112,3556224,3556224,3556224
    MSE = ((3.-3542112.)**2 + (5.-3556224.)**2 + (10.-3556224.)**2 + (11.-3556224.)**2) / 4.
    R2 = 1. - ((3.-3542112.)**2 + (5.-3556224.)**2 + (10.-3556224.)**2 + (11.-3556224.)**2) / (4.25**2+2.25**2+2.75**2+3.75**2) #mean=7.25
    Rp = 357. / ( math.sqrt(44.75) * math.sqrt(5292.) ) #mean=7.25,var=44.75, mean_pred=3552696,var_pred=5292, corr=(-4.25*-63 + -2.25*21 + 2.75*21 + 3.75*21)
    
    BNMTF = bnmtf_vb(R,M,K,L,False,hyperparams)
    BNMTF.exp_F = expF
    BNMTF.exp_S = expS
    BNMTF.exp_G = expG
    performances = BNMTF.predict(M_test)
    
    assert performances['MSE'] == MSE
    assert performances['R^2'] == R2
    assert performances['Rp'] == Rp
       
        
""" Test the evaluation measures MSE, R^2, Rp """
def test_compute_statistics():
    R = numpy.array([[1,2],[3,4]],dtype=float)
    M = numpy.array([[1,1],[0,1]])
    I, J, K, L = 2, 2, 3, 4
    lambdaF = 2*numpy.ones((I,K))
    lambdaS = 3*numpy.ones((K,L))
    lambdaG = 4*numpy.ones((J,L))
    alphatau, betatau = 3, 1
    hyperparams = { 'alphatau':alphatau, 'betatau':betatau, 'lambdaF':lambdaF, 'lambdaS':lambdaS, 'lambdaG':lambdaG }
    
    BNMTF = bnmtf_vb(R,M,K,L,False,hyperparams)
    
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
    lambdaF = 2*numpy.ones((I,K))
    lambdaS = 3*numpy.ones((K,L))
    lambdaG = 4*numpy.ones((J,L))
    alphatau, betatau = 3, 1
    hyperparams = { 'alphatau':alphatau, 'betatau':betatau, 'lambdaF':lambdaF, 'lambdaS':lambdaS, 'lambdaG':lambdaG }
    
    BNMTF = bnmtf_vb(R,M,K,L,False,hyperparams)
    BNMTF.exp_F = numpy.ones((I,K))
    BNMTF.exp_S = 2*numpy.ones((K,L))
    BNMTF.exp_G = 3*numpy.ones((J,L))
    BNMTF.exp_logtau = 5.
    BNMTF.exp_tau = 3.
    # expU*expV.T = [[72.]]
    
    log_likelihood = 3./2.*(5.-math.log(2*math.pi)) - 3./2. * (71**2 + 70**2 + 68**2)
    AIC = -2*log_likelihood +2*(2*3+3*4+2*4+1)
    BIC = -2*log_likelihood +(2*3+3*4+2*4+1)*math.log(3)
    MSE = (71**2+70**2+68**2)/3.
    
    assert log_likelihood == BNMTF.quality('loglikelihood')
    assert AIC == BNMTF.quality('AIC')
    assert BIC == BNMTF.quality('BIC')
    assert MSE == BNMTF.quality('MSE')
    with pytest.raises(AssertionError) as error:
        BNMTF.quality('FAIL')
    assert str(error.value) == "Unrecognised metric for model quality: FAIL."