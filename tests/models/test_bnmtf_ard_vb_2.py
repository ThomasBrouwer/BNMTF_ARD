"""
Tests for the BNMTF+ARD Variational Bayes algorithm.
Model 2.
"""

import sys
sys.path.append("/home/tab43/Documents/Projects/libraries/")
from BNMTF_ARD.code.models.bnmtf_ard_vb_2 import bnmtf_ard_vb_2

import numpy, math, pytest, itertools, scipy

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
        bnmtf_ard_vb_2(R1,M,K,L,priors)
    assert str(error.value) == "Input matrix R is not a two-dimensional array, but instead 1-dimensional."
    
    R2 = numpy.ones((4,3,2))
    with pytest.raises(AssertionError) as error:
        bnmtf_ard_vb_2(R2,M,K,L,priors)
    assert str(error.value) == "Input matrix R is not a two-dimensional array, but instead 3-dimensional."
    
    R3 = numpy.ones((3,2))
    with pytest.raises(AssertionError) as error:
        bnmtf_ard_vb_2(R3,M,K,L,priors)
    assert str(error.value) == "Input matrix R is not of the same size as the indicator matrix M: (3, 2) and (2, 3) respectively."
    
    # Test getting an exception if a row or column is entirely unknown
    R4 = numpy.ones((2,3))
    M1 = [[1,1,1],[0,0,0]]
    M2 = [[1,1,0],[1,0,0]]
    
    with pytest.raises(AssertionError) as error:
        bnmtf_ard_vb_2(R4,M1,K,L,priors)
    assert str(error.value) == "Fully unobserved row in R, row 1."
    with pytest.raises(AssertionError) as error:
        bnmtf_ard_vb_2(R4,M2,K,L,priors)
    assert str(error.value) == "Fully unobserved column in R, column 2."
    
    # Finally, a successful case
    I,J,K,L = 3,2,2,2
    R5 = 2*numpy.ones((I,J))
    M = numpy.ones((I,J))
    BNMTF = bnmtf_ard_vb_2(R5,M,K,L,priors)
    
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
    
    # First do expectation
    init = 'exp'
    BNMTF = bnmtf_ard_vb_2(R,M,K,L,priors)
    BNMTF.initialise(init)
    
    for k in xrange(0,K):
        assert BNMTF.alphaF[k] == alpha0 + I
        assert BNMTF.betaF[k] == beta0 + I  
        assert BNMTF.exp_lambdaF[k] == (alpha0 + I) / (beta0 + I)
        assert BNMTF.exp_loglambdaF[k] == scipy.special.psi(alpha0 + I) - math.log(beta0 + I)
    for l in xrange(0,L):
        assert BNMTF.alphaG[l] == alpha0 + J
        assert BNMTF.betaG[l] == beta0 + J  
        assert BNMTF.exp_lambdaG[l] == (alpha0 + J) / (beta0 + J)
        assert BNMTF.exp_loglambdaG[l] == scipy.special.psi(alpha0 + J) - math.log(beta0 + J)
        
    for i,k in itertools.product(xrange(0,I),xrange(0,K)):
        assert BNMTF.muF[i,k] == 1. / BNMTF.exp_lambdaF[k]
        assert BNMTF.tauF[i,k] == 1.
        mu, tau, x = BNMTF.muF[i,k], BNMTF.tauF[i,k], -BNMTF.muF[i,k]*math.sqrt(BNMTF.tauF[i,k])
        assert BNMTF.exp_F[i,k] == mu + 1./math.sqrt(tau) * scipy.stats.norm.pdf(x)/(0.5*scipy.special.erfc(x/math.sqrt(2)))
    for j,l in itertools.product(xrange(0,J),xrange(0,L)):
        assert BNMTF.muG[j,l] == 1. / BNMTF.exp_lambdaG[l]
        assert BNMTF.tauG[j,l] == 1.
        mu, tau, x = BNMTF.muG[j,l], BNMTF.tauG[j,l], -BNMTF.muG[j,l]*math.sqrt(BNMTF.tauG[j,l])
        assert BNMTF.exp_G[j,l] == mu + 1./math.sqrt(tau) * scipy.stats.norm.pdf(x)/(0.5*scipy.special.erfc(x/math.sqrt(2)))
    for k,l in itertools.product(xrange(0,K),xrange(0,L)):
        assert BNMTF.muS[k,l] == 1. / (BNMTF.exp_lambdaF[k] * BNMTF.exp_lambdaG[l])
        assert BNMTF.tauS[k,l] == 1.
        mu, tau, x = BNMTF.muS[k,l], BNMTF.tauS[k,l], -BNMTF.muS[k,l]*math.sqrt(BNMTF.tauS[k,l])
        assert BNMTF.exp_S[k,l] == mu + 1./math.sqrt(tau) * scipy.stats.norm.pdf(x)/(0.5*scipy.special.erfc(x/math.sqrt(2)))
        
    assert BNMTF.alphaR_s == alphaR + I*J/2.
    assert BNMTF.betaR_s == 1109.2238084144074
    
    # Then random initialisation - check whether values have changed
    init = 'random'
    BNMTF = bnmtf_ard_vb_2(R,M,K,L,priors)
    BNMTF.initialise(init)
    
    for i,k in itertools.product(xrange(0,I),xrange(0,K)):
        assert BNMTF.muF[i,k] != 1. / BNMTF.exp_lambdaF[k]
        assert BNMTF.tauF[i,k] == 1.
        mu, tau, x = BNMTF.muF[i,k], BNMTF.tauF[i,k], -BNMTF.muF[i,k]*math.sqrt(BNMTF.tauF[i,k])
        assert BNMTF.exp_F[i,k] == mu + 1./math.sqrt(tau) * scipy.stats.norm.pdf(x)/(0.5*scipy.special.erfc(x/math.sqrt(2)))
    for j,l in itertools.product(xrange(0,J),xrange(0,L)):
        assert BNMTF.muG[j,l] != 1. / BNMTF.exp_lambdaG[l]
        assert BNMTF.tauG[j,l] == 1.
        mu, tau, x = BNMTF.muG[j,l], BNMTF.tauG[j,l], -BNMTF.muG[j,l]*math.sqrt(BNMTF.tauG[j,l])
        assert BNMTF.exp_G[j,l] == mu + 1./math.sqrt(tau) * scipy.stats.norm.pdf(x)/(0.5*scipy.special.erfc(x/math.sqrt(2)))
    for k,l in itertools.product(xrange(0,K),xrange(0,L)):
        assert BNMTF.muS[k,l] != 1. / (BNMTF.exp_lambdaF[k] * BNMTF.exp_lambdaG[l])
        assert BNMTF.tauS[k,l] == 1.
        mu, tau, x = BNMTF.muS[k,l], BNMTF.tauS[k,l], -BNMTF.muS[k,l]*math.sqrt(BNMTF.tauS[k,l])
        assert BNMTF.exp_S[k,l] == mu + 1./math.sqrt(tau) * scipy.stats.norm.pdf(x)/(0.5*scipy.special.erfc(x/math.sqrt(2)))
    
    # Initialisation of F and G using Kmeans
    init = 'kmeans'
    BNMTF = bnmtf_ard_vb_2(R,M,K,L,priors)
    BNMTF.initialise(init)
    
    for i,k in itertools.product(xrange(0,I),xrange(0,K)):
        assert BNMTF.muF[i,k] == 0. or BNMTF.muF[i,k] == 1.
        assert BNMTF.tauF[i,k] == 1.
        mu, tau, x = BNMTF.muF[i,k], BNMTF.tauF[i,k], -BNMTF.muF[i,k]*math.sqrt(BNMTF.tauF[i,k])
        assert BNMTF.exp_F[i,k] == mu + 1./math.sqrt(tau) * scipy.stats.norm.pdf(x)/(0.5*scipy.special.erfc(x/math.sqrt(2)))
    for j,l in itertools.product(xrange(0,J),xrange(0,L)):
        assert BNMTF.muG[j,l] == 0. or BNMTF.muG[j,l] == 1.
        assert BNMTF.tauG[j,l] == 1.
        mu, tau, x = BNMTF.muG[j,l], BNMTF.tauG[j,l], -BNMTF.muG[j,l]*math.sqrt(BNMTF.tauG[j,l])
        assert BNMTF.exp_G[j,l] == mu + 1./math.sqrt(tau) * scipy.stats.norm.pdf(x)/(0.5*scipy.special.erfc(x/math.sqrt(2)))
    for k,l in itertools.product(xrange(0,K),xrange(0,L)):
        assert BNMTF.muS[k,l] != 1. / (BNMTF.exp_lambdaF[k] * BNMTF.exp_lambdaG[l])
        assert BNMTF.tauS[k,l] == 1.
        mu, tau, x = BNMTF.muS[k,l], BNMTF.tauS[k,l], -BNMTF.muS[k,l]*math.sqrt(BNMTF.tauS[k,l])
        assert BNMTF.exp_S[k,l] == mu + 1./math.sqrt(tau) * scipy.stats.norm.pdf(x)/(0.5*scipy.special.erfc(x/math.sqrt(2)))
    
        
""" Test computing the ELBO. """
def test_elbo():
    I,J,K,L = 5,3,2,4
    R = numpy.ones((I,J))
    M = numpy.ones((I,J))
    M[0,0], M[2,2], M[3,1] = 0, 0, 0
    size_Omega = 12.
    
    alpha0, beta0 = 4., 2.
    alphaR, betaR = 3., 1.    
    priors = { 'alpha0':alpha0, 'beta0':beta0, 'alphaR':alphaR, 'betaR':betaR }
    
    exp_F = 5*numpy.ones((I,K))
    exp_S = 6*numpy.ones((K,L))
    exp_G = 7*numpy.ones((J,L))
    var_F = 11*numpy.ones((I,K))
    var_S = 12*numpy.ones((K,L))
    var_G = 13*numpy.ones((J,L))
    
    exp_lambdaF = 22.*numpy.ones(K)
    exp_lambdaG = 24.*numpy.ones(L)
    exp_loglambdaF = 25.*numpy.ones(K)
    exp_loglambdaG = 27.*numpy.ones(L)
    
    exp_tau = 8.
    exp_logtau = 9.
    
    muF = 14*numpy.ones((I,K))
    muS = 15*numpy.ones((K,L))
    muG = 16*numpy.ones((J,L))
    tauF = numpy.ones((I,K))/100.
    tauS = numpy.ones((K,L))/101.
    tauG = numpy.ones((J,L))/102.
    
    alphaF = 28.*numpy.ones(K)
    alphaG = 30.*numpy.ones(L)
    betaF = 31.*numpy.ones(K)
    betaG = 33.*numpy.ones(L)
    
    alphaR_s = 20.
    betaR_s = 21.
    
    # Compute the ELBO elements
    
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
    
    p_R = size_Omega/2.*(exp_logtau - math.log(2*math.pi)) - exp_tau/2.*(33828492+12763008)
    p_tau = alphaR*math.log(betaR) - numpy.log(math.gamma(alphaR)) + (alphaR-1)*exp_logtau - betaR*exp_tau
    p_F = I*K*(exp_loglambdaF[0] - exp_lambdaF[0]*exp_F[0,0])
    p_S = K*L*(exp_loglambdaF[0] + exp_loglambdaG[0] - exp_lambdaF[0]*exp_lambdaG[0]*exp_S[0,0])
    p_G = J*L*(exp_loglambdaG[0] - exp_lambdaG[0]*exp_G[0,0])
    p_lambdaF = K*(alpha0*math.log(beta0) - numpy.log(math.gamma(alpha0)) + (alpha0-1)*exp_loglambdaF[0] - beta0*exp_lambdaF[0])
    p_lambdaG = L*(alpha0*math.log(beta0) - numpy.log(math.gamma(alpha0)) + (alpha0-1)*exp_loglambdaG[0] - beta0*exp_lambdaG[0])
    q_tau = alphaR_s*math.log(betaR_s) - numpy.log(math.gamma(alphaR_s)) + (alphaR_s-1)*exp_logtau - betaR_s*exp_tau
    q_F = I*K*(1./2.*math.log(tauF[0,0]) - 1./2.*math.log(2*math.pi) - tauF[0,0]/2. * ( var_F[0,0] + (exp_F[0,0] - muF[0,0])**2 ) \
          - math.log(1.-0.080756659233771066) )
    q_S = K*L*(1./2.*math.log(tauS[0,0]) - 1./2.*math.log(2*math.pi) - tauS[0,0]/2. * ( var_S[0,0] + (exp_S[0,0] - muS[0,0])**2 ) \
          - math.log(1.-0.067776752211548219) )
    q_G = J*L*(1./2.*math.log(tauG[0,0]) - 1./2.*math.log(2*math.pi) - tauG[0,0]/2. * ( var_G[0,0] + (exp_G[0,0] - muG[0,0])**2 ) \
          - math.log(1.-0.056570004076003155) )
    q_lambdaF = K*(alphaF[0]*math.log(betaF[0]) - numpy.log(math.gamma(alphaF[0])) + (alphaF[0]-1)*exp_loglambdaF[0] - betaF[0]*exp_lambdaF[0])
    q_lambdaG = L*(alphaG[0]*math.log(betaG[0]) - numpy.log(math.gamma(alphaG[0])) + (alphaG[0]-1)*exp_loglambdaG[0] - betaG[0]*exp_lambdaG[0])
    
    # Initialise the fields     
    BNMTF = bnmtf_ard_vb_2(R,M,K,L,priors)
    BNMTF.exp_F = exp_F
    BNMTF.exp_S = exp_S
    BNMTF.exp_G = exp_G
    BNMTF.var_F = var_F
    BNMTF.var_S = var_S
    BNMTF.var_G = var_G
    BNMTF.exp_lambdaF = exp_lambdaF
    BNMTF.exp_lambdaG = exp_lambdaG
    BNMTF.exp_loglambdaF = exp_loglambdaF
    BNMTF.exp_loglambdaG = exp_loglambdaG
    BNMTF.exp_tau = exp_tau
    BNMTF.exp_logtau = exp_logtau
    BNMTF.muF = muF
    BNMTF.muS = muS
    BNMTF.muG = muG
    BNMTF.tauF = tauF
    BNMTF.tauS = tauS
    BNMTF.tauG = tauG
    BNMTF.alphaF = alphaF
    BNMTF.alphaG = alphaG
    BNMTF.betaF = betaF
    BNMTF.betaG = betaG
    BNMTF.alphaR_s = alphaR_s
    BNMTF.betaR_s = betaR_s
    
    # Check for equivalence
    ELBO = BNMTF.elbo()
    assert BNMTF.p_R == p_R
    assert BNMTF.p_tau == p_tau
    assert BNMTF.p_F == p_F
    assert BNMTF.p_S == p_S
    assert BNMTF.p_G == p_G
    assert BNMTF.p_lambdaF == p_lambdaF
    assert BNMTF.p_lambdaG == p_lambdaG
    assert BNMTF.q_tau == q_tau
    assert BNMTF.q_F == q_F
    assert BNMTF.q_S == q_S
    assert BNMTF.q_G == q_G
    assert BNMTF.q_lambdaF == q_lambdaF
    assert BNMTF.q_lambdaG == q_lambdaG
    assert ELBO == p_R + p_tau + p_F + p_G + p_S + p_lambdaF + p_lambdaG - q_tau - q_F - q_G - q_S - q_lambdaF - q_lambdaG
    
        
""" Test updating parameters tau, F, S, G, lambdas. """          
I,J,K,L = 5,3,2,4
R = numpy.ones((I,J))
M = numpy.ones((I,J))
M[0,0], M[2,2], M[3,1] = 0, 0, 0 # size Omega = 12

exp_lambdaF = 2*numpy.ones(K)
lambdaS = 3.
exp_lambdaG = 4*numpy.ones(L)

alpha0, beta0 = 4., 2.
alphaR, betaR = 3., 1.    
priors = { 'alpha0':alpha0, 'beta0':beta0, 'alphaR':alphaR, 'betaR':betaR }

init = 'exp'

def test_exp_square_diff():
    BNMTF = bnmtf_ard_vb_2(R,M,K,L,priors)
    BNMTF.exp_F = numpy.array([1./exp_lambdaF for i in range(0,I)]) #[[1./2.]]
    BNMTF.exp_S = 1./lambdaS * numpy.ones((K,L)) #[[1./3.]]
    BNMTF.exp_G = numpy.array([1./exp_lambdaG for i in range(0,J)]) #[[1./4.]]
    BNMTF.var_F = numpy.ones((I,K))*2 #[[2.]]
    BNMTF.var_S = numpy.ones((K,L))*3 #[[3.]]
    BNMTF.var_G = numpy.ones((J,L))*4 #[[4.]]
    # expF * expS * expV.T = [[1./3.]]. (varF+expF^2)=2.25, (varS+expS^2)=3.+1./9., (varG+expG^2)=4.0625
    # 12.*(4./9.) + 12.*(2*4*(2.25*(3.+1./9.)*4.0625-1./4.*1./9.*1./16. + 2./3./4.*((4-1)/3./4.) +4./2./3.*((2-1)/2./3.) ))
    exp_square_diff = 2749+5./6. # 
    assert abs(BNMTF.exp_square_diff() - exp_square_diff) < 0.000000000001

def test_update_tau():
    BNMTF = bnmtf_ard_vb_2(R,M,K,L,priors)
    BNMTF.exp_F = numpy.array([1./exp_lambdaF for i in range(0,I)]) #[[1./2.]]
    BNMTF.exp_S = 1./lambdaS * numpy.ones((K,L)) #[[1./3.]]
    BNMTF.exp_G = numpy.array([1./exp_lambdaG for i in range(0,J)]) #[[1./4.]]
    BNMTF.var_F = numpy.ones((I,K))*2 #[[2.]]
    BNMTF.var_S = numpy.ones((K,L))*3 #[[3.]]
    BNMTF.var_G = numpy.ones((J,L))*4 #[[4.]]
    BNMTF.update_tau()
    assert BNMTF.alphaR_s == alphaR + 12./2.
    assert abs(BNMTF.betaR_s - (betaR + (2749+5./6.)/2.)) < 0.000000000001
    
def test_update_F():
    for k in range(0,K):
        BNMTF = bnmtf_ard_vb_2(R,M,K,L,priors)
        BNMTF.muF = numpy.zeros((I,K))
        BNMTF.tauF = numpy.zeros((I,K))
        BNMTF.exp_lambdaF = exp_lambdaF
        BNMTF.exp_F = numpy.array([1./exp_lambdaF for i in range(0,I)]) #[[1./2.]]
        BNMTF.exp_S = 1./lambdaS * numpy.ones((K,L)) #[[1./3.]]
        BNMTF.exp_G = numpy.array([1./exp_lambdaG for i in range(0,J)]) #[[1./4.]]
        BNMTF.var_F = numpy.ones((I,K))*2 #[[2.]]
        BNMTF.var_S = numpy.ones((K,L))*3 #[[3.]]
        BNMTF.var_G = numpy.ones((J,L))*4 #[[4.]]
        BNMTF.exp_tau = 3.
        BNMTF.update_F(k)
        
        for i in range(0,I):
            tauFik = 3. * sum([
                sum([BNMTF.exp_S[k,l]*BNMTF.exp_G[j,l] for l in range(0,L)])**2 \
                + sum([(BNMTF.var_S[k,l]+BNMTF.exp_S[k,l]**2)*(BNMTF.var_G[j,l]+BNMTF.exp_G[j,l]**2) - BNMTF.exp_S[k,l]**2*BNMTF.exp_G[j,l]**2 for l in range(0,L)])
            for j in range(0,J) if M[i,j]])
            muFik = 1./tauFik * (-exp_lambdaF[k] + BNMTF.exp_tau * sum([
                sum([BNMTF.exp_S[k,l]*BNMTF.exp_G[j,l] for l in range(0,L)]) * \
                (R[i,j] - sum([BNMTF.exp_F[i,kp]*BNMTF.exp_S[kp,l]*BNMTF.exp_G[j,l] for kp,l in itertools.product(xrange(0,K),xrange(0,L)) if kp != k]))
                - sum([
                    BNMTF.exp_S[k,l] * BNMTF.var_G[j,l] * sum([BNMTF.exp_F[i,kp] * BNMTF.exp_S[kp,l] for kp in range(0,K) if kp != k])
                for l in range(0,L)])
            for j in range(0,J) if M[i,j]]))
                
            assert BNMTF.tauF[i,k] == tauFik
            assert abs(BNMTF.muF[i,k] - muFik) < 0.00000000000000001
    
def test_update_S():
    for k,l in itertools.product(range(0,K),range(0,L)):
        BNMTF = bnmtf_ard_vb_2(R,M,K,L,priors)
        BNMTF.muS = numpy.zeros((K,L))
        BNMTF.tauS = numpy.zeros((K,L))
        BNMTF.exp_lambdaF = exp_lambdaF
        BNMTF.exp_lambdaG = exp_lambdaG
        BNMTF.exp_F = numpy.array([1./exp_lambdaF for i in range(0,I)]) #[[1./2.]]
        BNMTF.exp_S = 1./lambdaS * numpy.ones((K,L)) #[[1./3.]]
        BNMTF.exp_G = numpy.array([1./exp_lambdaG for i in range(0,J)]) #[[1./4.]]
        BNMTF.var_F = numpy.ones((I,K))*2 #[[2.]]
        BNMTF.var_S = numpy.ones((K,L))*3 #[[3.]]
        BNMTF.var_G = numpy.ones((J,L))*4 #[[4.]]
        BNMTF.exp_tau = 3.
        BNMTF.update_S(k,l)
        
        tauSkl = 3. * sum([
            BNMTF.exp_F[i,k]**2 * BNMTF.exp_G[j,l]**2 \
            + (BNMTF.var_F[i,k]+BNMTF.exp_F[i,k]**2)*(BNMTF.var_G[j,l]+BNMTF.exp_G[j,l]**2) - BNMTF.exp_F[i,k]**2*BNMTF.exp_G[j,l]**2
        for i,j in itertools.product(xrange(0,I),xrange(0,J)) if M[i,j]])
        muSkl = 1./tauSkl * (-exp_lambdaF[k]*exp_lambdaG[l] + BNMTF.exp_tau * sum([
            BNMTF.exp_F[i,k]*BNMTF.exp_G[j,l]*(R[i,j] - sum([BNMTF.exp_F[i,kp]*BNMTF.exp_S[kp,lp]*BNMTF.exp_G[j,lp] for kp,lp in itertools.product(xrange(0,K),xrange(0,L)) if (kp != k or lp != l)]))
            - BNMTF.var_F[i,k] * BNMTF.exp_G[j,l] * sum([BNMTF.exp_S[k,lp] * BNMTF.exp_G[j,lp] for lp in range(0,L) if lp != l])
            - BNMTF.exp_F[i,k] * BNMTF.var_G[j,l] * sum([BNMTF.exp_F[i,kp] * BNMTF.exp_S[kp,l] for kp in range(0,K) if kp != k])
        for i,j in itertools.product(xrange(0,I),xrange(0,J)) if M[i,j]]))
        
        assert BNMTF.tauS[k,l] == tauSkl
        assert abs(BNMTF.muS[k,l] - muSkl) < 0.0000000000000001
    
def test_update_G():
    for l in range(0,L):
        BNMTF = bnmtf_ard_vb_2(R,M,K,L,priors)
        BNMTF.muG = numpy.zeros((J,L))
        BNMTF.tauG = numpy.zeros((J,L))
        BNMTF.exp_lambdaG = exp_lambdaG
        BNMTF.exp_F = numpy.array([1./exp_lambdaF for i in range(0,I)]) #[[1./2.]]
        BNMTF.exp_S = 1./lambdaS * numpy.ones((K,L)) #[[1./3.]]
        BNMTF.exp_G = numpy.array([1./exp_lambdaG for i in range(0,J)]) #[[1./4.]]
        BNMTF.var_F = numpy.ones((I,K))*2 #[[2.]]
        BNMTF.var_S = numpy.ones((K,L))*3 #[[3.]]
        BNMTF.var_G = numpy.ones((J,L))*4 #[[4.]]
        BNMTF.exp_tau = 3.
        BNMTF.update_G(l)
        
        for j in range(0,J):
            tauGjl = 3. * sum([
                sum([BNMTF.exp_F[i,k]*BNMTF.exp_S[k,l] for k in range(0,K)])**2 \
                + sum([(BNMTF.var_S[k,l]+BNMTF.exp_S[k,l]**2)*(BNMTF.var_F[i,k]+BNMTF.exp_F[i,k]**2) - BNMTF.exp_S[k,l]**2*BNMTF.exp_F[i,k]**2 for k in range(0,K)])
            for i in range(0,I) if M[i,j]])
            muGjl = 1./tauGjl * (-exp_lambdaG[l] + BNMTF.exp_tau * sum([
                sum([BNMTF.exp_F[i,k]*BNMTF.exp_S[k,l] for k in range(0,K)]) * \
                (R[i,j] - sum([BNMTF.exp_F[i,k]*BNMTF.exp_S[k,lp]*BNMTF.exp_G[j,lp] for k,lp in itertools.product(xrange(0,K),xrange(0,L)) if lp != l]))
                - sum([
                    BNMTF.var_F[i,k] * BNMTF.exp_S[k,l] * sum([BNMTF.exp_S[k,lp] * BNMTF.exp_G[j,lp] for lp in range(0,L) if lp != l])
                for k in range(0,K)])
            for i in range(0,I) if M[i,j]]))
            
            assert BNMTF.tauG[j,l] == tauGjl
            assert BNMTF.muG[j,l] == muGjl

def test_update_lambdaF():
    for k in range(0,K):
        BNMTF = bnmtf_ard_vb_2(R,M,K,L,priors)
        BNMTF.initialise('exp')
        BNMTF.exp_F = numpy.array([1./exp_lambdaF for i in range(0,I)]) #[[1./2.]]
        BNMTF.exp_S = 1./lambdaS * numpy.ones((K,L)) #[[1./3.]]
        BNMTF.exp_lambdaG = exp_lambdaG #[[4.]]
        BNMTF.update_lambdaF(k)
        assert BNMTF.alphaF[k] == alpha0 + I + L
        assert BNMTF.betaF[k] == beta0 + 1./2.*I + L*4./3.

def test_update_lambdaG():
    for l in range(0,L):
        BNMTF = bnmtf_ard_vb_2(R,M,K,L,priors)
        BNMTF.initialise('exp')
        BNMTF.exp_G = numpy.array([1./exp_lambdaG for i in range(0,J)]) #[[1./4.]]
        BNMTF.exp_S = 1./lambdaS * numpy.ones((K,L)) #[[1./3.]]
        BNMTF.exp_lambdaF = exp_lambdaF #[[2.]]
        BNMTF.update_lambdaG(l)
        assert BNMTF.alphaG[l] == alpha0 + J + K
        assert BNMTF.betaG[l] == beta0 + 1./4.*J + K*2./3.
    

""" Test computing expectation, variance F, S, G, tau """     
def test_update_exp_tau():
    BNMTF = bnmtf_ard_vb_2(R,M,K,L,priors)
    BNMTF.initialise(init)  
    BNMTF.exp_F = numpy.array([1./exp_lambdaF for i in range(0,I)]) #[[1./2.]]
    BNMTF.exp_S = 1./lambdaS * numpy.ones((K,L)) #[[1./3.]]
    BNMTF.exp_G = numpy.array([1./exp_lambdaG for i in range(0,J)]) #[[1./4.]]
    BNMTF.var_F = numpy.ones((I,K))*2 #[[2.]]
    BNMTF.var_S = numpy.ones((K,L))*3 #[[3.]]
    BNMTF.var_G = numpy.ones((J,L))*4 #[[4.]]
    BNMTF.update_tau()
    BNMTF.update_exp_tau()
    
    assert abs(BNMTF.exp_tau - 9./1375.91666667) < 0.0000000000001
    assert abs(BNMTF.exp_logtau - (2.14064147795560999 - math.log(1375.91666667))) < 0.00000000001
    
def test_update_exp_F():
    for k in range(0,K):
        BNMTF = bnmtf_ard_vb_2(R,M,K,L,priors)
        BNMTF.initialise(init)
        # muF = [[0.5]], tauF = [[4.]]
        BNMTF.muF = 1./2.*numpy.ones((I,K))
        BNMTF.tauF = 4*numpy.ones((I,K))  
        BNMTF.update_exp_F(k) #-mu*sqrt(tau) = -0.5*2 = -1. lambda(1) = 0.241971 / (1-0.1587) = 0.2876155949126352. gamma = 0.37033832534958433
        for i in range(0,I):        
            assert abs(BNMTF.exp_F[i,k] - (0.5 + 1./2. * 0.2876155949126352)) < 0.00001
            assert abs(BNMTF.var_F[i,k] - 1./4.*(1.-0.37033832534958433)) < 0.00001

def test_update_exp_S():
    for k,l in itertools.product(xrange(0,K),xrange(0,L)):
        BNMTF = bnmtf_ard_vb_2(R,M,K,L,priors)
        BNMTF.initialise(init) 
        # muS = [[1./3.]], tauS = [[4.]]
        BNMTF.muS = 1./3.*numpy.ones((K,L))
        BNMTF.tauS = 4*numpy.ones((K,L)) 
        BNMTF.update_exp_S(k,l) #-mu*sqrt(tau) = -2./3., lambda(..) = 0.319448 / (1-0.2525) = 0.4273551839464883, gamma = 
        assert abs(BNMTF.exp_S[k,l] - (1./3. + 1./2. * 0.4273551839464883)) < 0.00001
        assert abs(BNMTF.var_S[k,l] - 1./4.*(1. - 0.4675359092102624)) < 0.00001

def test_update_exp_G():
    for l in range(0,L):
        BNMTF = bnmtf_ard_vb_2(R,M,K,L,priors)
        BNMTF.initialise(init) 
        # muG = [[1./4.]], tauG = [[4.]]
        BNMTF.muG = 1./4.*numpy.ones((J,L))
        BNMTF.tauG = 4*numpy.ones((J,L)) 
        BNMTF.update_exp_G(l) #-mu*sqrt(tau) = -0.5., lambda(..) = 0.352065 / (1-0.3085) = 0.5091323210412148, gamma = 0.5137818808494219
        for j in range(0,J):        
            assert abs(BNMTF.exp_G[j,l] - (1./4. + 1./2. * 0.5091323210412148)) < 0.0001
            assert abs(BNMTF.var_G[j,l] - 1./4.*(1. - 0.5137818808494219)) < 0.0001
    
def test_update_exp_lambdaF():
    for k in range(0,K):
        BNMTF = bnmtf_ard_vb_2(R,M,K,L,priors)
        BNMTF.initialise(init)
        # alphaF = [[0.5]], betaF = [[4.]]
        BNMTF.alphaF = 1./2.*numpy.ones(K)
        BNMTF.betaF = 4*numpy.ones(K) 
        BNMTF.update_exp_lambdaF(k)
        assert BNMTF.exp_lambdaF[k] == 0.5 / 4.
        assert BNMTF.exp_loglambdaF[k] == scipy.special.psi(0.5) - math.log(4.)
    
def test_update_exp_lambdaG():
    for l in range(0,L):
        BNMTF = bnmtf_ard_vb_2(R,M,K,L,priors)
        BNMTF.initialise(init)
        # alphaG = [[1./4.]], betaG = [[4.]]
        BNMTF.alphaG = 1./4.*numpy.ones(L)
        BNMTF.betaG = 4*numpy.ones(L) 
        BNMTF.update_exp_lambdaG(l)
        assert BNMTF.exp_lambdaG[l] == (1./4.) / 4.
        assert BNMTF.exp_loglambdaG[l] == scipy.special.psi(1./4.) - math.log(4.)
    

""" Test two iterations of run(), and that all values have changed. """
def test_run():
    I,J,K,L = 10,5,3,2
    R = numpy.ones((I,J))
    M = numpy.ones((I,J))
    M[0,0], M[2,2], M[3,1] = 0, 0, 0
    R[0,1], R[0,2] = 2., 3.
    
    alpha0, beta0 = 4., 2.
    alphaR, betaR = 3., 1.    
    priors = { 'alpha0':alpha0, 'beta0':beta0, 'alphaR':alphaR, 'betaR':betaR }
    
    iterations = 2
    
    BNMTF = bnmtf_ard_vb_2(R,M,K,L,priors)
    BNMTF.initialise(init='exp')
    BNMTF.run(iterations)
    
    for i,k in itertools.product(xrange(0,I),xrange(0,K)):
        # lambdaF
        assert BNMTF.alphaF[k] != (alpha0 + 1.) / (beta0 + 1.)
        assert BNMTF.betaF[k] != (alpha0 + 1.) / (beta0 + 1.)
        assert BNMTF.exp_lambdaF[k] != 1.
        assert BNMTF.exp_loglambdaF[k] != scipy.special.psi(alpha0 + 1.) - math.log(beta0 + 1.)
        # F
        assert BNMTF.muF[i,k] != 1. / (alpha0 / beta0)
        assert BNMTF.tauF[i,k] != 1.
        assert BNMTF.exp_F[i,k] != numpy.inf and not math.isnan(BNMTF.exp_F[i,k])
        assert BNMTF.var_F[i,k] != numpy.inf and not math.isnan(BNMTF.var_F[i,k])
    for k,l in itertools.product(xrange(0,K),xrange(0,L)):
        # S
        assert BNMTF.muS[k,l] != 1. / (alpha0 / beta0)
        assert BNMTF.tauS[k,l] != 1.
        assert BNMTF.exp_S[k,l] != numpy.inf and not math.isnan(BNMTF.exp_S[k,l])
        assert BNMTF.var_S[k,l] != numpy.inf and not math.isnan(BNMTF.var_S[k,l])
    for j,l in itertools.product(xrange(0,J),xrange(0,L)):
        # lambdaG
        assert BNMTF.alphaG[l] != (alpha0 + 1.) / (beta0 + 1.)
        assert BNMTF.betaG[l] != (alpha0 + 1.) / (beta0 + 1.)
        assert BNMTF.exp_lambdaG[l] != 1.
        assert BNMTF.exp_loglambdaG[l] != scipy.special.psi(alpha0 + 1.) - math.log(beta0 + 1.)
        # G
        assert BNMTF.muG[j,l] != 1. / (alpha0 / beta0)
        assert BNMTF.tauG[j,l] != 1.
        assert BNMTF.exp_G[j,l] != numpy.inf and not math.isnan(BNMTF.exp_G[j,l])
        assert BNMTF.var_G[j,l] != numpy.inf and not math.isnan(BNMTF.var_G[j,l])
    assert BNMTF.alphaR_s != alphaR
    assert BNMTF.betaR_s != betaR
    assert BNMTF.exp_tau != numpy.inf and not math.isnan(BNMTF.exp_tau)
    assert BNMTF.exp_logtau != numpy.inf and not math.isnan(BNMTF.exp_logtau)
    

""" Test computing the performance of the predictions using the expectations """
def test_predict():
    (I,J,K,L) = (5,3,2,4)
    R = numpy.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15]],dtype=float)
    M = numpy.ones((I,J))
    
    alpha0, beta0 = 4., 2.
    alphaR, betaR = 3., 1.    
    priors = { 'alpha0':alpha0, 'beta0':beta0, 'alphaR':alphaR, 'betaR':betaR }
    
    exp_F = numpy.array([[125.,126.],[126.,126.],[126.,126.],[126.,126.],[126.,126.]])
    exp_S = numpy.array([[84.,84.,84.,84.],[84.,84.,84.,84.]])
    exp_G = numpy.array([[42.,42.,42.,42.],[42.,42.,42.,42.],[42.,42.,42.,42.]])
    
    M_test = numpy.array([[0,0,1],[0,1,0],[0,0,0],[1,1,0],[0,0,0]]) #R->3,5,10,11, R_pred->3542112,3556224,3556224,3556224
    MSE = ((3.-3542112.)**2 + (5.-3556224.)**2 + (10.-3556224.)**2 + (11.-3556224.)**2) / 4.
    R2 = 1. - ((3.-3542112.)**2 + (5.-3556224.)**2 + (10.-3556224.)**2 + (11.-3556224.)**2) / (4.25**2+2.25**2+2.75**2+3.75**2) #mean=7.25
    Rp = 357. / ( math.sqrt(44.75) * math.sqrt(5292.) ) #mean=7.25,var=44.75, mean_pred=3552696,var_pred=5292, corr=(-4.25*-63 + -2.25*21 + 2.75*21 + 3.75*21)
    
    BNMTF = bnmtf_ard_vb_2(R,M,K,L,priors)
    BNMTF.exp_F = exp_F
    BNMTF.exp_S = exp_S
    BNMTF.exp_G = exp_G
    performances = BNMTF.predict(M_test)
    
    assert performances['MSE'] == MSE
    assert performances['R^2'] == R2
    assert performances['Rp'] == Rp
       
        
""" Test the evaluation measures MSE, R^2, Rp """
def test_compute_statistics():
    R = numpy.array([[1,2],[3,4]],dtype=float)
    M = numpy.array([[1,1],[0,1]])
    K, L = 3, 4
    
    alpha0, beta0 = 4., 2.
    alphaR, betaR = 3., 1.    
    priors = { 'alpha0':alpha0, 'beta0':beta0, 'alphaR':alphaR, 'betaR':betaR }
    
    BNMTF = bnmtf_ard_vb_2(R,M,K,L,priors)
    
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
    
    BNMTF = bnmtf_ard_vb_2(R,M,K,L,priors)
    BNMTF.exp_F = numpy.ones((I,K))
    BNMTF.exp_S = 2*numpy.ones((K,L))
    BNMTF.exp_G = 3*numpy.ones((J,L))
    BNMTF.exp_logtau = 5.
    BNMTF.exp_tau = 3.
    # expU*expV.T = [[72.]]
    
    log_likelihood = 3./2.*(5.-math.log(2*math.pi)) - 3./2. * (71**2 + 70**2 + 68**2)
    AIC = -2*log_likelihood +2*(2*3+3*4+2*4+3+4+3*4)
    BIC = -2*log_likelihood +(2*3+3*4+2*4+3+4+3*4)*math.log(3)
    MSE = (71**2+70**2+68**2)/3.
    
    assert log_likelihood == BNMTF.quality('loglikelihood')
    assert AIC == BNMTF.quality('AIC')
    assert BIC == BNMTF.quality('BIC')
    assert MSE == BNMTF.quality('MSE')
    with pytest.raises(AssertionError) as error:
        BNMTF.quality('FAIL')
    assert str(error.value) == "Unrecognised metric for model quality: FAIL."