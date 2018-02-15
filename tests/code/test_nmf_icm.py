"""
Tests for the Iterated Conditional Modes algorithm.
"""

import sys, os
project_location = os.path.dirname(__file__)+"/../../../"
sys.path.append(project_location)

import numpy, math, pytest, itertools
from BNMTF_ARD.code.models.nmf_icm import nmf_icm


""" Test constructor """
def test_init():
    # Test getting an exception when R and M are different sizes, and when R is not a 2D array.
    R1 = numpy.ones(3)
    M = numpy.ones((2,3))
    I,J,K = 5,3,1
    lambdaU = numpy.ones((I,K))
    lambdaV = numpy.ones((J,K))
    alphatau, betatau = 3, 1    
    alpha0, beta0 = 6, 2
    hyperparams = { 'alphatau':alphatau, 'betatau':betatau, 'alpha0':alpha0, 'beta0':beta0 }
    ARD = True
    
    # Test R is 2-dim
    with pytest.raises(AssertionError) as error:
        nmf_icm(R1,M,K,ARD,hyperparams)
    assert str(error.value) == "Input matrix R is not a two-dimensional array, but instead 1-dimensional."
    
    R2 = numpy.ones((4,3,2))
    with pytest.raises(AssertionError) as error:
        nmf_icm(R2,M,K,ARD,hyperparams)
    assert str(error.value) == "Input matrix R is not a two-dimensional array, but instead 3-dimensional."
    
    # Test R is the same shape as M
    R3 = numpy.ones((3,2))
    with pytest.raises(AssertionError) as error:
        nmf_icm(R3,M,K,ARD,hyperparams)
    assert str(error.value) == "Input matrix R is not of the same size as the indicator matrix M: (3, 2) and (2, 3) respectively."
    
    # Similarly for lambdaU, lambdaV
    R4 = numpy.ones((2,3))
    lambdaU = numpy.ones((2+1,1))
    hyperparams = { 'alphatau':alphatau, 'betatau':betatau, 'lambdaU':lambdaU, 'lambdaV':lambdaV }
    with pytest.raises(AssertionError) as error:
        nmf_icm(R4,M,K,False,hyperparams)
    assert str(error.value) == "Prior matrix lambdaU has the wrong shape: (3, 1) instead of (2, 1)."
    
    lambdaU = numpy.ones((2,1))
    lambdaV = numpy.ones((3+1,1))
    hyperparams = { 'alphatau':alphatau, 'betatau':betatau, 'lambdaU':lambdaU, 'lambdaV':lambdaV }
    with pytest.raises(AssertionError) as error:
        nmf_icm(R4,M,K,False,hyperparams)
    assert str(error.value) == "Prior matrix lambdaV has the wrong shape: (4, 1) instead of (3, 1)."
    
    # Test getting an exception if a row or column is entirely unknown
    M1 = [[1,1,1],[0,0,0]]
    M2 = [[1,1,0],[1,0,0]]
    hyperparams = { 'alphatau':alphatau, 'betatau':betatau, 'alpha0':alpha0, 'beta0':beta0 }
    
    with pytest.raises(AssertionError) as error:
        nmf_icm(R4,M1,K,ARD,hyperparams)
    assert str(error.value) == "Fully unobserved row in R, row 1."
    with pytest.raises(AssertionError) as error:
        nmf_icm(R4,M2,K,ARD,hyperparams)
    assert str(error.value) == "Fully unobserved column in R, column 2."
    
    # Finally, a successful case
    I,J,K = 3,2,2
    R5 = 2*numpy.ones((I,J))
    M = numpy.ones((I,J))
    hyperparams = { 'alphatau':alphatau, 'betatau':betatau, 'alpha0':alpha0, 'beta0':beta0 }
    BNMF = nmf_icm(R5,M,K,ARD,hyperparams)
    
    assert numpy.array_equal(BNMF.R,R5)
    assert numpy.array_equal(BNMF.M,M)
    assert BNMF.I == I
    assert BNMF.J == J
    assert BNMF.K == K
    assert BNMF.size_Omega == I*J
    assert BNMF.alphatau == alphatau
    assert BNMF.betatau == betatau
    assert BNMF.alpha0 == alpha0
    assert BNMF.beta0 == beta0
    
    # And when lambdaU and lambdaV are integers    
    I,J,K = 3,2,2
    R5 = 2*numpy.ones((I,J))
    lambdaU = 3.
    lambdaV = 4.
    M = numpy.ones((I,J))
    hyperparams = { 'alphatau':alphatau, 'betatau':betatau, 'lambdaU':lambdaU, 'lambdaV':lambdaV }
    BNMF = nmf_icm(R5,M,K,False,hyperparams)
    
    assert numpy.array_equal(BNMF.R,R5)
    assert numpy.array_equal(BNMF.M,M)
    assert BNMF.I == I
    assert BNMF.J == J
    assert BNMF.K == K
    assert BNMF.size_Omega == I*J
    assert BNMF.alphatau == alphatau
    assert BNMF.betatau == betatau
    assert numpy.array_equal(BNMF.lambdaU,lambdaU*numpy.ones((I,K)))
    assert numpy.array_equal(BNMF.lambdaV,lambdaV*numpy.ones((J,K)))
    
    
""" Test initialing parameters """
def test_initialise():
    I,J,K = 5,3,2
    R = numpy.ones((I,J))
    M = numpy.ones((I,J))
    
    alphatau, betatau = 3, 1
    alpha0, beta0 = 6, 2
    priors = { 'alphatau':alphatau, 'betatau':betatau, 'alpha0':alpha0, 'beta0':beta0 }
    
    # First do a random initialisation, ARD - we can then only check whether values are correctly initialised
    init_UV = 'random'
    BNMF = nmf_icm(R,M,K,True,priors)
    BNMF.initialise(init_UV)
    
    assert BNMF.tau >= 0.0
    for k in range(K):
        assert BNMF.lambdak[k] == alpha0 / beta0
    for i,k in itertools.product(range(I),range(K)):
        assert BNMF.U[i,k] >= 0.0
    for j,k in itertools.product(range(J),range(K)):
        assert BNMF.V[j,k] >= 0.0
        
    # Then initialise with expectation values, no ARD
    lambdaU, lambdaV = 2., 3.
    priors = { 'alphatau':alphatau, 'betatau':betatau, 'lambdaU':lambdaU, 'lambdaV':lambdaV }
    init_UV = 'exp'
    BNMF = nmf_icm(R,M,K,False,priors)
    BNMF.initialise(init_UV)
    
    assert BNMF.tau >= 0.0
    for i,k in itertools.product(range(I),range(K)):
        assert BNMF.U[i,k] == 1./2.
    for j,k in itertools.product(range(J),range(K)):
        assert BNMF.V[j,k] == 1./3.
    
    
""" Test computing values for alpha, beta, mu, tau. """
I,J,K = 5,3,2
R = numpy.ones((I,J))
M = numpy.ones((I,J))
M[0,0], M[2,2], M[3,1] = 0, 0, 0

lambdaU = 2*numpy.ones((I,K))
lambdaV = 3*numpy.ones((J,K))
alphatau, betatau = 3, 1
alpha0, beta0 = 6, 2
hyperparams = { 'alphatau':alphatau, 'betatau':betatau, 'alpha0':alpha0, 'beta0':beta0, 'lambdaU':lambdaU, 'lambdaV':lambdaV }
init = 'exp' #U=1/2,V=1/3

def test_alpha_s():
    BNMF = nmf_icm(R,M,K,False,hyperparams)
    BNMF.initialise(init)
    alpha_s = alphatau + 6.
    assert BNMF.alpha_s() == alpha_s

def test_beta_s():
    BNMF = nmf_icm(R,M,K,False,hyperparams)
    BNMF.initialise(init)
    beta_s = betatau + .5*(12*(2./3.)**2) #U*V.T = [[1/6+1/6,..]]
    assert abs(BNMF.beta_s() - beta_s) < 0.000000000000001

def test_alphak_s():
    BNMF = nmf_icm(R,M,K,True,hyperparams)
    BNMF.initialise(init)
    alphak_s = alpha0 + I + J
    for k in range(K):
        assert BNMF.alphak_s(k) == alphak_s

def test_betak_s():
    BNMF = nmf_icm(R,M,K,True,hyperparams)
    BNMF.initialise(init)
    betak_s = beta0 + I/3. + J/3.
    for k in range(K):
        assert abs(BNMF.betak_s(k) - betak_s) < 0.000000000000001
    
def test_tauU():
    BNMF = nmf_icm(R,M,K,False,hyperparams)
    BNMF.initialise(init)
    BNMF.tau = 3.
    #V^2 = [[1/9,1/9],[1/9,1/9],[1/9,1/9]], sum_j V^2 = [2/9,1/3,2/9,2/9,1/3] (index=i)
    tauU = 3.*numpy.array([[2./9.,2./9.],[1./3.,1./3.],[2./9.,2./9.],[2./9.,2./9.],[1./3.,1./3.]])
    for i,k in itertools.product(range(I),range(K)):
        assert BNMF.tauU(k)[i] == tauU[i,k]
        
def test_muU():
    BNMF = nmf_icm(R,M,K,False,hyperparams)
    BNMF.initialise(init)
    BNMF.tau = 3.
    #U*V^T - Uik*Vjk = [[1/6,..]], so Rij - Ui * Vj + Uik * Vjk = 5/6
    tauU = 3.*numpy.array([[2./9.,2./9.],[1./3.,1./3.],[2./9.,2./9.],[2./9.,2./9.],[1./3.,1./3.]])
    muU = 1./tauU * ( 3. * numpy.array([[2.*(5./6.)*(1./3.),10./18.],[15./18.,15./18.],[10./18.,10./18.],[10./18.,10./18.],[15./18.,15./18.]]) - lambdaU )
    for i,k in itertools.product(range(I),xrange(K)):
        assert abs(BNMF.muU(tauU[:,k],k)[i] - muU[i,k]) < 0.000000000000001
        
def test_tauV():
    BNMF = nmf_icm(R,M,K,False,hyperparams)
    BNMF.initialise(init)
    BNMF.tau = 3.
    #U^2 = [[1/4,1/4],[1/4,1/4],[1/4,1/4],[1/4,1/4],[1/4,1/4]], sum_i U^2 = [1,1,1] (index=j)
    tauV = 3.*numpy.array([[1.,1.],[1.,1.],[1.,1.]])
    for j,k in itertools.product(range(J),range(K)):
        assert BNMF.tauV(k)[j] == tauV[j,k]
        
def test_muV():
    BNMF = nmf_icm(R,M,K,False,hyperparams)
    BNMF.initialise(init)
    BNMF.tau = 3.
    #U*V^T - Uik*Vjk = [[1/6,..]], so Rij - Ui * Vj + Uik * Vjk = 5/6
    tauV = 3.*numpy.array([[1.,1.],[1.,1.],[1.,1.]])
    muV = 1./tauV * ( 3. * numpy.array([[4.*(5./6.)*(1./2.),4.*(5./6.)*(1./2.)],[4.*(5./6.)*(1./2.),4.*(5./6.)*(1./2.)],[4.*(5./6.)*(1./2.),4.*(5./6.)*(1./2.)]]) - lambdaV )
    for j,k in itertools.product(range(0,J),range(K)):
        assert BNMF.muV(tauV[:,k],k)[j] == muV[j,k]
        
        
""" Test some iterations, and that the values have changed in U and V. """
def test_run():
    I,J,K = 10,5,2
    R = numpy.ones((I,J))
    M = numpy.ones((I,J))
    M[0,0], M[2,2], M[3,1] = 0, 0, 0
    
    alphatau, betatau = 3, 1
    alpha0, beta0 = 6, 2
    hyperparams = { 'alphatau':alphatau, 'betatau':betatau, 'alpha0':alpha0, 'beta0':beta0 }
    init_UV = 'exp'
    
    iterations = 15
    
    BNMF = nmf_icm(R,M,K,True,hyperparams)
    BNMF.initialise(init_UV)
    BNMF.run(iterations)
    
    assert BNMF.all_U.shape == (iterations,I,K)
    assert BNMF.all_V.shape == (iterations,J,K)
    assert BNMF.all_lambdak.shape == (iterations,K)
    assert BNMF.all_tau.shape == (iterations,)
    
    for k in range(K):
        assert BNMF.all_lambdak[0,k] != alpha0 / float(beta0)
    for i,k in itertools.product(range(I),range(K)):
        assert BNMF.all_U[0,i,k] != 1. / (alpha0 / float(beta0))
    for j,k in itertools.product(range(J),xrange(K)):
        assert BNMF.all_V[0,j,k] != 1. / (alpha0 / float(beta0))
    assert BNMF.all_tau[0] != alphatau/float(betatau)
    
    
""" Test approximating the expectations for U, V, tau """
def test_approx_expectation():
    burn_in = 2
    thinning = 3 # so index 2,5,8 -> m=3,m=6,m=9
    (I,J,K) = (5,3,2)
    Us = [numpy.ones((I,K)) * 3*m**2 for m in range(1,10+1)] #first is 1's, second is 4's, third is 9's, etc.
    Vs = [numpy.ones((J,K)) * 2*m**2 for m in range(1,10+1)]
    taus = [m**2 for m in range(1,10+1)]
    
    expected_exp_tau = (9.+36.+81.)/3.
    expected_exp_U = numpy.array([[9.+36.+81.,9.+36.+81.],[9.+36.+81.,9.+36.+81.],[9.+36.+81.,9.+36.+81.],[9.+36.+81.,9.+36.+81.],[9.+36.+81.,9.+36.+81.]])
    expected_exp_V = numpy.array([[(9.+36.+81.)*(2./3.),(9.+36.+81.)*(2./3.)],[(9.+36.+81.)*(2./3.),(9.+36.+81.)*(2./3.)],[(9.+36.+81.)*(2./3.),(9.+36.+81.)*(2./3.)]])
    
    R = numpy.ones((I,J))
    M = numpy.ones((I,J))
    lambdaU = 2*numpy.ones((I,K))
    lambdaV = 3*numpy.ones((J,K))
    alphatau, betatau = 3, 1
    hyperparams = { 'alphatau':alphatau, 'betatau':betatau, 'lambdaU':lambdaU, 'lambdaV':lambdaV }
    
    BNMF = nmf_icm(R,M,K,False,hyperparams)
    BNMF.all_U = Us
    BNMF.all_V = Vs
    BNMF.all_tau = taus
    (exp_U, exp_V, exp_tau, exp_lambdak) = BNMF.approx_expectation(burn_in,thinning)
    
    assert exp_lambdak is None
    assert expected_exp_tau == exp_tau
    assert numpy.array_equal(expected_exp_U,exp_U)
    assert numpy.array_equal(expected_exp_V,exp_V)

    
""" Test computing the performance of the predictions using the expectations """
def test_predict():
    burn_in = 2
    thinning = 3 # so index 2,5,8 -> m=3,m=6,m=9
    (I,J,K) = (5,3,2)
    Us = [numpy.ones((I,K)) * 3*m**2 for m in range(1,10+1)] #first is 1's, second is 4's, third is 9's, etc.
    Vs = [numpy.ones((J,K)) * 2*m**2 for m in range(1,10+1)]
    Us[2][0,0] = 24 #instead of 27 - to ensure we do not get 0 variance in our predictions
    taus = [m**2 for m in range(1,10+1)]
    
    R = numpy.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15]],dtype=float)
    M = numpy.ones((I,J))
    lambdaU = 2*numpy.ones((I,K))
    lambdaV = 3*numpy.ones((J,K))
    alphatau, betatau = 3, 1
    hyperparams = { 'alphatau':alphatau, 'betatau':betatau, 'lambdaU':lambdaU, 'lambdaV':lambdaV }
    
    #expected_exp_U = numpy.array([[125.,126.],[126.,126.],[126.,126.],[126.,126.],[126.,126.]])
    #expected_exp_V = numpy.array([[84.,84.],[84.,84.],[84.,84.]])
    #R_pred = numpy.array([[21084.,21084.,21084.],[ 21168.,21168.,21168.],[21168.,21168.,21168.],[21168.,21168.,21168.],[21168.,21168.,21168.]])
    
    M_test = numpy.array([[0,0,1],[0,1,0],[0,0,0],[1,1,0],[0,0,0]]) #R->3,5,10,11, P_pred->21084,21168,21168,21168
    MSE = (444408561. + 447872569. + 447660964. + 447618649) / 4.
    R2 = 1. - (444408561. + 447872569. + 447660964. + 447618649) / (4.25**2+2.25**2+2.75**2+3.75**2) #mean=7.25
    Rp = 357. / ( math.sqrt(44.75) * math.sqrt(5292.) ) #mean=7.25,var=44.75, mean_pred=21147,var_pred=5292, corr=(-4.25*-63 + -2.25*21 + 2.75*21 + 3.75*21)
    
    BNMF = nmf_icm(R,M,K,False,hyperparams)
    BNMF.all_U = Us
    BNMF.all_V = Vs
    BNMF.all_tau = taus
    performances = BNMF.predict(M_test,burn_in,thinning)
    
    assert performances['MSE'] == MSE
    assert performances['R^2'] == R2
    assert performances['Rp'] == Rp
    
    
""" Test the evaluation measures MSE, R^2, Rp """
def test_compute_statistics():
    R = numpy.array([[1,2],[3,4]],dtype=float)
    M = numpy.array([[1,1],[0,1]])
    I, J, K = 2, 2, 3
    lambdaU = 2*numpy.ones((I,K))
    lambdaV = 3*numpy.ones((J,K))
    alphatau, betatau = 3, 1
    hyperparams = { 'alphatau':alphatau, 'betatau':betatau, 'lambdaU':lambdaU, 'lambdaV':lambdaV }
    
    BNMF = nmf_icm(R,M,K,False,hyperparams)
    
    R_pred = numpy.array([[500,550],[1220,1342]],dtype=float)
    M_pred = numpy.array([[0,0],[1,1]])
    
    MSE_pred = (1217**2 + 1338**2) / 2.0
    R2_pred = 1. - (1217**2+1338**2)/(0.5**2+0.5**2) #mean=3.5
    Rp_pred = 61. / ( math.sqrt(.5) * math.sqrt(7442.) ) #mean=3.5,var=0.5,mean_pred=1281,var_pred=7442,cov=61
    
    assert MSE_pred == BNMF.compute_MSE(M_pred,R,R_pred)
    assert R2_pred == BNMF.compute_R2(M_pred,R,R_pred)
    assert Rp_pred == BNMF.compute_Rp(M_pred,R,R_pred)
    
    
""" Test the model quality measures. """
def test_log_likelihood():
    R = numpy.array([[1,2],[3,4]],dtype=float)
    M = numpy.array([[1,1],[0,1]])
    I, J, K = 2, 2, 3
    lambdaU = 2*numpy.ones((I,K))
    lambdaV = 3*numpy.ones((J,K))
    alphatau, betatau = 3, 1
    hyperparams = { 'alphatau':alphatau, 'betatau':betatau, 'lambdaU':lambdaU, 'lambdaV':lambdaV }
    
    iterations = 10
    burnin, thinning = 4, 2
    BNMF = nmf_icm(R,M,K,False,hyperparams)
    BNMF.all_U = [numpy.ones((I,K)) for i in range(0,iterations)]
    BNMF.all_V = [2*numpy.ones((J,K)) for i in range(0,iterations)]
    BNMF.all_tau = [3. for i in range(0,iterations)]
    # expU*expV.T = [[6.]]
    
    log_likelihood = 3./2.*(math.log(3.)-math.log(2*math.pi)) - 3./2. * (5**2 + 4**2 + 2**2)
    AIC = -2*log_likelihood + 2*(2*3+2*3+1)
    BIC = -2*log_likelihood + (2*3+2*3+1)*math.log(3)
    MSE = (5**2+4**2+2**2)/3.
    
    assert log_likelihood == BNMF.quality('loglikelihood',burnin,thinning)
    assert AIC == BNMF.quality('AIC',burnin,thinning)
    assert BIC == BNMF.quality('BIC',burnin,thinning)
    assert MSE == BNMF.quality('MSE',burnin,thinning)
    with pytest.raises(AssertionError) as error:
        BNMF.quality('FAIL',burnin,thinning)
    assert str(error.value) == "Unrecognised metric for model quality: FAIL."