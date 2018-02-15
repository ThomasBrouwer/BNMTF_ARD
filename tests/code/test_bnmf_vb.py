"""
Tests for the BNMF Variational Bayes algorithm, with optimised matrix operation updates.
"""

import sys, os
project_location = os.path.dirname(__file__)+"/../../../"
sys.path.append(project_location)

import numpy, math, pytest, itertools
from BNMTF_ARD.code.models.bnmf_vb import bnmf_vb


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
        bnmf_vb(R1,M,K,ARD,hyperparams)
    assert str(error.value) == "Input matrix R is not a two-dimensional array, but instead 1-dimensional."
    
    R2 = numpy.ones((4,3,2))
    with pytest.raises(AssertionError) as error:
        bnmf_vb(R2,M,K,ARD,hyperparams)
    assert str(error.value) == "Input matrix R is not a two-dimensional array, but instead 3-dimensional."
    
    # Test R is the same shape as M
    R3 = numpy.ones((3,2))
    with pytest.raises(AssertionError) as error:
        bnmf_vb(R3,M,K,ARD,hyperparams)
    assert str(error.value) == "Input matrix R is not of the same size as the indicator matrix M: (3, 2) and (2, 3) respectively."
    
    # Similarly for lambdaU, lambdaV
    R4 = numpy.ones((2,3))
    lambdaU = numpy.ones((2+1,1))
    hyperparams = { 'alphatau':alphatau, 'betatau':betatau, 'lambdaU':lambdaU, 'lambdaV':lambdaV }
    with pytest.raises(AssertionError) as error:
        bnmf_vb(R4,M,K,False,hyperparams)
    assert str(error.value) == "Prior matrix lambdaU has the wrong shape: (3, 1) instead of (2, 1)."
    
    lambdaU = numpy.ones((2,1))
    lambdaV = numpy.ones((3+1,1))
    hyperparams = { 'alphatau':alphatau, 'betatau':betatau, 'lambdaU':lambdaU, 'lambdaV':lambdaV }
    with pytest.raises(AssertionError) as error:
        bnmf_vb(R4,M,K,False,hyperparams)
    assert str(error.value) == "Prior matrix lambdaV has the wrong shape: (4, 1) instead of (3, 1)."
    
    # Test getting an exception if a row or column is entirely unknown
    M1 = [[1,1,1],[0,0,0]]
    M2 = [[1,1,0],[1,0,0]]
    hyperparams = { 'alphatau':alphatau, 'betatau':betatau, 'alpha0':alpha0, 'beta0':beta0 }
    
    with pytest.raises(AssertionError) as error:
        bnmf_vb(R4,M1,K,ARD,hyperparams)
    assert str(error.value) == "Fully unobserved row in R, row 1."
    with pytest.raises(AssertionError) as error:
        bnmf_vb(R4,M2,K,ARD,hyperparams)
    assert str(error.value) == "Fully unobserved column in R, column 2."
    
    # Finally, a successful case
    I,J,K = 3,2,2
    R5 = 2*numpy.ones((I,J))
    M = numpy.ones((I,J))
    hyperparams = { 'alphatau':alphatau, 'betatau':betatau, 'alpha0':alpha0, 'beta0':beta0 }
    BNMF = bnmf_vb(R5,M,K,ARD,hyperparams)
    
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
    BNMF = bnmf_vb(R5,M,K,False,hyperparams)
    
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
    
    lambdaU = 2*numpy.ones((I,K))
    lambdaV = 3*numpy.ones((J,K))
    alphatau, betatau = 3, 1
    hyperparams = { 'alphatau':alphatau, 'betatau':betatau, 'lambdaU':lambdaU, 'lambdaV':lambdaV }

    # Initialisation with expectation
    init = 'exp'
    BNMF = bnmf_vb(R,M,K,False,hyperparams)
    BNMF.initialise(init)
    
    assert BNMF.alpha_s == alphatau + 15./2.
    assert BNMF.beta_s == betatau + BNMF.exp_square_diff()/2.
    
    for i,k in itertools.product(range(I),range(K)):
        assert BNMF.tau_U[i,k] == 1.
        assert BNMF.mu_U[i,k] == 1./lambdaU[i,k]
    for j,k in itertools.product(range(J),range(K)):
        assert BNMF.tau_V[j,k] == 1.
        assert BNMF.mu_V[j,k] == 1./lambdaV[j,k]
        
    assert BNMF.exp_tau == (alphatau + 15./2.) / (betatau + BNMF.exp_square_diff()/2.)
    
    for i,k in itertools.product(range(I),range(K)):
        assert abs(BNMF.exp_U[i,k] - (0.5 + 0.352065 / (1-0.3085))) < 0.0001
    for j,k in itertools.product(range(J),range(K)):
        assert abs(BNMF.exp_V[j,k] - (1./3. + 0.377383 / (1-0.3694))) < 0.0001
        
    # With ARD
    init_UV = 'exp'
    alphatau, betatau = 3, 1
    alpha0, beta0 = 6, 2
    hyperparams = { 'alphatau':alphatau, 'betatau':betatau, 'alpha0':alpha0, 'beta0':beta0 }

    BNMF = bnmf_vb(R,M,K,True,hyperparams)
    BNMF.initialise(init_UV)
    for k in range(K):
        assert BNMF.exp_lambdak[k] == alpha0 / float(beta0)
        assert BNMF.exp_loglambdak[k] == 1.0129704878718551
    for i,k in itertools.product(range(I),range(K)):
        assert BNMF.tau_U[i,k] == 1.
        assert BNMF.mu_U[i,k] == 1./(alpha0 / float(beta0))
    for j,k in itertools.product(range(J),range(K)):
        assert BNMF.tau_V[j,k] == 1.
        assert BNMF.mu_V[j,k] == 1./(alpha0 / float(beta0))
    
        
""" Test computing the ELBO. """
def test_elbo():
    I,J,K = 5,3,2
    R = numpy.ones((I,J))
    M = numpy.ones((I,J))
    M[0,0], M[2,2], M[3,1] = 0, 0, 0 # size Omega = 12
    
    lambdaU = 2*numpy.ones((I,K))
    lambdaV = 3*numpy.ones((J,K))
    alphatau, betatau = 3, 1
    hyperparams = { 'alphatau':alphatau, 'betatau':betatau, 'lambdaU':lambdaU, 'lambdaV':lambdaV }

    exp_U = 5*numpy.ones((I,K))
    exp_V = 6*numpy.ones((J,K))
    var_U = 11*numpy.ones((I,K))
    var_V = 12*numpy.ones((J,K))
    exp_tau = 8.
    exp_logtau = 9.
    
    mu_U = 14*numpy.ones((I,K))
    mu_V = 15*numpy.ones((J,K))
    tau_U = numpy.ones((I,K))/100.
    tau_V = numpy.ones((J,K))/101.
    alpha_s = 20.
    beta_s = 21.
    
    # expU * expV = [[60]]
    # (R - expU*expV)^2 = 12*59^2 = 41772
    # Var[U*V] = 12*K*((11+5^2)*(12+6^2)-5^2*6^2) = 12*2*828 = 19872
    
    # -muU*sqrt(tauU) = -14*math.sqrt(100) = -1.4
    # -muV*sqrt(tauV) = -15*math.sqrt(101) = -1.4925557853149838
    # cdf(-1.4) = 0.080756659233771066
    # cdf(-1.4925557853149838) = 0.067776752211548219
    
    ELBO = 12./2.*(exp_logtau - math.log(2*math.pi)) - 8./2.*(41772+19872) \
         + 5*2*(math.log(2.) - 2.*5.) + 3*2*(math.log(3.) - 3.*6.) \
         + 3.*numpy.log(1.) - numpy.log(math.gamma(3.)) + 2.*9. - 1.*8. \
         - 20.*numpy.log(21.) + numpy.log(math.gamma(20.)) - 19.*9. + 21.*8. \
         - 0.5*5*2*math.log(1./100.) + 0.5*5*2*math.log(2*math.pi) + 5*2*math.log(1.-0.080756659233771066) \
         + 0.5*5*2*1./100.*(11.+81.) \
         - 0.5*3*2*math.log(1./101.) + 0.5*3*2*math.log(2*math.pi) + 3*2*math.log(1.-0.067776752211548219) \
         + 0.5*3*2*1./101.*(12.+81.)
         
    BNMF = bnmf_vb(R,M,K,False,hyperparams)
    BNMF.exp_U = exp_U
    BNMF.exp_V = exp_V
    BNMF.var_U = var_U
    BNMF.var_V = var_V
    BNMF.exp_tau = exp_tau
    BNMF.exp_logtau = exp_logtau
    BNMF.mu_U = mu_U
    BNMF.mu_V = mu_V
    BNMF.tau_U = tau_U
    BNMF.tau_V = tau_V
    BNMF.alpha_s = alpha_s
    BNMF.beta_s = beta_s
    assert BNMF.elbo() == ELBO
    
        
""" Test updating parameters U, V, tau """          
I,J,K = 5,3,2
R = numpy.ones((I,J))
M = numpy.ones((I,J))
M[0,0], M[2,2], M[3,1] = 0, 0, 0

lambdaU = 2*numpy.ones((I,K))
lambdaV = 3*numpy.ones((J,K))
alphatau, betatau = 3, 1
alpha0, beta0 = 6, 2
hyperparams = { 'alphatau':alphatau, 'betatau':betatau, 'alpha0':alpha0, 'beta0':beta0, 'lambdaU':lambdaU, 'lambdaV':lambdaV }

def test_exp_square_diff():
    BNMF = bnmf_vb(R,M,K,False,hyperparams)
    BNMF.exp_U = 1./lambdaU #[[1./2.]]
    BNMF.exp_V = 1./lambdaV #[[1./3.]]
    BNMF.var_U = numpy.ones((I,K))*2 #[[2.]]
    BNMF.var_V = numpy.ones((J,K))*3 #[[3.]]
    # expU * expV.T = [[1./3.]]. (varU+expU^2)=2.25, (varV+expV^2)=3.+1./9.
    exp_square_diff = 172.66666666666666 #12.*(4./9.) + 12.*(2*(2.25*(3.+1./9.)-0.25/9.)) 
    assert BNMF.exp_square_diff() == exp_square_diff

def test_update_tau():
    BNMF = bnmf_vb(R,M,K,False,hyperparams)
    BNMF.exp_U = 1./lambdaU #[[1./2.]]
    BNMF.exp_V = 1./lambdaV #[[1./3.]]
    BNMF.var_U = numpy.ones((I,K))*2 #[[2.]]
    BNMF.var_V = numpy.ones((J,K))*3 #[[3.]]
    BNMF.update_tau()
    assert BNMF.alpha_s == alphatau + 12./2.
    assert BNMF.beta_s == betatau + 172.66666666666666/2.
    
def test_update_lambdak():
    BNMF = bnmf_vb(R,M,K,True,hyperparams)
    BNMF.alphak_s, BNMF.betak_s = numpy.zeros(K), numpy.zeros(K)
    BNMF.exp_U, BNMF.exp_V = 1./lambdaU, 1./lambdaV # [[1./2.]], [[1./3.]]
    for k in range(K):
        BNMF.update_lambdak(k)
        assert BNMF.alphak_s[k] == alpha0 + I + J
        assert BNMF.betak_s[k] == (beta0 + I*1./2. + J*1./3.)
    
def test_update_U():
    for k in range(0,K):
        BNMF = bnmf_vb(R,M,K,False,hyperparams)
        BNMF.mu_U = numpy.zeros((I,K))
        BNMF.tau_U = numpy.zeros((I,K))
        BNMF.exp_U = 1./lambdaU #[[1./2.]]
        BNMF.exp_V = 1./lambdaV #[[1./3.]]
        BNMF.var_U = numpy.ones((I,K))*2 #[[2.]]
        BNMF.var_V = numpy.ones((J,K))*3 #[[3.]]
        BNMF.exp_tau = 3.
        BNMF.update_U(k)
        for i in range(0,I):
            assert BNMF.tau_U[i,k] == 3. * (M[i] * ( BNMF.exp_V[:,k]*BNMF.exp_V[:,k] + BNMF.var_V[:,k] )).sum()
            assert BNMF.mu_U[i,k] == (1./(3. * (M[i] * ( BNMF.exp_V[:,k]*BNMF.exp_V[:,k] + BNMF.var_V[:,k] )).sum())) * \
                                    ( -2. + BNMF.exp_tau * (M[i]*( (BNMF.R[i] - numpy.dot(BNMF.exp_U[i],BNMF.exp_V.T) + BNMF.exp_U[i,k]*BNMF.exp_V[:,k])*BNMF.exp_V[:,k] )).sum() )

def test_update_V():
    for k in range(0,K):
        BNMF = bnmf_vb(R,M,K,False,hyperparams)
        BNMF.mu_V = numpy.zeros((J,K))
        BNMF.tau_V = numpy.zeros((J,K))
        BNMF.exp_U = 1./lambdaU #[[1./2.]]
        BNMF.exp_V = 1./lambdaV #[[1./3.]]
        BNMF.var_U = numpy.ones((I,K))*2 #[[2.]]
        BNMF.var_V = numpy.ones((J,K))*3 #[[3.]]
        BNMF.exp_tau = 3.
        BNMF.update_V(k)
        for j in range(0,J):
            assert BNMF.tau_V[j,k] == 3. * (M[:,j] * ( BNMF.exp_U[:,k]*BNMF.exp_U[:,k] + BNMF.var_U[:,k] )).sum()
            assert BNMF.mu_V[j,k] == (1./(3. * (M[:,j] * ( BNMF.exp_U[:,k]*BNMF.exp_U[:,k] + BNMF.var_U[:,k] )).sum())) * \
                                    ( -3. + BNMF.exp_tau * (M[:,j]*( (BNMF.R[:,j] - numpy.dot(BNMF.exp_U,BNMF.exp_V[j]) + BNMF.exp_U[:,k]*BNMF.exp_V[j,k])*BNMF.exp_U[:,k] )).sum() )


""" Test computing expectation, variance U, V, tau """   
def test_update_exp_tau():
    BNMF = bnmf_vb(R,M,K,False,hyperparams)
    BNMF.initialise()  
    assert abs(BNMF.exp_tau - (3+12./2.)/(1+35.4113198623/2.)) < 0.000000000001
    assert abs(BNMF.exp_logtau - (2.1406414779556 - math.log(1+35.4113198623/2.))) < 0.000000000001
   
def test_update_exp_lambdak():
    BNMF = bnmf_vb(R,M,K,True,hyperparams)
    BNMF.initialise(init_UV='exp')  
    for k in range(K):
        assert abs(BNMF.exp_lambdak[k] - (alpha0+I+J)/(beta0+(I+J)*beta0/float(alpha0))) < 0.000000000001
        assert BNMF.exp_loglambdak[k] == 1.0129704878718551
    
def test_update_exp_U():
    for k in range(0,K):
        BNMF = bnmf_vb(R,M,K,False,hyperparams)
        BNMF.initialise()
        BNMF.tau_U = 4*numpy.ones((I,K)) # muU = [[0.5]], tauU = [[4.]]
        BNMF.update_exp_U(k) #-mu*sqrt(tau) = -0.5*2 = -1. lambda(1) = 0.241971 / (1-0.1587) = 0.2876155949126352. gamma = 0.37033832534958433
        for i in range(0,I):        
            assert abs(BNMF.exp_U[i,k] - (0.5 + 1./2. * 0.2876155949126352)) < 0.00001
            assert abs(BNMF.var_U[i,k] - 1./4.*(1.-0.37033832534958433)) < 0.00001

def test_update_exp_V():
    for k in range(0,K):
        BNMF = bnmf_vb(R,M,K,False,hyperparams)
        BNMF.initialise() 
        BNMF.tau_V = 4*numpy.ones((J,K)) # muV = [[1./3.]], tauV = [[4.]]
        BNMF.update_exp_V(k) #-mu*sqrt(tau) = -2./3., lambda(..) = 0.319448 / (1-0.2525) = 0.4273551839464883, gamma = 
        for j in range(0,J):        
            assert abs(BNMF.exp_V[j,k] - (1./3. + 1./2. * 0.4273551839464883)) < 0.00001
            assert abs(BNMF.var_V[j,k] - 1./4.*(1. - 0.4675359092102624)) < 0.00001
    

""" Test two iterations of run(), and that all values have changed. """
def test_run():
    I,J,K = 10,5,2
    R = numpy.ones((I,J))
    M = numpy.ones((I,J))
    M[0,0], M[2,2], M[3,1] = 0, 0, 0
    R[0,1], R[0,2] = 2., 3.
    
    alphatau, betatau = 3, 1
    alpha0, beta0 = 6, 2
    hyperparams = { 'alphatau':alphatau, 'betatau':betatau, 'alpha0':alpha0, 'beta0':beta0 }
    
    iterations = 15
    
    BNMF = bnmf_vb(R,M,K,True,hyperparams)
    BNMF.initialise()
    BNMF.run(iterations)
    
    for k in range(K):
        assert BNMF.alphak_s[k] != alpha0
        assert BNMF.betak_s[k] != beta0
        assert BNMF.exp_lambdak[k] != alpha0 / float(beta0)
    for i,k in itertools.product(range(I),range(K)):
        assert BNMF.mu_U[i,k] != 1./(alpha0 / float(beta0))
        assert BNMF.tau_U[i,k] != 1.
        assert BNMF.exp_U[i,k] != numpy.inf and not math.isnan(BNMF.exp_U[i,k])
        assert BNMF.tau_U[i,k] != numpy.inf and not math.isnan(BNMF.tau_U[i,k])
    for j,k in itertools.product(range(J),range(K)):
        assert BNMF.mu_V[j,k] != 1./(alpha0 / float(beta0))
        assert BNMF.tau_V[j,k] != 1.
        assert BNMF.exp_V[j,k] != numpy.inf and not math.isnan(BNMF.exp_V[j,k])
        assert BNMF.tau_V[j,k] != numpy.inf and not math.isnan(BNMF.tau_V[j,k])
    assert BNMF.alpha_s != alphatau
    assert BNMF.beta_s != betatau
    assert BNMF.exp_tau != numpy.inf and not math.isnan(BNMF.exp_tau)
    assert BNMF.exp_logtau != numpy.inf and not math.isnan(BNMF.exp_logtau)
    

""" Test computing the performance of the predictions using the expectations """
def test_predict():
    (I,J,K) = (5,3,2)
    R = numpy.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15]],dtype=float)
    M = numpy.ones((I,J))
    K = 3
    lambdaU = 2*numpy.ones((I,K))
    lambdaV = 3*numpy.ones((J,K))
    alphatau, betatau = 3, 1
    hyperparams = { 'alphatau':alphatau, 'betatau':betatau, 'lambdaU':lambdaU, 'lambdaV':lambdaV }
    
    expU = numpy.array([[125.,126.],[126.,126.],[126.,126.],[126.,126.],[126.,126.]])
    expV = numpy.array([[84.,84.],[84.,84.],[84.,84.]])
    
    M_test = numpy.array([[0,0,1],[0,1,0],[0,0,0],[1,1,0],[0,0,0]]) #R->3,5,10,11, P_pred->21084,21168,21168,21168
    MSE = (444408561. + 447872569. + 447660964. + 447618649) / 4.
    R2 = 1. - (444408561. + 447872569. + 447660964. + 447618649) / (4.25**2+2.25**2+2.75**2+3.75**2) #mean=7.25
    Rp = 357. / ( math.sqrt(44.75) * math.sqrt(5292.) ) #mean=7.25,var=44.75, mean_pred=21147,var_pred=5292, corr=(-4.25*-63 + -2.25*21 + 2.75*21 + 3.75*21)
    
    BNMF = bnmf_vb(R,M,K,False,hyperparams)
    BNMF.exp_U = expU
    BNMF.exp_V = expV
    performances = BNMF.predict(M_test)
    
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
    
    BNMF = bnmf_vb(R,M,K,False,hyperparams)
    
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
    
    BNMF = bnmf_vb(R,M,K,False,hyperparams)
    BNMF.exp_U = numpy.ones((I,K))
    BNMF.exp_V = 2*numpy.ones((J,K))
    BNMF.exp_logtau = 5.
    BNMF.exp_tau = 3.
    # expU*expV.T = [[6.]]
    
    log_likelihood = 3./2.*(5.-math.log(2*math.pi)) - 3./2. * (5**2 + 4**2 + 2**2)
    AIC = -2*log_likelihood + 2*(2*3+2*3+1)
    BIC = -2*log_likelihood + (2*3+2*3+1)*math.log(3)
    MSE = (5**2+4**2+2**2)/3.
    
    assert log_likelihood == BNMF.quality('loglikelihood')
    assert AIC == BNMF.quality('AIC')
    assert BIC == BNMF.quality('BIC')
    assert MSE == BNMF.quality('MSE')
    with pytest.raises(AssertionError) as error:
        BNMF.quality('FAIL')
    assert str(error.value) == "Unrecognised metric for model quality: FAIL."