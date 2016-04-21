"""
Variational Bayesian inference for BNMTF, with ARD.

We expect the following arguments:
- R, the matrix
- M, the mask matrix indicating observed values (1) and unobserved ones (0)
- K, the maximum number of row clusters
- L, the maximum number of column clusters
- priors = { 'alphaR' : alphaR, 'betaR' : betaR, 'alpha0' : alpha0, 'beta0' : beta0 }
    a dictionary defining the priors over tau, F, S, G.
    
INITIALISATION
Argument init of initialise() function controls initialisation of muF, muS, muG (default 'kmeans'):
- 'random' : draw values randomly from the model assumption distributions
- 'exp' : use the expectation of the model assumption distributions
- 'kmeans': use K-means clustering on the rows of R (+0.2) to initialise F, and similarly on columns of R for G. Initialise S randomly.
The tau parameters for F, S, G, get initialised to 1.
The alpha and beta parameters to the lambda get initialised to alpha0 + (I or J or 1), and beta0 + (I or J or 1). So a peak at lambda = 1.
The tau parameter for R gets initialised using the model updates.

USAGE
    BNMTF = bnmtf_ard_vb(R,M,K,L,priors)
    BNMTF.initisalise(init)
    BNMTF.run(iterations)
Or:
    BNMTF = bnmf_gibbs(R,M,K,L,priors)
    BNMTF.train(iterations,init)
    
OUTPUT
The model stores the following fields after running:
- all_performances: dictionary {'MSE', 'R^2', 'Rp'} storing a list of performances across iterations.
- all_times: the time it took to complete the first i iterations
- All expectations of tau, lambdaF and lambdaG are stored in lists: all_exp_tau, all_exp_lambdaF, all_exp_lambdaG.

PERFORMANCE
We can test the performance of our model on a test dataset, specifying our test set with a mask M. 
    performance = BNMTF.predict(M_pred)
This gives a dictionary of performances,
    performance = { 'MSE', 'R^2', 'Rp' }
    
QUALITY
Finally, we can return the goodness of fit of the data using the quality(metric) function:
- metric = 'loglikelihood' -> return p(D|theta)
         = 'BIC'           -> return Bayesian Information Criterion
         = 'AIC'           -> return Afaike Information Criterion
         = 'MSE'           -> return Mean Square Error
         = 'ELBO'          -> return Evidence Lower BOund
(we want to maximise the loglikelihood, and minimise the others)
"""

import sys
sys.path.append("/home/tab43/Documents/Projects/libraries/")

from BNMTF_ARD.code.kmeans.kmeans import KMeans
from BNMTF_ARD.code.distributions.gamma import gamma_expectation, gamma_expectation_log
from BNMTF_ARD.code.distributions.truncated_normal import TN_expectation, TN_variance
from BNMTF_ARD.code.distributions.truncated_normal_vector import TN_vector_expectation, TN_vector_variance
from BNMTF_ARD.code.distributions.exponential import exponential_draw

import numpy, itertools, math, scipy, time
   
PERFORMANCE_METRICS = ['MSE','R^2','Rp']
QUALITY_MEASURES = ['loglikelihood','BIC','AIC','MSE','ELBO']

class bnmtf_ard_vb:
    def __init__(self,R,M,K,L,priors):
        ''' Initialise the class by checking whether all the parameters are correct. '''
        self.R = numpy.array(R,dtype=float)
        self.M = numpy.array(M,dtype=float)
        self.K = K
        self.L = L
        
        assert len(self.R.shape) == 2, "Input matrix R is not a two-dimensional array, " \
            "but instead %s-dimensional." % len(self.R.shape)
        assert self.R.shape == self.M.shape, "Input matrix R is not of the same size as " \
            "the indicator matrix M: %s and %s respectively." % (self.R.shape,self.M.shape)
            
        (self.I,self.J) = self.R.shape
        self.size_Omega = self.M.sum()
        self.check_empty_rows_columns()      
        
        self.alphaR, self.betaR = float(priors['alphaR']), float(priors['betaR'])
        self.alpha0, self.beta0 = float(priors['alpha0']), float(priors['beta0'])
           
        
    def check_empty_rows_columns(self):
        ''' Check whether each row and column of R has at least one observed entry (1 in M). '''
        sums_columns = self.M.sum(axis=0)
        sums_rows = self.M.sum(axis=1)
                    
        # Assert none of the rows or columns are entirely unknown values
        for i,c in enumerate(sums_rows):
            assert c != 0, "Fully unobserved row in R, row %s." % i
        for j,c in enumerate(sums_columns):
            assert c != 0, "Fully unobserved column in R, column %s." % j


    def train(self,iterations,init='kmeans'):
        ''' Initialise and run the sampler. '''
        self.initialise(init=init)
        return self.run(iterations)


    def initialise(self,init='kmeans'):
        ''' Initialise the matrices F, S, G, and lambda parameters.
            Options are:
            - 'random' : draw values randomly from the model assumption distributions
            - 'exp' : use the expectation of the model assumption distributions
            - 'kmeans': use K-means clustering on the rows of R (+0.2) to initialise F, and similarly on columns of R for G. Initialise S randomly.
            The tau parameters for F, S, G, get initialised to 1.
            The alpha and beta parameters to the lambda get initialised to alpha0 + (I or J or 1), and beta0 + (I or J or 1). So a peak at lambda = 1.
            The tau parameter for R gets initialised using the model updates.
        '''
        
        assert init in ['random','exp','kmeans'], "Unknown initialisation option: %s. Should be 'random', 'exp', or 'kmeans." % init
        
        ''' First initialise the lambdaS[k,l], lambdaF[k], lambdaG[l]. '''
        self.alphaF, self.betaF = (self.alpha0 * numpy.ones(self.K) + self.I), (self.beta0 * numpy.ones(self.K) + self.I)
        self.alphaG, self.betaG = (self.alpha0 * numpy.ones(self.L) + self.J), (self.beta0 * numpy.ones(self.L) + self.J)
        self.alphaS, self.betaS = (self.alpha0 *  numpy.ones((self.K,self.L)) + 1.), (self.beta0 * numpy.ones((self.K,self.L)) + 1.)
        
        ''' Initialise the expectations and variances for lambdaS, lambdaF, lambdaG. '''
        self.exp_lambdaF, self.exp_loglambdaF = numpy.zeros(self.K), numpy.zeros(self.K)
        self.exp_lambdaG, self.exp_loglambdaG = numpy.zeros(self.L), numpy.zeros(self.L)
        self.exp_lambdaS, self.exp_loglambdaS = numpy.zeros((self.K,self.L)), numpy.zeros((self.K,self.L))
        
        for k in xrange(0,self.K):
            self.update_exp_lambdaF(k)
        for l in xrange(0,self.L):
            self.update_exp_lambdaG(l)
        for k,l in itertools.product(xrange(0,self.K),xrange(0,self.L)):
            self.update_exp_lambdaS(k,l)
        
        ''' Then initialise muS, tauS. '''
        self.muS, self.tauS = 1. / self.exp_lambdaS, numpy.ones((self.K,self.L))
        if init == 'random' or init == 'kmeans':
            for k,l in itertools.product(xrange(0,self.K),xrange(0,self.L)): 
                self.muS[k,l] = exponential_draw(self.exp_lambdaS[k,l])
                
        ''' Then initialise muF/G, tauF/G. ''' 
        self.muF, self.tauF = numpy.array([1. / self.exp_lambdaF for i in range(0,self.I)]), numpy.ones((self.I,self.K))
        self.muG, self.tauG = numpy.array([1. / self.exp_lambdaG for j in range(0,self.J)]), numpy.ones((self.J,self.L))
        if init == 'random':
            for i,k in itertools.product(xrange(0,self.I),xrange(0,self.K)):
                self.muF[i,k] = exponential_draw(self.exp_lambdaF[k])
            for j,l in itertools.product(xrange(0,self.J),xrange(0,self.L)):
                self.muG[j,l] = exponential_draw(self.exp_lambdaG[l])
        elif init == 'kmeans':
            print "Initialising F using KMeans."
            kmeansF = KMeans(self.R,self.M,self.K)
            kmeansF.initialise()
            kmeansF.cluster()
            self.muF = kmeansF.clustering_results           
            
            print "Initialising G using KMeans."
            kmeansG = KMeans(self.R.T,self.M.T,self.L)   
            kmeansG.initialise()
            kmeansG.cluster()
            self.muG = kmeansG.clustering_results
            
        ''' Initialise the expectations and variances of F, S, G. '''
        self.exp_F, self.var_F = numpy.zeros((self.I,self.K)), numpy.zeros((self.I,self.K))
        self.exp_S, self.var_S = numpy.zeros((self.K,self.L)), numpy.zeros((self.K,self.L))
        self.exp_G, self.var_G = numpy.zeros((self.J,self.L)), numpy.zeros((self.J,self.L))
        
        for k in range(0,self.K):
            self.update_exp_F(k)
        for k,l in itertools.product(xrange(0,self.K),xrange(0,self.L)):
            self.update_exp_S(k,l)
        for l in range(0,self.L):
            self.update_exp_G(l)
            
        ''' Finally, initialise tau using the updates. '''
        self.update_tau()
        self.update_exp_tau()
        
        
    def run(self,iterations):
        ''' Run the variational inference for the specified number of iterations. '''
        self.all_exp_tau = numpy.zeros(iterations)
        self.all_exp_lambdaF = numpy.zeros((iterations,self.K))   
        self.all_exp_lambdaG = numpy.zeros((iterations,self.L)) 
        
        self.all_times = [] # to plot performance against time     
        self.all_performances = {} # for plotting convergence of metrics
        for metric in PERFORMANCE_METRICS:
            self.all_performances[metric] = []
        
        time_start = time.time()
        for it in range(0,iterations): 
            ''' Update lambdaF and F. '''
            for k in range(0,self.K):
                self.update_lambdaF(k)
                self.update_exp_lambdaF(k)
                self.update_F(k)
                self.update_exp_F(k)
                
            ''' Update lambdaG and G. '''
            for l in range(0,self.L):
                self.update_lambdaG(l)
                self.update_exp_lambdaG(l)
                self.update_G(l)
                self.update_exp_G(l)
                
            ''' Update lambdaS and S. '''
            for k,l in itertools.product(xrange(0,self.K),xrange(0,self.L)):
                self.update_lambdaS(k,l)
                self.update_exp_lambdaS(k,l)
                self.update_S(k,l)
                self.update_exp_S(k,l)
                
            ''' Update tau. '''
            self.update_tau()
            self.update_exp_tau()
            
            ''' Compute the performances of this iteration's draws, and print them. '''
            perf, elbo = self.predict(self.M), self.elbo()
            for metric in PERFORMANCE_METRICS:
                self.all_performances[metric].append(perf[metric])
                
            print "Iteration %s. ELBO: %s. MSE: %s. R^2: %s. Rp: %s." % (it+1,elbo,perf['MSE'],perf['R^2'],perf['Rp'])
                        
            ''' Store the new draws, and the time it took. '''
            self.all_exp_lambdaF[it] = numpy.copy(self.exp_lambdaF)
            self.all_exp_lambdaG[it] = numpy.copy(self.exp_lambdaG)
            self.all_exp_tau[it] = self.exp_tau
            
            time_iteration = time.time()
            self.all_times.append(time_iteration-time_start)            
            
        
    def elbo(self):
        ''' Compute the ELBO. '''
        self.p_R = self.size_Omega / 2. * ( self.exp_logtau - math.log(2*math.pi) ) - self.exp_tau / 2. * self.exp_square_diff()
        
        self.p_tau = self.alphaR * math.log(self.betaR) - scipy.special.gammaln(self.alphaR) \
                     + (self.alphaR - 1.)*self.exp_logtau - self.betaR * self.exp_tau
        self.p_F = self.I * self.exp_loglambdaF.sum() - ( self.exp_lambdaF * self.exp_F ).sum()
        self.p_G = self.J * self.exp_loglambdaG.sum() - ( self.exp_lambdaG * self.exp_G ).sum()
        self.p_S = self.exp_loglambdaS.sum() - ( self.exp_lambdaS * self.exp_S ).sum()
        self.p_lambdaF = self.K * self.alpha0 * math.log(self.beta0) - self.K * scipy.special.gammaln(self.alpha0) \
                         + (self.alpha0 - 1.)*self.exp_loglambdaF.sum() - self.beta0 * self.exp_lambdaF.sum()
        self.p_lambdaG = self.L * self.alpha0 * math.log(self.beta0) - self.L * scipy.special.gammaln(self.alpha0) \
                         + (self.alpha0 - 1.)*self.exp_loglambdaG.sum() - self.beta0 * self.exp_lambdaG.sum()
        self.p_lambdaS = self.K * self.L * self.alpha0 * math.log(self.beta0) - self.K * self.L * scipy.special.gammaln(self.alpha0) \
                         + (self.alpha0 - 1.)*self.exp_loglambdaS.sum() - self.beta0 * self.exp_lambdaS.sum()
        
        self.q_tau = self.alphaR_s * math.log(self.betaR_s) - scipy.special.gammaln(self.alphaR_s) \
                     + (self.alphaR_s - 1.)*self.exp_logtau - self.betaR_s * self.exp_tau
        self.q_F = 1./2.*numpy.log(self.tauF).sum() - self.I*self.K/2.*math.log(2*math.pi) \
                   - numpy.log(0.5*scipy.special.erfc(-self.muF*numpy.sqrt(self.tauF)/math.sqrt(2))).sum() \
                   - ( self.tauF / 2. * ( self.var_F + (self.exp_F - self.muF)**2 ) ).sum()
        self.q_G = .5*numpy.log(self.tauG).sum() - self.J*self.L/2.*math.log(2*math.pi) \
                   - numpy.log(0.5*scipy.special.erfc(-self.muG*numpy.sqrt(self.tauG)/math.sqrt(2))).sum() \
                   - ( self.tauG / 2. * ( self.var_G + (self.exp_G - self.muG)**2 ) ).sum()      
        self.q_S = .5*numpy.log(self.tauS).sum() - self.K*self.L/2.*math.log(2*math.pi) \
                   - numpy.log(0.5*scipy.special.erfc(-self.muS*numpy.sqrt(self.tauS)/math.sqrt(2))).sum() \
                   - ( self.tauS / 2. * ( self.var_S + (self.exp_S - self.muS)**2 ) ).sum()
        self.q_lambdaF = ( self.alphaF * scipy.log(self.betaF) - scipy.special.gammaln(self.alphaF) + \
                           (self.alphaF - 1.)*self.exp_loglambdaF - self.betaF * self.exp_lambdaF ).sum()
        self.q_lambdaG = ( self.alphaG * scipy.log(self.betaG) - scipy.special.gammaln(self.alphaG) + \
                           (self.alphaG - 1.)*self.exp_loglambdaG - self.betaG * self.exp_lambdaG ).sum()
        self.q_lambdaS = ( self.alphaS * scipy.log(self.betaS) - scipy.special.gammaln(self.alphaS) + \
                           (self.alphaS - 1.)*self.exp_loglambdaS - self.betaS * self.exp_lambdaS ).sum()
        
        return self.p_R + self.p_tau + self.p_F + self.p_G + self.p_S + self.p_lambdaF + self.p_lambdaG + self.p_lambdaS \
               - self.q_tau - self.q_F - self.q_G - self.q_S - self.q_lambdaF - self.q_lambdaG - self.q_lambdaS


    def triple_dot(self,M1,M2,M3):
        ''' Compute the dot product of three matrices. 
            Let shape(M2) = (K,L). If K > L it is more efficient to do (M1*M2)*M3, and if L > K then we do M1*(M2*M3).'''
        K, L = M2.shape
        if K > L:
            return numpy.dot(numpy.dot(M1,M2),M3)
        else:
            return numpy.dot(M1,numpy.dot(M2,M3))
        
        
    ''' Update the parameters and expectation for tau. '''
    def update_tau(self):   
        self.alphaR_s = self.alphaR + self.size_Omega/2.0
        self.betaR_s = self.betaR + 0.5*self.exp_square_diff()
        
    def update_exp_tau(self):
        self.exp_tau = gamma_expectation(self.alphaR_s,self.betaR_s)
        self.exp_logtau = gamma_expectation_log(self.alphaR_s,self.betaR_s)
        
    def exp_square_diff(self): # Compute: sum_Omega E_q(F,S,G) [ ( Rij - Fi S Gj )^2 ]
        return (self.M*( self.R - self.triple_dot(self.exp_F,self.exp_S,self.exp_G.T) )**2).sum() + \
               (self.M*( self.triple_dot(self.var_F+self.exp_F**2, self.var_S+self.exp_S**2, (self.var_G+self.exp_G**2).T ) - self.triple_dot(self.exp_F**2,self.exp_S**2,(self.exp_G**2).T) )).sum() + \
               (self.M*( numpy.dot(self.var_F, ( numpy.dot(self.exp_S,self.exp_G.T)**2 - numpy.dot(self.exp_S**2,self.exp_G.T**2) ) ) )).sum() + \
               (self.M*( numpy.dot( numpy.dot(self.exp_F,self.exp_S)**2 - numpy.dot(self.exp_F**2,self.exp_S**2), self.var_G.T ) )).sum()
    
    ''' Update the parameters and expectation for lambdaF. '''
    def update_lambdaF(self,k):
        self.alphaF[k] = self.alpha0 + self.I
        self.betaF[k] = self.beta0 + self.exp_F[:,k].sum()
        
    def update_exp_lambdaF(self,k):
        self.exp_lambdaF[k] = gamma_expectation(self.alphaF[k],self.betaF[k])
        self.exp_loglambdaF[k] = gamma_expectation_log(self.alphaF[k],self.betaF[k])
    
    ''' Update the parameters and expectation for lambdaG. '''
    def update_lambdaG(self,l):
        self.alphaG[l] = self.alpha0 + self.J
        self.betaG[l] = self.beta0 + self.exp_G[:,l].sum()
        
    def update_exp_lambdaG(self,l):
        self.exp_lambdaG[l] = gamma_expectation(self.alphaG[l],self.betaG[l])
        self.exp_loglambdaG[l] = gamma_expectation_log(self.alphaG[l],self.betaG[l])
    
    ''' Update the parameters and expectation for lambdaS. '''
    def update_lambdaS(self,k,l):
        self.alphaS[k,l] = self.alpha0 + 1.
        self.betaS[k,l] = self.beta0 + self.exp_S[k,l]
        
    def update_exp_lambdaS(self,k,l):
        self.exp_lambdaS[k,l] = gamma_expectation(self.alphaS[k,l],self.betaS[k,l])
        self.exp_loglambdaS[k,l] = gamma_expectation_log(self.alphaS[k,l],self.betaS[k,l])
    
    ''' Update the parameters and expectation for F. '''
    def update_F(self,k):  
        var_SkG = numpy.dot( self.var_S[k]+self.exp_S[k]**2 , (self.var_G+self.exp_G**2).T ) - numpy.dot( self.exp_S[k]**2 , (self.exp_G**2).T ) # Vector of size J
        self.tauF[:,k] = self.exp_tau * numpy.dot( var_SkG + ( numpy.dot(self.exp_S[k],self.exp_G.T) )**2 , self.M.T ) 
        
        diff_term = (self.M * ( (self.R-self.triple_dot(self.exp_F,self.exp_S,self.exp_G.T)+numpy.outer(self.exp_F[:,k],numpy.dot(self.exp_S[k],self.exp_G.T)) ) * numpy.dot(self.exp_S[k],self.exp_G.T) )).sum(axis=1)        
        cov_term = ( self.M * ( ( numpy.dot(self.exp_S[k]*numpy.dot(self.exp_F,self.exp_S), self.var_G.T) - numpy.outer(self.exp_F[:,k], numpy.dot( self.exp_S[k]**2, self.var_G.T )) ) ) ).sum(axis=1)
        self.muF[:,k] = 1./self.tauF[:,k] * (
            - self.exp_lambdaF[k]
            + self.exp_tau * diff_term
            - self.exp_tau * cov_term
        ) 
        
    def update_exp_F(self,k):
        self.exp_F[:,k] = TN_vector_expectation(self.muF[:,k],self.tauF[:,k])
        self.var_F[:,k] = TN_vector_variance(self.muF[:,k],self.tauF[:,k])
        
    ''' Update the parameters and expectation for S. '''
    def update_S(self,k,l):       
        self.tauS[k,l] = self.exp_tau*(self.M*( numpy.outer( self.var_F[:,k]+self.exp_F[:,k]**2 , self.var_G[:,l]+self.exp_G[:,l]**2 ) )).sum()
        
        diff_term = (self.M * ( (self.R-self.triple_dot(self.exp_F,self.exp_S,self.exp_G.T)+self.exp_S[k,l]*numpy.outer(self.exp_F[:,k],self.exp_G[:,l]) ) * numpy.outer(self.exp_F[:,k],self.exp_G[:,l]) )).sum()
        cov_term_G = (self.M * numpy.outer( self.exp_F[:,k] * ( numpy.dot(self.exp_F,self.exp_S[:,l]) - self.exp_F[:,k]*self.exp_S[k,l] ), self.var_G[:,l] )).sum()
        cov_term_F = (self.M * numpy.outer( self.var_F[:,k], self.exp_G[:,l]*(numpy.dot(self.exp_S[k],self.exp_G.T) - self.exp_S[k,l]*self.exp_G[:,l]) )).sum()        
        self.muS[k,l] = 1./self.tauS[k,l] * (
            - self.exp_lambdaS[k,l] 
            + self.exp_tau * diff_term
            - self.exp_tau * cov_term_G
            - self.exp_tau * cov_term_F
        ) 
        
    def update_exp_S(self,k,l):
        self.exp_S[k,l] = TN_expectation(self.muS[k,l],self.tauS[k,l])
        self.var_S[k,l] = TN_variance(self.muS[k,l],self.tauS[k,l])
        
    ''' Update the parameters and expectation for G. '''
    def update_G(self,l):  
        var_FSl = numpy.dot( self.var_F+self.exp_F**2 , self.var_S[:,l]+self.exp_S[:,l]**2 ) - numpy.dot( self.exp_F**2 , self.exp_S[:,l]**2 ) # Vector of size I
        self.tauG[:,l] = self.exp_tau * numpy.dot( ( var_FSl + ( numpy.dot(self.exp_F,self.exp_S[:,l]) )**2 ).T, self.M) #sum over i, so columns        
        
        diff_term = (self.M * ( (self.R-self.triple_dot(self.exp_F,self.exp_S,self.exp_G.T)+numpy.outer(numpy.dot(self.exp_F,self.exp_S[:,l]), self.exp_G[:,l]) ).T * numpy.dot(self.exp_F,self.exp_S[:,l]) ).T ).sum(axis=0)
        cov_term = (self.M * ( numpy.dot(self.var_F, (self.exp_S[:,l]*numpy.dot(self.exp_S,self.exp_G.T).T).T) - numpy.outer(numpy.dot(self.var_F,self.exp_S[:,l]**2), self.exp_G[:,l]) )).sum(axis=0)
        self.muG[:,l] = 1./self.tauG[:,l] * (
            - self.exp_lambdaG[l] 
            + self.exp_tau * diff_term
            - self.exp_tau * cov_term
        )
        
    def update_exp_G(self,l):
        self.exp_G[:,l] = TN_vector_expectation(self.muG[:,l],self.tauG[:,l])
        self.var_G[:,l] = TN_vector_variance(self.muG[:,l],self.tauG[:,l])


    def predict(self,M_pred):
        ''' Compute the performance of predicting missing values. '''
        R_pred = self.triple_dot(self.exp_F,self.exp_S,self.exp_G.T)
        MSE = self.compute_MSE(M_pred,self.R,R_pred)
        R2 = self.compute_R2(M_pred,self.R,R_pred)    
        Rp = self.compute_Rp(M_pred,self.R,R_pred)        
        return {'MSE':MSE,'R^2':R2,'Rp':Rp}
        
        
    ''' Functions for computing MSE, R^2 (coefficient of determination), Rp (Pearson correlation). '''
    def compute_MSE(self,M,R,R_pred):
        return (M * (R-R_pred)**2).sum() / float(M.sum())
        
    def compute_R2(self,M,R,R_pred):
        mean = (M*R).sum() / float(M.sum())
        SS_total = float((M*(R-mean)**2).sum())
        SS_res = float((M*(R-R_pred)**2).sum())
        return 1. - SS_res / SS_total if SS_total != 0. else numpy.inf
        
    def compute_Rp(self,M,R,R_pred):
        mean_real = (M*R).sum() / float(M.sum())
        mean_pred = (M*R_pred).sum() / float(M.sum())
        covariance = (M*(R-mean_real)*(R_pred-mean_pred)).sum()
        variance_real = (M*(R-mean_real)**2).sum()
        variance_pred = (M*(R_pred-mean_pred)**2).sum()
        return covariance / float(math.sqrt(variance_real)*math.sqrt(variance_pred))
        
        
    ''' Functions for model selection, measuring the goodness of fit vs model complexity. '''
    def quality(self,metric):
        assert metric in ['loglikelihood','BIC','AIC','MSE','ELBO'], 'Unrecognised metric for model quality: %s.' % metric
        log_likelihood = self.log_likelihood()
        if metric == 'loglikelihood':
            return log_likelihood
        elif metric == 'BIC':
            # -2*loglikelihood + (no. free parameters * log(no data points))
            return - 2 * log_likelihood + (self.I*self.K+self.K + 2*self.K*self.L + self.J*self.L+self.L) * math.log(self.size_Omega)
        elif metric == 'AIC':
            # -2*loglikelihood + 2*no. free parameters
            return - 2 * log_likelihood + 2 * (self.I*self.K+self.K + 2*self.K*self.L + self.J*self.L+self.L)
        elif metric == 'MSE':
            R_pred = self.triple_dot(self.exp_F,self.exp_S,self.exp_G.T)
            return self.compute_MSE(self.M,self.R,R_pred)
        elif metric == 'ELBO':
            return self.elbo()
        
    def log_likelihood(self):
        ''' Return the likelihood of the data given the trained model's parameters. '''
        return self.size_Omega / 2. * ( self.exp_logtau - math.log(2*math.pi) ) \
             - self.exp_tau / 2. * (self.M*( self.R - self.triple_dot(self.exp_F,self.exp_S,self.exp_G.T) )**2).sum()