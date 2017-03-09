"""
Variational Bayesian inference for BNMTF, with ARD.
This is model 3: 
- ARD per factor on F, G.
- No ARD on S, regular Exp prior.

We expect the following arguments:
- R, the matrix
- M, the mask matrix indicating observed values (1) and unobserved ones (0)
- K, the maximum number of row clusters
- L, the maximum number of column clusters
- priors = { 'alphaR' : alphaR, 'betaR' : betaR, 'alpha0' : alpha0, 'beta0' : beta0, 'lambdaS' : lambdaS }
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

from BNMTF_ARD.code.distributions.exponential import exponential_draw
from bnmtf_ard_vb_1 import bnmtf_ard_vb_1

import numpy, itertools, math, scipy, random
   
PERFORMANCE_METRICS = ['MSE','R^2','Rp']
QUALITY_MEASURES = ['loglikelihood','BIC','AIC','MSE','ELBO']
RANDOMISE_UPDATES = True

class bnmtf_ard_vb_3(bnmtf_ard_vb_1):
    def __init__(self,R,M,K,L,priors):
        super(bnmtf_ard_vb_3,self).__init__(R,M,K,L,priors)
        self.lambdaS = priors['lambdaS']
           
           
    def initialise(self,init='kmeans'):
        ''' Use the initialisation of Model 1, but then overwrite it for S. '''
        super(bnmtf_ard_vb_3,self).initialise(init)
        
        self.alphaS, self.betaS, self.exp_lambdaS, self.exp_loglambdaS = None, None, None, None        
        self.muS, self.tauS = 1. / self.lambdaS * numpy.ones((self.K,self.L)), numpy.ones((self.K,self.L))
        if init == 'random' or init == 'kmeans':
            for k,l in itertools.product(xrange(0,self.K),xrange(0,self.L)): 
                self.muS[k,l] = exponential_draw(self.lambdaS)
                
        self.exp_S, self.var_S = numpy.zeros((self.K,self.L)), numpy.zeros((self.K,self.L))
        for k,l in itertools.product(xrange(0,self.K),xrange(0,self.L)):
            self.update_exp_S(k,l)
            
        self.update_tau()
        self.update_exp_tau()
        
        
    """ Methods for updating all of F, S, G, tau. """
    def run_S(self):
        ''' Update S. '''
        indices = itertools.product(xrange(0,self.K),xrange(0,self.L))     
        if RANDOMISE_UPDATES: random.shuffle(indices)   
        
        for k,l in indices:
            self.update_S(k,l)
            self.update_exp_S(k,l)   
        
        
    def elbo(self):
        ''' Compute the ELBO. '''
        self.p_R = self.size_Omega / 2. * ( self.exp_logtau - math.log(2*math.pi) ) - self.exp_tau / 2. * self.exp_square_diff()
        
        self.p_tau = self.alphaR * math.log(self.betaR) - scipy.special.gammaln(self.alphaR) \
                     + (self.alphaR - 1.)*self.exp_logtau - self.betaR * self.exp_tau
        self.p_F = self.I * self.exp_loglambdaF.sum() - ( self.exp_lambdaF * self.exp_F ).sum()
        self.p_G = self.J * self.exp_loglambdaG.sum() - ( self.exp_lambdaG * self.exp_G ).sum()
        self.p_S = self.K*self.L*math.log(self.lambdaS) - ( self.lambdaS * self.exp_S ).sum()
        self.p_lambdaF = self.K * self.alpha0 * math.log(self.beta0) - self.K * scipy.special.gammaln(self.alpha0) \
                         + (self.alpha0 - 1.)*self.exp_loglambdaF.sum() - self.beta0 * self.exp_lambdaF.sum()
        self.p_lambdaG = self.L * self.alpha0 * math.log(self.beta0) - self.L * scipy.special.gammaln(self.alpha0) \
                         + (self.alpha0 - 1.)*self.exp_loglambdaG.sum() - self.beta0 * self.exp_lambdaG.sum()
        
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
        
        return self.p_R + self.p_tau + self.p_F + self.p_G + self.p_S + self.p_lambdaF + self.p_lambdaG \
               - self.q_tau - self.q_F - self.q_G - self.q_S - self.q_lambdaF - self.q_lambdaG


    ''' Update the parameters and expectation for S. '''
    def update_S(self,k,l):       
        self.tauS[k,l] = self.exp_tau*(self.M*( numpy.outer( self.var_F[:,k]+self.exp_F[:,k]**2 , self.var_G[:,l]+self.exp_G[:,l]**2 ) )).sum()
        
        diff_term = (self.M * ( (self.R-self.triple_dot(self.exp_F,self.exp_S,self.exp_G.T)+self.exp_S[k,l]*numpy.outer(self.exp_F[:,k],self.exp_G[:,l]) ) * numpy.outer(self.exp_F[:,k],self.exp_G[:,l]) )).sum()
        cov_term_G = (self.M * numpy.outer( self.exp_F[:,k] * ( numpy.dot(self.exp_F,self.exp_S[:,l]) - self.exp_F[:,k]*self.exp_S[k,l] ), self.var_G[:,l] )).sum()
        cov_term_F = (self.M * numpy.outer( self.var_F[:,k], self.exp_G[:,l]*(numpy.dot(self.exp_S[k],self.exp_G.T) - self.exp_S[k,l]*self.exp_G[:,l]) )).sum()        
        self.muS[k,l] = 1./self.tauS[k,l] * (
            - self.lambdaS 
            + self.exp_tau * diff_term
            - self.exp_tau * cov_term_G
            - self.exp_tau * cov_term_F
        ) 