"""
Gibbs sampler for BNMTF, with ARD.

We expect the following arguments:
- R, the matrix
- M, the mask matrix indicating observed values (1) and unobserved ones (0)
- K, the maximum number of row clusters
- L, the maximum number of column clusters
- priors = { 'alphaR' : alphaR, 'betaR' : betaR, 'alpha0' : alpha0, 'beta0' : beta0 }
    a dictionary defining the priors over tau, F, S, G.
    
Initialisation can be done by running the initialise() function, with argument
init. Options (default 'kmeans'):
- 'random' : draw values randomly from the model assumption distributions
- 'exp' : use the expectation of the model assumption distributions
- 'kmeans': use K-means clustering on the rows of R (+0.2) to initialise F, and similarly on columns of R for G. Initialise the rest randomly.

Usage of class:
    BNMTF = bnmtf_ard_gibbs(R,M,K,L,priors)
    BNMTF.initisalise(init)
    BNMTF.run(iterations)
Or:
    BNMTF = bnmf_gibbs(R,M,K,L,priors)
    BNMTF.train(iterations,init)
    
The model stores the following fields after running:
- all_performances: dictionary {'MSE', 'R^2', 'Rp'} storing a list of performances across iterations.
- all_times: the time it took to complete the first i iterations
- All draws are stored in lists: all_tau, all_F, all_lambdaF, all_G, all_lambdaG, all_S, all_lambdaS.

The expectation can be computed by specifying a burn-in and thinning rate, and using:
    BNMTF.approx_expectation(burn_in,thinning)
    
We can test the performance of our model on a test dataset, specifying our test set with a mask M. 
    performance = BNMTF.predict(M_pred,burn_in,thinning)
This gives a dictionary of performances,
    performance = { 'MSE', 'R^2', 'Rp' }
    
Finally, we can return the goodness of fit of the data using the quality(metric) function:
- metric = 'loglikelihood' -> return p(D|theta)
         = 'BIC'           -> return Bayesian Information Criterion
         = 'AIC'           -> return Afaike Information Criterion
         = 'MSE'           -> return Mean Square Error
(we want to maximise the loglikelihood, and minimise the others)
"""

import sys
sys.path.append("/home/tab43/Documents/Projects/libraries/")

from BNMTF_ARD.code.kmeans.kmeans import KMeans
from BNMTF_ARD.code.distributions.exponential import exponential_draw
from BNMTF_ARD.code.distributions.gamma import gamma_draw, gamma_expectation
from BNMTF_ARD.code.distributions.truncated_normal import TN_draw
from BNMTF_ARD.code.distributions.truncated_normal_vector import TN_vector_draw

import numpy, itertools, math, time
   
PERFORMANCE_METRICS = ['MSE','R^2','Rp']
QUALITY_MEASURES = ['loglikelihood','BIC','AIC','MSE','ELBO']

class bnmtf_ard_gibbs:
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
        self.initialise(init)
        return self.run(iterations)


    def initialise(self,init='kmeans'):
        ''' Initialise the matrices F, S, G, and lambda parameters.
            Options are:
            - 'random' : draw values randomly from the model assumption distributions
            - 'exp' : use the expectation of the model assumption distributions
            - 'kmeans': use K-means clustering on the rows of R (+0.2) to initialise F, and similarly on columns of R for G. Initialise the rest randomly.
        '''
        
        assert init in ['random','exp','kmeans'], "Unknown initialisation option: %s. Should be 'random', 'exp', or 'kmeans." % init
        
        ''' First initialise the lambdaS[k,l], lambdaF[k], lambdaG[l]. '''
        self.lambdaS = self.alpha0 / self.beta0 * numpy.ones((self.K,self.L)) 
        self.lambdaF = self.alpha0 / self.beta0 * numpy.ones(self.K)
        self.lambdaG = self.alpha0 / self.beta0 * numpy.ones(self.L)
        if init == 'random' or init == 'kmeans':
            for k,l in itertools.product(xrange(0,self.K),xrange(0,self.L)):
                self.lambdaS[k,l] = gamma_draw(self.alpha0,self.beta0)
            for k in xrange(0,self.K):
                self.lambdaF[k] = gamma_draw(self.alpha0,self.beta0)
            for l in xrange(0,self.L):
                self.lambdaG[l] = gamma_draw(self.alpha0,self.beta0)
        
        ''' Then initialise S '''
        self.S = 1. / self.lambdaS
        if init == 'random' or init == 'kmeans':
            for k,l in itertools.product(xrange(0,self.K),xrange(0,self.L)): 
                self.S[k,l] = exponential_draw(self.lambdaS[k,l])
                
        ''' Then initialise the F[i,k], G[j,l]. ''' 
        self.F = numpy.ones((self.I,self.K)) / self.lambdaF
        self.G = numpy.ones((self.J,self.L)) / self.lambdaG
        if init == 'random':
            for i,k in itertools.product(xrange(0,self.I),xrange(0,self.K)): 
                    self.F[i,k] = exponential_draw(self.lambdaF[k])
            for j,l in itertools.product(xrange(0,self.J),xrange(0,self.L)): 
                    self.G[j,l] = exponential_draw(self.lambdaG[l])
        elif init == 'kmeans':
            print "Initialising F using KMeans."
            kmeansF = KMeans(self.R,self.M,self.K)
            kmeansF.initialise()
            kmeansF.cluster()
            self.F = kmeansF.clustering_results + 0.2            
            
            print "Initialising G using KMeans."
            kmeansG = KMeans(self.R.T,self.M.T,self.L)   
            kmeansG.initialise()
            kmeansG.cluster()
            self.G = kmeansG.clustering_results + 0.2

        ''' Finally, initialise tau using the updates from the model.'''
        if init == 'exp':
            self.tau = gamma_expectation(self.alphaR_s(),self.betaR_s())
        elif init == 'random' or init == 'kmeans':
            self.tau = gamma_draw(self.alphaR_s(),self.betaR_s())
        

    def run(self,iterations):
        ''' Run the Gibbs sampler for the specified number of iterations. '''
        self.all_F = numpy.zeros((iterations,self.I,self.K))  
        self.all_S = numpy.zeros((iterations,self.K,self.L))   
        self.all_G = numpy.zeros((iterations,self.J,self.L))  
        self.all_tau = numpy.zeros(iterations)
        self.all_lambdaF = numpy.zeros((iterations,self.K))  
        self.all_lambdaS = numpy.zeros((iterations,self.K,self.L))   
        self.all_lambdaG = numpy.zeros((iterations,self.L)) 
        
        self.all_times = [] # to plot performance against time     
        self.all_performances = {} # for plotting convergence of metrics
        for metric in PERFORMANCE_METRICS:
            self.all_performances[metric] = []
        
        time_start = time.time()
        for it in range(0,iterations):  
            ''' Draw new values for lambdaF and F. '''
            for k in range(0,self.K):
                alphaFk, betaFk = self.alphaF(k), self.betaF(k)
                self.lambdaF[k] = gamma_draw(alphaFk,betaFk)

                tauFk = self.tauF(k)
                muFk = self.muF(tauFk,k)
                self.F[:,k] = TN_vector_draw(muFk,tauFk)
                
            ''' Draw new values for lambdaG and G. '''
            for l in range(0,self.L):
                alphaGl, betaGl = self.alphaG(l), self.betaG(l)
                self.lambdaG[l] = gamma_draw(alphaGl,betaGl)

                tauGl = self.tauG(l)
                muGl = self.muG(tauGl,l)
                self.G[:,l] = TN_vector_draw(muGl,tauGl)
                
            ''' Draw new values for lambdaS and S. '''
            for k,l in itertools.product(xrange(0,self.K),xrange(0,self.L)):
                alphaSkl, betaSkl = self.alphaS(k,l), self.betaS(k,l)
                self.lambdaS[k,l] = gamma_draw(alphaSkl,betaSkl)

                tauSkl = self.tauS(k,l)
                muSkl = self.muS(tauSkl,k,l)
                self.S[k,l] = TN_draw(muSkl,tauSkl)
                
            ''' Draw a new value for tau. '''
            self.tau = gamma_draw(self.alphaR_s(),self.betaR_s())
            
            ''' Compute the performances of this iteration's draws, and print them. '''
            perf = self.predict_while_running()
            for metric in PERFORMANCE_METRICS:
                self.all_performances[metric].append(perf[metric])
                
            print "Iteration %s. MSE: %s. R^2: %s. Rp: %s." % (it+1,perf['MSE'],perf['R^2'],perf['Rp'])
        
            ''' Store the new draws, and the time it took. '''
            self.all_F[it], self.all_lambdaF[it] = numpy.copy(self.F), numpy.copy(self.lambdaF)
            self.all_S[it], self.all_lambdaS[it] = numpy.copy(self.S), numpy.copy(self.lambdaS)
            self.all_G[it], self.all_lambdaG[it] = numpy.copy(self.G), numpy.copy(self.lambdaG)
            self.all_tau[it] = self.tau
            
            time_iteration = time.time()
            self.all_times.append(time_iteration-time_start)            
            
        return
        

    def triple_dot(self,M1,M2,M3):
        ''' Compute the dot product of three matrices. 
            Let shape(M2) = (K,L). If K > L it is more efficient to do (M1*M2)*M3, and if L > K then we do M1*(M2*M3).'''
        K, L = M2.shape
        if K > L:
            return numpy.dot(numpy.dot(M1,M2),M3)
        else:
            return numpy.dot(M1,numpy.dot(M2,M3))
        
        
    ''' Compute the parameters for the posterior distributions of tau. '''
    def alphaR_s(self):   
        return self.alphaR + self.size_Omega/2.0
    
    def betaR_s(self):   
        return self.betaR + 0.5*(self.M*(self.R-self.triple_dot(self.F,self.S,self.G.T))**2).sum()
        
    ''' Compute the parameters for the posterior distributions of lambdaF[k]. '''
    def alphaF(self,k):   
        return self.alpha0 + self.I
        
    def betaF(self,k):   
        return self.beta0 + self.F[:,k].sum()
        
    ''' Compute the parameters for the posterior distributions of lambdaG[l]. '''
    def alphaG(self,l):   
        return self.alpha0 + self.J
        
    def betaG(self,l):   
        return self.beta0 + self.G[:,l].sum()
        
    ''' Compute the parameters for the posterior distributions of lambdaS[k,l]. '''
    def alphaS(self,k,l):   
        return self.alpha0 + 1.
        
    def betaS(self,k,l):   
        return self.beta0 + self.S[k,l]
        
    ''' Compute the parameters for the posterior distributions of F[:,k]. '''
    def tauF(self,k):       
        return self.tau * ( self.M * numpy.dot(self.S[k],self.G.T)**2 ).sum(axis=1)
        
    def muF(self,tauFk,k):
        return 1./tauFk * (-self.lambdaF[k] + self.tau*(self.M * ( (self.R-self.triple_dot(self.F,self.S,self.G.T)+numpy.outer(self.F[:,k],numpy.dot(self.S[k],self.G.T)))*numpy.dot(self.S[k],self.G.T) )).sum(axis=1)) 
        
    ''' Compute the parameters for the posterior distributions of S[k,l]. '''
    def tauS(self,k,l):       
        return self.tau * ( self.M * numpy.outer(self.F[:,k]**2,self.G[:,l]**2) ).sum()
        
    def muS(self,tauSkl,k,l):
        return 1./tauSkl * (-self.lambdaS[k,l] + self.tau*(self.M * ( (self.R-self.triple_dot(self.F,self.S,self.G.T)+self.S[k,l]*numpy.outer(self.F[:,k],self.G[:,l]))*numpy.outer(self.F[:,k],self.G[:,l]) )).sum()) 
        
    ''' Compute the parameters for the posterior distributions of G[:,l]. '''
    def tauG(self,l):       
        return self.tau * ( self.M.T * numpy.dot(self.F,self.S[:,l])**2 ).T.sum(axis=0)
        
    def muG(self,tauGl,l):
        return 1./tauGl * (-self.lambdaG[l] + self.tau*(self.M * ( (self.R-self.triple_dot(self.F,self.S,self.G.T)+numpy.outer(numpy.dot(self.F,self.S[:,l]),self.G[:,l])).T * numpy.dot(self.F,self.S[:,l]) ).T).sum(axis=0)) 
        

    def approx_expectation(self,burn_in,thinning):
        ''' Compute our approximation of the expectation (average across iterations).
            Throw away the first <burn_in> samples, and then use every <thinning>th after.
            Return (exp_tau, exp_F, exp_S, exp_G, exp_lambdaF, exp_lambdaS, exp_lambdaG). '''
        indices = range(burn_in,len(self.all_F),thinning)
        assert len(indices) > 0, 'Want to approximate expectation but no samples selected! Burn in is %s, thinning is %s.' % (burn_in,thinning)
        
        exp_tau = sum([self.all_tau[i] for i in indices]) / float(len(indices))
        
        exp_F = numpy.array([self.all_F[i] for i in indices]).sum(axis=0) / float(len(indices))      
        exp_S = numpy.array([self.all_S[i] for i in indices]).sum(axis=0) / float(len(indices))   
        exp_G = numpy.array([self.all_G[i] for i in indices]).sum(axis=0) / float(len(indices))  
        
        exp_lambdaF = numpy.array([self.all_lambdaF[i] for i in indices]).sum(axis=0) / float(len(indices))      
        exp_lambdaS = numpy.array([self.all_lambdaS[i] for i in indices]).sum(axis=0) / float(len(indices))   
        exp_lambdaG = numpy.array([self.all_lambdaG[i] for i in indices]).sum(axis=0) / float(len(indices))  
        
        return (exp_tau, exp_F, exp_S, exp_G, exp_lambdaF, exp_lambdaS, exp_lambdaG)


    def predict(self,M_pred,burn_in,thinning):
        ''' Compute the expectation of F, S, G, and use it to predict missing values. '''
        (_,exp_F,exp_S,exp_G,_,_,_) = self.approx_expectation(burn_in,thinning)
        R_pred = self.triple_dot(exp_F,exp_S,exp_G.T)
        MSE = self.compute_MSE(M_pred,self.R,R_pred)
        R2 = self.compute_R2(M_pred,self.R,R_pred)    
        Rp = self.compute_Rp(M_pred,self.R,R_pred)        
        return {'MSE':MSE,'R^2':R2,'Rp':Rp}
        
        
    def predict_while_running(self):
        ''' Compute the prediction performance of the current draws. '''
        R_pred = self.triple_dot(self.F,self.S,self.G.T)
        MSE = self.compute_MSE(self.M,self.R,R_pred)
        R2 = self.compute_R2(self.M,self.R,R_pred)    
        Rp = self.compute_Rp(self.M,self.R,R_pred)        
        return {'MSE':MSE,'R^2':R2,'Rp':Rp}
        
        
    ''' Functions for computing MSE, R^2 (coefficient of determination), Rp (Pearson correlation). '''
    def compute_MSE(self,M,R,R_pred):
        print R
        print R_pred
        print M * (R-R_pred)
        print M * (R-R_pred)**2
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
        
        
    ''' Functions for model selection, measuring the goodness of fit vs model complexity '''
    def quality(self,metric,burn_in,thinning):
        assert metric in QUALITY_MEASURES, 'Unrecognised metric for model quality: %s.' % metric
        
        (exp_tau,exp_F,exp_S,exp_G,_,_,_) = self.approx_expectation(burn_in,thinning)
        log_likelihood = self.log_likelihood(exp_F,exp_S,exp_G,exp_tau)
        
        if metric == 'loglikelihood':
            return log_likelihood
        elif metric == 'BIC':
            # -2*loglikelihood + (no. free parameters * log(no data points))
            return - 2 * log_likelihood + (self.I*self.K+self.K*self.L+self.J*self.L) * math.log(self.size_Omega)
        elif metric == 'AIC':
            # -2*loglikelihood + 2*no. free parameters
            return - 2 * log_likelihood + 2 * (self.I*self.K+self.K*self.L+self.J*self.L)
        elif metric == 'MSE':
            R_pred = self.triple_dot(exp_F,exp_S,exp_G.T)
            return self.compute_MSE(self.M,self.R,R_pred)
        elif metric == 'ELBO':
            return 0.
        
    def log_likelihood(self,expF,expS,expG,exptau):
        # Return the likelihood of the data given the trained model's parameters
        explogtau = math.log(exptau)
        return self.size_Omega / 2. * ( explogtau - math.log(2*math.pi) ) \
             - exptau / 2. * (self.M*( self.R - self.triple_dot(expF,expS,expG.T) )**2).sum()