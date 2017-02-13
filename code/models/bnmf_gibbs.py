"""
Gibbs sampler for non-negative matrix factorisation, with ARD.

We expect the following arguments:
- R, the matrix.
- M, the mask matrix indicating observed values (1) and unobserved ones (0).
- K, the number of latent factors.
- ARD, a boolean indicating whether we use ARD in this model or not.
- hyperparameters = { 'alphatau', 'betatau', 'alpha0', 'beta0', 'lambdaU', 'lambdaV' },
    alphatau, betatau - non-negative reals defining prior over noise parameter tau.
    alpha0, beta0     - if using the ARD, non-negative reals defining prior over ARD lambda.
    lambdaU, lambdaV  - if not using the ARD, nonnegative reals defining prior over U and V
    
The random variables are initialised as follows:
    U, V: expectation ('exp') or random ('random')
    tau: using updates, and then random draw
    lambda: expectation
We initialise the values of U and V according to the given argument 'init_UV'. 

Usage of class:
    BNMF = bnmf_gibbs(R,M,K,ARD,priors)
    BNMF.initisalise(init_UV)
    BNMF.run(iterations)
Or:
    BNMF = bnmf_gibbs(R,M,K,ARD,priors)
    BNMF.train(init_UV,iterations)
    
The draws for all iterations are stored in: all_U, all_V, all_lambdak, all_tau.
    
The expectation can be computed by specifying a burn-in and thinning rate, and using:
    BNMF.approx_expectation(burn_in,thinning)
This returns a tuple (exp_U, exp_V, exp_tau, exp_lambda).

We can test the performance of our model on a test dataset, specifying our test set with a mask M. 
    performance = BNMF.predict(M_pred,burn_in,thinning)
This gives a dictionary of performances,
    performance = { 'MSE', 'R^2', 'Rp' }    
    
The performances of all iterations are stored in BNMF.all_performances, which 
is a dictionary from 'MSE', 'R^2', or 'Rp' to a list of performances.
    
Finally, we can return the goodness of fit of the data using the quality(metric) function:
- metric = 'loglikelihood' -> return p(D|theta)
         = 'BIC'           -> return Bayesian Information Criterion
         = 'AIC'           -> return Afaike Information Criterion
         = 'MSE'           -> return Mean Square Error
         = 'ELBO'          -> N/A (only for VB)
"""

from distributions.exponential import exponential_draw
from distributions.gamma import gamma_draw
from distributions.truncated_normal_vector import TN_vector_draw

import numpy, itertools, math, time

ALL_METRICS = ['MSE','R^2','Rp']
ALL_QUALITY = ['loglikelihood','BIC','AIC','MSE','ELBO']
OPTIONS_INIT_UV = ['random', 'exp']

class bnmf_gibbs_optimised:
    def __init__(self,R,M,K,ARD,priors):
        ''' Set up the class and do some checks on the values passed. '''
        self.R = numpy.array(R,dtype=float)
        self.M = numpy.array(M,dtype=float)
        self.K = K
        self.ARD = ARD
        
        assert len(self.R.shape) == 2, "Input matrix R is not a two-dimensional array, " \
            "but instead %s-dimensional." % len(self.R.shape)
        assert self.R.shape == self.M.shape, "Input matrix R is not of the same size as " \
            "the indicator matrix M: %s and %s respectively." % (self.R.shape,self.M.shape)
            
        (self.I,self.J) = self.R.shape
        self.size_Omega = self.M.sum()
        self.check_empty_rows_columns()      
        
        self.alphatau, self.betatau = float(priors['alphatau']), float(priors['betatau'])
        if self.ARD:
            self.alpha0, self.beta0 = float(priors['alpha0']), float(priors['beta0'])
        else:
            self.lambdaU, self.lambdaV = float(priors['lambdaU']), float(priors['lambdaV'])
            # Make lambdaU/V into a numpy array if they are an integer
            if self.lambdaU.shape == ():
                self.lambdaU = self.lambdaU * numpy.ones((self.I,self.K))
            if self.lambdaV.shape == ():
                self.lambdaV = self.lambdaV * numpy.ones((self.J,self.K))
        
            assert self.lambdaU.shape == (self.I,self.K), "Prior matrix lambdaU has the wrong shape: %s instead of (%s, %s)." % (self.lambdaU.shape,self.I,self.K)
            assert self.lambdaV.shape == (self.J,self.K), "Prior matrix lambdaV has the wrong shape: %s instead of (%s, %s)." % (self.lambdaV.shape,self.J,self.K)
                
            
    def check_empty_rows_columns(self):
        ''' Raise an exception if an entire row or column is empty. '''
        sums_columns = self.M.sum(axis=0)
        sums_rows = self.M.sum(axis=1)
                    
        # Assert none of the rows or columns are entirely unknown values
        for i,c in enumerate(sums_rows):
            assert c != 0, "Fully unobserved row in R, row %s." % i
        for j,c in enumerate(sums_columns):
            assert c != 0, "Fully unobserved column in R, column %s." % j


    def train(self,init_UV,iterations):
        ''' Initialise and run the sampler. '''
        self.initialise(init_UV=init_UV)
        self.run(iterations)


    def initialise(self,init_UV='random'):
        ''' Initialise U, V, tau, and lambda (if ARD). '''
        assert init_UV in OPTIONS_INIT_UV, "Unknown initialisation option: %s. Should be in %s." % (init_UV, OPTIONS_INIT_UV)
        
        self.U = numpy.zeros((self.I,self.K))
        self.V = numpy.zeros((self.J,self.K))
        self.lambdak = numpy.zeros(self.K)  
        
        # Initialise lambdak
        if self.ARD:
            for k in range(self.K):
                self.lambdak[k] = self.alpha0 / self.beta0
        
        # Initialise U, V
        for i,k in itertools.product(range(self.I),range(self.K)):    
            hyperparam = self.lambdak[k] if self.ARD else self.lambdaU[i,k]
            self.U[i,k] = exponential_draw(hyperparam) if init_UV == 'random' else 1.0/hyperparam
        for j,k in itertools.product(range(self.J),range(self.K)):
            hyperparam = self.lambdak[k] if self.ARD else self.lambdaV[j,k]
            self.V[j,k] = exponential_draw(hyperparam) if init_UV == 'random' else 1.0/hyperparam
        
        # Initialise tau
        self.tau = gamma_draw(self.alpha_s(),self.beta_s())


    def run(self,iterations):
        ''' Run the Gibbs sampler. '''
        self.all_U = numpy.zeros((iterations,self.I,self.K))  
        self.all_V = numpy.zeros((iterations,self.J,self.K))   
        self.all_tau = numpy.zeros(iterations) 
        self.all_lambdak = numpy.zeros((iterations,self.K))
        self.all_times = [] # to plot performance against time
        
        self.all_performances = {} # for plotting convergence of metrics
        for metric in ALL_METRICS:
            self.all_performances[metric] = []
        
        time_start = time.time()
        for it in range(iterations): 
            # Update lambdak
            if self.ARD:
                for k in range(self.K):
                    self.lambdak[k] = gamma_draw(self.alphak_s(k),self.betak_s(k))
            
            # Update U
            for k in range(0,self.K):   
                tauUk = self.tauU(k)
                muUk = self.muU(tauUk,k)
                self.U[:,k] = TN_vector_draw(muUk,tauUk)
                
            # Update V
            for k in range(0,self.K):
                tauVk = self.tauV(k)
                muVk = self.muV(tauVk,k)
                self.V[:,k] = TN_vector_draw(muVk,tauVk)
                
            # Update tau
            self.tau = gamma_draw(self.alpha_s(),self.beta_s())
            
            # Store draws
            self.all_U[it], self.all_V[it], self.all_tau[it] = numpy.copy(self.U), numpy.copy(self.V), self.tau
            if self.ARD:
                self.all_lambdak[it] = numpy.copy(self.lambdak)
            
            # Store and print performances
            perf = self.predict_while_running()
            for metric in ALL_METRICS:
                self.all_performances[metric].append(perf[metric])
                
            print "Iteration %s. MSE: %s. R^2: %s. Rp: %s." % (it+1,perf['MSE'],perf['R^2'],perf['Rp'])
            
            # Store time taken for iteration
            time_iteration = time.time()
            self.all_times.append(time_iteration-time_start)            
        
        
    ''' Compute the parameters for the distributions we sample from. '''
    def alpha_s(self):   
        ''' alpha* for tau. '''
        return self.alphatau + self.size_Omega/2.0
    
    def beta_s(self):   
        ''' beta* for tau. '''
        return self.betatau + 0.5*(self.M*(self.R-numpy.dot(self.U,self.V.T))**2).sum()
        
    def alphak_s(self,k):   
        ''' alphak* for lambdak. '''
        return self.alpha0 + self.I + self.J
    
    def betak_s(self,k):   
        ''' betaK* for lambdak. '''
        return self.beta0 + self.U[:,k].sum() + self.V[:,k].sum()
        
    def tauU(self,k):
        ''' tauUk for Uk. '''
        return self.tau * ( self.M * self.V[:,k]**2 ).sum(axis=1)
        
    def muU(self,tauUk,k):
        ''' muUk for Uk. '''
        lamb = self.lambdak[k] if self.ARD else self.lambdaU[:,k]
        return 1./tauUk * (-lamb + self.tau*(self.M * ( (self.R-numpy.dot(self.U,self.V.T)+numpy.outer(self.U[:,k],self.V[:,k]))*self.V[:,k] )).sum(axis=1)) 
        
    def tauV(self,k):
        ''' tauVk for Vk. '''
        return self.tau*(self.M.T*self.U[:,k]**2).T.sum(axis=0)
        
    def muV(self,tauVk,k):
        ''' muVk for Vk. '''
        lamb = self.lambdak[k] if self.ARD else self.lambdaV[:,k]
        return 1./tauVk * (-lamb + self.tau*(self.M.T * ( (self.R-numpy.dot(self.U,self.V.T)+numpy.outer(self.U[:,k],self.V[:,k])).T*self.U[:,k] )).T.sum(axis=0)) 


    def approx_expectation(self,burn_in,thinning):
        ''' Return our expectation of U, V, tau, lambdak. '''
        indices = range(burn_in,len(self.all_U),thinning)
        exp_U = numpy.array([self.all_U[i] for i in indices]).sum(axis=0) / float(len(indices))      
        exp_V = numpy.array([self.all_V[i] for i in indices]).sum(axis=0) / float(len(indices))  
        exp_tau = sum([self.all_tau[i] for i in indices]) / float(len(indices))
        exp_lambdak = None if not self.ARD else sum(
            [self.all_lambdak[i] for i in indices]) / float(len(indices))
        return (exp_U, exp_V, exp_tau, exp_lambdak)


    def predict(self,M_pred,burn_in,thinning):
        ''' Compute the expectation of U and V, and use it to predict missing values. '''
        (exp_U,exp_V,_,_) = self.approx_expectation(burn_in,thinning)
        R_pred = numpy.dot(exp_U, exp_V.T)
        MSE = self.compute_MSE(M_pred, self.R, R_pred)
        R2 = self.compute_R2(M_pred, self.R, R_pred)    
        Rp = self.compute_Rp(M_pred, self.R, R_pred)        
        return { 'MSE': MSE, 'R^2': R2, 'Rp': Rp }
        
    def predict_while_running(self):
        ''' Predict the training error while running. '''
        R_pred = numpy.dot(self.U, self.V.T)
        MSE = self.compute_MSE(self.M, self.R, R_pred)
        R2 = self.compute_R2(self.M, self.R, R_pred)    
        Rp = self.compute_Rp(self.M, self.R, R_pred)        
        return {'MSE': MSE, 'R^2': R2, 'Rp': Rp}
        
        
    ''' Functions for computing MSE, R^2 (coefficient of determination), Rp (Pearson correlation) '''
    def compute_MSE(self,M,R,R_pred):
        ''' Return the MSE of predictions in R_pred, expected values in R, for the entries in M. '''
        return (M * (R-R_pred)**2).sum() / float(M.sum())
        
    def compute_R2(self,M,R,R_pred):
        ''' Return the R^2 of predictions in R_pred, expected values in R, for the entries in M. '''
        mean = (M*R).sum() / float(M.sum())
        SS_total = float((M*(R-mean)**2).sum())
        SS_res = float((M*(R-R_pred)**2).sum())
        return 1. - SS_res / SS_total if SS_total != 0. else numpy.inf
        
    def compute_Rp(self,M,R,R_pred):
        ''' Return the Rp of predictions in R_pred, expected values in R, for the entries in M. '''
        mean_real = (M*R).sum() / float(M.sum())
        mean_pred = (M*R_pred).sum() / float(M.sum())
        covariance = (M*(R-mean_real)*(R_pred-mean_pred)).sum()
        variance_real = (M*(R-mean_real)**2).sum()
        variance_pred = (M*(R_pred-mean_pred)**2).sum()
        return covariance / float(math.sqrt(variance_real)*math.sqrt(variance_pred))
        
    
    def quality(self,metric,burn_in,thinning):
        ''' Return the model quality, either as log likelihood, BIC, AIC, MSE, or ELBO. '''
        assert metric in ALL_QUALITY, 'Unrecognised metric for model quality: %s.' % metric
        
        (exp_U, exp_V, exp_tau, _) = self.approx_expectation(burn_in,thinning)
        log_likelihood = self.log_likelihood(exp_U, exp_V, exp_tau)
        
        if metric == 'loglikelihood':
            return log_likelihood
        elif metric == 'BIC':
            # -2*loglikelihood + (no. free parameters * log(no data points))
            return - 2 * log_likelihood + self.number_parameters() * math.log(self.size_Omega)
        elif metric == 'AIC':
            # -2*loglikelihood + 2*no. free parameters
            return - 2 * log_likelihood + 2 * self.number_parameters()
        elif metric == 'MSE':
            R_pred = numpy.dot(exp_U, exp_V.T)
            return self.compute_MSE(self.M, self.R, R_pred)
        elif metric == 'ELBO':
            return 0.
        
    def log_likelihood(self, exp_U, exp_V, exp_tau):
        ''' Return the likelihood of the data given the trained model's parameters. '''
        exp_logtau = math.log(exp_tau)      
        return self.size_Omega / 2. * ( exp_logtau - math.log(2*math.pi) ) \
            - exp_tau / 2. * (self.M*( self.R - numpy.dot(exp_U,exp_V.T))**2).sum()
             
    def number_parameters(self):
        ''' Return the number of free variables in the model. '''
        return (self.I*self.K + self.J*self.K + 1) + (self.K if self.ARD else 0)