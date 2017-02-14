"""
Variational Bayesian inference for non-negative matrix factorisation, with ARD.

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
    (lambdak) alphak_s, betak_s - set to alpha0, beta0
    (U,V) muU, muV - expectation ('exp') or random ('random')
    (U,V) tauU, tauV - set to 1
    (tau) alpha_s, beta_s - using updates
We initialise the values of U and V according to the given argument 'init_UV'. 

Usage of class:
    BNMF = bnmf_vb(R,M,K,ARD,hyperparameters)
    BNMF.initisalise(init_UV)      
    BNMF.run(iterations)
Or:
    BNMF = bnmf_vb(R,M,K,ARD,hyperparameters)
    BNMF.train(init_UV,iterations)

We can test the performance of our model on a test dataset, specifying our test set with a mask M. 
    performance = BNMF.predict(M_pred)
This gives a dictionary of performances,
    performance = { 'MSE', 'R^2', 'Rp' }
    
The performances of all iterations are stored in BNMF.all_performances, which 
is a dictionary from 'MSE', 'R^2', or 'Rp' to a list of performances.
    
Finally, we can return the goodness of fit of the data using the quality(metric) function:
- metric = 'loglikelihood' -> return p(D|theta)
         = 'BIC'        -> return Bayesian Information Criterion
         = 'AIC'        -> return Afaike Information Criterion
         = 'MSE'        -> return Mean Square Error
         = 'ELBO'       -> return Evidence Lower Bound
"""

from distributions.gamma import gamma_expectation, gamma_expectation_log
from distributions.truncated_normal_vector import TN_vector_expectation, TN_vector_variance
from distributions.exponential import exponential_draw

import numpy, itertools, math, scipy, time

ALL_METRICS = ['MSE','R^2','Rp']
ALL_QUALITY = ['loglikelihood','BIC','AIC','MSE','ELBO']
OPTIONS_INIT_UV = ['random', 'exp']

class bnmf_vb:
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
        ''' Initialise and run the algorithm. '''
        self.initialise(init_UV=init_UV)
        self.run(iterations)


    def initialise(self,init_UV='exp'):
        ''' Initialise U, V, tau, and lambda (if ARD). '''
        assert init_UV in OPTIONS_INIT_UV, "Unknown initialisation option: %s. Should be in %s." % (init_UV, OPTIONS_INIT_UV)
        
        # Initialise lambdak, and compute expectation
        if self.ARD:
            self.alphak_s, self.betak_s = numpy.zeros(self.K), numpy.zeros(self.K)
            for k in range(self.K):
                self.alphak_s[k] = self.alpha0
                self.betak_s[k] = self.beta0
                self.update_exp_lambdak(k)
                
        # Initialise parameters U, V
        self.mu_U, self.tau_U = numpy.zeros((self.I,self.K)), numpy.zeros((self.I,self.K))
        self.mu_V, self.tau_V = numpy.zeros((self.J,self.K)), numpy.zeros((self.J,self.K))
        
        for i,k in itertools.product(range(self.I),range(self.K)):  
            self.tau_U[i,k] = 1.
            hyperparam = self.lambdak[k] if self.ARD else self.lambdaU[i,k]
            self.mu_U[i,k] = exponential_draw(hyperparam) if init_UV == 'random' else 1.0/hyperparam
        for j,k in itertools.product(range(self.J),range(self.K)):
            self.tau_V[j,k] = 1.
            hyperparam = self.lambdak[k] if self.ARD else self.lambdaV[j,k]
            self.mu_V[j,k] = exponential_draw(hyperparam) if init_UV == 'random' else 1.0/hyperparam
        
        # Compute expectations and variances U, V
        self.exp_U, self.var_U = numpy.zeros((self.I,self.K)), numpy.zeros((self.I,self.K))
        self.exp_V, self.var_V = numpy.zeros((self.J,self.K)), numpy.zeros((self.J,self.K))
        
        for k in range(self.K):
            self.update_exp_U(k)
        for k in range(self.K):
            self.update_exp_V(k)

        # Initialise tau and compute expectation
        self.update_tau()
        self.update_exp_tau()
        

    def run(self,iterations):
        ''' Run the Gibbs sampler. '''
        self.all_exp_tau = []  # to check for convergence 
        self.all_times = [] # to plot performance against time
        
        self.all_performances = {} # for plotting convergence of metrics
        for metric in ALL_METRICS:
            self.all_performances[metric] = []
        
        time_start = time.time()
        for it in range(iterations):
            # Update lambdak
            if self.ARD:
                for k in range(self.K):
                    self.update_lambdak(k)
                    self.update_exp_lambdak(k)
            
            # Update U
            for k in range(self.K):
                self.update_U(k)
                self.update_exp_U(k)    
                
            # Update V
            for k in range(self.K):
                self.update_V(k)
                self.update_exp_V(k)
                
            # Update tau
            self.update_tau()
            self.update_exp_tau()
            
            # Store expectations
            self.all_exp_tau.append(self.exptau)
            
            # Store and print performances
            perf, elbo = self.predict(self.M), self.elbo()
            for metric in ALL_METRICS:
                self.all_performances[metric].append(perf[metric])
                
            print "Iteration %s. ELBO: %s. MSE: %s. R^2: %s. Rp: %s." % (it+1,elbo,perf['MSE'],perf['R^2'],perf['Rp'])
            
            # Store time taken for iteration
            time_iteration = time.time()
            self.all_times.append(time_iteration-time_start)            
        
        
    def elbo(self):
        ''' Compute the ELBO. '''
        total_elbo = 0.
        
        # Log likelihood               
        total_elbo += self.size_Omega / 2. * ( self.explogtau - math.log(2*math.pi) ) \
                      - self.exptau / 2. * self.exp_square_diff()
                      
        # Prior lambdak, if using ARD, and prior U, V
        if self.ARD:
            total_elbo += self.alpha0 * math.log(self.beta0) - scipy.special.gammaln(self.alpha0) \
                          + (self.alpha0 - 1.)*self.exp_loglambdak.sum() - self.beta0 * self.exp_lambdak
            
            total_elbo += self.I * numpy.log(self.exp_lambdak).sum() - ( self.exp_lambdak * self.exp_U ).sum()
            total_elbo += self.J * numpy.log(self.exp_lambdak).sum() - ( self.exp_lambdak * self.exp_V ).sum()
            
        else:
            total_elbo += numpy.log(self.lambdaU).sum() - ( self.lambdaU * self.exp_U ).sum()
            total_elbo += numpy.log(self.lambdaV).sum() - ( self.lambdaV * self.exp_V ).sum()
        
        # Prior tau
        total_elbo += self.alphatau * math.log(self.betatau) - scipy.special.gammaln(self.alphatau) \
                      + (self.alphatau - 1.)*self.explogtau - self.betatau * self.exptau
        
        # q for lambdak, if using ARD
        if self.ARD:
            total_elbo += - (self.alphak_s * math.log(self.betak_s)).sum() + scipy.special.gammaln(self.alphak_s).sum() \
                          - ((self.alphak_s - 1.)*self.exp_loglambdak).sum() + (self.betak_s * self.exp_lambdak).sum()
            
        # q for U, V
        total_elbo += - .5*numpy.log(self.tau_U).sum() + self.I*self.K/2.*math.log(2*math.pi) \
                      + numpy.log(0.5*scipy.special.erfc(-self.mu_U*numpy.sqrt(self.tau_U)/math.sqrt(2))).sum() \
                      + ( self.tau_U / 2. * ( self.var_U + (self.exp_U - self.mu_U)**2 ) ).sum()
        total_elbo += - .5*numpy.log(self.tauV).sum() + self.J*self.K/2.*math.log(2*math.pi) \
                      + numpy.log(0.5*scipy.special.erfc(-self.mu_V*numpy.sqrt(self.tau_V)/math.sqrt(2))).sum() \
                      + ( self.tau_V / 2. * ( self.var_V + (self.exp_V - self.mu_V)**2 ) ).sum()
        
        # q for tau
        total_elbo += - self.alpha_s * math.log(self.beta_s) + scipy.special.gammaln(self.alpha_s) \
                      - (self.alpha_s - 1.)*self.exp_logtau + self.beta_s * self.exp_tau
        
        return total_elbo
        
        
    ''' Update the parameters for the distributions. '''
    def update_tau(self):   
        ''' Parameter updates tau. '''
        self.alpha_s = self.alphatau + self.size_Omega/2.0
        self.beta_s = self.betatau + 0.5*self.exp_square_diff()
        
    def exp_square_diff(self): 
        ''' Compute: sum_Omega E_q(U,V) [ ( Rij - Ui Vj )^2 ]. '''
        return (self.M *( ( self.R - numpy.dot(self.exp_U,self.exp_V.T) )**2 + \
                          ( numpy.dot(self.var_U+self.exp_U**2, (self.var_V+self.exp_V**2).T) - numpy.dot(self.exp_U**2,(self.exp_V**2).T) ) ) ).sum()
        
    def update_lambdak(self,k):   
        ''' Parameter updates lambdak. '''
        self.alphak_s = self.alpha0 + self.I + self.J
        self.betak_s = self.beta0 + self.exp_U[:,k].sum() + self.exp_V[:,k].sum()
        
    def update_U(self,k):   
        ''' Parameter updates U. '''   
        lamb = self.exp_lambdak[k] if self.ARD else self.lambdaU[:,k]
        self.tau_U[:,k] = self.exp_tau*(self.M*( self.var_V[:,k] + self.exp_V[:,k]**2 )).sum(axis=1) #sum over j, so rows
        self.mu_U[:,k] = 1./self.tau_U[:,k] * (-lamb + self.exp_tau*(self.M * ( (self.R-numpy.dot(self.exp_U,self.exp_V.T)+numpy.outer(self.exp_U[:,k],self.exp_V[:,k]))*self.exp_V[:,k] )).sum(axis=1)) 
        
    def update_V(self,k):
        ''' Parameter updates V. '''
        lamb = self.exp_lambdak[k] if self.ARD else self.lambdaV[:,k]
        self.tau_V[:,k] = self.exp_tau*(self.M.T*( self.var_U[:,k] + self.exp_U[:,k]**2 )).T.sum(axis=0) #sum over i, so columns
        self.mu_V[:,k] = 1./self.tau_V[:,k] * (-lamb + self.exp_tau*(self.M.T * ( (self.R-numpy.dot(self.exp_U,self.exp_V.T)+numpy.outer(self.exp_U[:,k],self.exp_V[:,k])).T*self.exp_U[:,k] )).T.sum(axis=0)) 
        
        
    ''' Update the expectations and variances. '''
    def update_exp_tau(self):
        ''' Update expectation tau. '''
        self.exp_tau = gamma_expectation(self.alpha_s,self.beta_s)
        self.exp_logtau = gamma_expectation_log(self.alpha_s,self.beta_s)
        
    def update_exp_lambdak(self,k):
        ''' Update expectation lambdak. '''
        self.exp_lambdak[k] = gamma_expectation(self.alphak_s[k],self.betak_s[k])
        self.exp_loglambdak[k] = gamma_expectation_log(self.alphak_s[k],self.betak_s[k])
    
    def update_exp_U(self,k):
        ''' Update expectation U. '''
        self.exp_U[:,k] = TN_vector_expectation(self.muU[:,k],self.tauU[:,k])
        self.var_U[:,k] = TN_vector_variance(self.muU[:,k],self.tauU[:,k])
        
    def update_exp_V(self,k):
        ''' Update expectation V. '''
        self.exp_V[:,k] = TN_vector_expectation(self.muV[:,k],self.tauV[:,k])
        self.var_V[:,k] = TN_vector_variance(self.muV[:,k],self.tauV[:,k])


    def predict(self, M_pred):
        ''' Predict missing values in R. '''
        R_pred = numpy.dot(self.expU, self.expV.T)
        MSE = self.compute_MSE(M_pred, self.R, R_pred)
        R2 = self.compute_R2(M_pred, self.R, R_pred)    
        Rp = self.compute_Rp(M_pred, self.R, R_pred)        
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
        
        
    def quality(self,metric):        
        ''' Return the model quality, either as log likelihood, BIC, AIC, MSE, or ELBO. '''
        assert metric in ALL_QUALITY, 'Unrecognised metric for model quality: %s.' % metric
        
        log_likelihood = self.log_likelihood()
        if metric == 'loglikelihood':
            return log_likelihood
        elif metric == 'BIC':
            # -2*loglikelihood + (no. free parameters * log(no data points))
            return - 2 * log_likelihood + self.number_parameters() * math.log(self.size_Omega)
        elif metric == 'AIC':
            # -2*loglikelihood + 2*no. free parameters
            return - 2 * log_likelihood + 2 * self.number_parameters()
        elif metric == 'MSE':
            R_pred = numpy.dot(self.expU,self.expV.T)
            return self.compute_MSE(self.M,self.R,R_pred)
        elif metric == 'ELBO':
            return self.elbo()
        
    def log_likelihood(self):
        ''' Return the likelihood of the data given the trained model's parameters. '''
        return self.size_Omega / 2. * ( self.exp_logtau - math.log(2*math.pi) ) \
             - self.exp_tau / 2. * (self.M*( self.R - numpy.dot(self.exp_U,self.exp_V.T))**2).sum()
             
    def number_parameters(self):
        ''' Return the number of free variables in the model. '''
        return (self.I*self.K + self.J*self.K + 1) + (self.K if self.ARD else 0)