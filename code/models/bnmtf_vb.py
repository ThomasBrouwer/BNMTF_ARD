"""
Variational Bayesian inference for non-negative matrix tri-factorisation, with ARD.

We expect the following arguments:
- R, the matrix
- M, the mask matrix indicating observed values (1) and unobserved ones (0)
- K, the number of row clusters
- L, the number of column clusters
- hyperparameters = { 'alphatau', 'betatau', 'alpha0', 'beta0', 'lambdaS', 'lambdaF', 'lambdaG' },
    alphatau, betatau - non-negative reals defining prior over noise parameter tau.
    alpha0, beta0     - if using the ARD, non-negative reals defining prior over ARD lambdaFk and lambdaGl.
    lambdaS           - nonnegative reals defining prior over S
    lambdaF, lambdaG  - if not using the ARD, nonnegative reals defining prior over U and V
   
The random variables are initialised as follows:
    (lambdaFk, lambdaGl) alphaFk_s, betaFk_s, alphaGl_s, betaGl_s - set to alpha0, beta0
    (F,G) muF, muF - K-means ('kmeans'), expectation ('exp'), or random ('random')
    (S) muS - expectation ('exp') or random ('random')
    (F,G,S) tauF, tauG, tauS - set to 1
    (tau) alpha_s, beta_s - using updates
We initialise the values of F and G according to the given argument 'init_FG',
and S according to 'init_S'. 

Usage of class:
    BNMTF = bnmtf_vb(R,M,K,L,ARD,hyperparameters)
    BNMTF.initisalise(init_FG,init_S) 
    BNMTF.run(iterations)
Or:
    BNMTF = bnmtf_vb(R,M,K,L,ARD,hyperparameters)
    BNMTF.train(init_FG,init_S,iterations)
    
We can test the performance of our model on a test dataset, specifying our test set with a mask M. 
    performance = BNMTF.predict(M_pred)
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

from kmeans.kmeans import KMeans
from distributions.gamma import gamma_expectation, gamma_expectation_log
from distributions.truncated_normal import TN_expectation, TN_variance
from distributions.truncated_normal_vector import TN_vector_expectation, TN_vector_variance
from distributions.exponential import exponential_draw

import numpy, itertools, math, scipy, time

ALL_METRICS = ['MSE','R^2','Rp']
ALL_QUALITY = ['loglikelihood','BIC','AIC','MSE','ELBO']
OPTIONS_INIT_FG = ['kmeans', 'random', 'exp']
OPTIONS_INIT_S = ['random', 'exp']

class bnmtf_vb:
    def __init__(self,R,M,K,L,ARD,hyperparameters):
        self.R = numpy.array(R,dtype=float)
        self.M = numpy.array(M,dtype=float)
        self.K = K
        self.L = L
        self.ARD = ARD
        
        assert len(self.R.shape) == 2, "Input matrix R is not a two-dimensional array, " \
            "but instead %s-dimensional." % len(self.R.shape)
        assert self.R.shape == self.M.shape, "Input matrix R is not of the same size as " \
            "the indicator matrix M: %s and %s respectively." % (self.R.shape,self.M.shape)
            
        (self.I,self.J) = self.R.shape
        self.size_Omega = self.M.sum()
        self.check_empty_rows_columns()      
        
        self.alphatau, self.betatau = float(hyperparameters['alphatau']), float(hyperparameters['betatau'])
        self.lambdaS = numpy.array(hyperparameters['lambdaS'])
        if self.lambdaS.shape == ():
            self.lambdaS = self.lambdaS * numpy.ones((self.K,self.L))
        assert self.lambdaS.shape == (self.K,self.L), "Prior matrix lambdaS has the wrong shape: %s instead of (%s, %s)." % (self.lambdaS.shape,self.K,self.L)
            
        if self.ARD:
            self.alpha0, self.beta0 = float(hyperparameters['alpha0']), float(hyperparameters['beta0'])
        else:
            self.lambdaF, self.lambdaG = numpy.array(hyperparameters['lambdaF']), numpy.array(hyperparameters['lambdaG'])
            # Make lambdaF/G into a numpy array if they are a float
            if self.lambdaF.shape == ():
                self.lambdaF = self.lambdaF * numpy.ones((self.I,self.K))
            if self.lambdaG.shape == ():
                self.lambdaG = self.lambdaG * numpy.ones((self.J,self.L))
        
            assert self.lambdaF.shape == (self.I,self.K), "Prior matrix lambdaF has the wrong shape: %s instead of (%s, %s)." % (self.lambdaF.shape,self.I,self.K)
            assert self.lambdaG.shape == (self.J,self.L), "Prior matrix lambdaG has the wrong shape: %s instead of (%s, %s)." % (self.lambdaG.shape,self.J,self.L)
                
            
    def check_empty_rows_columns(self):
        ''' Raise an exception if an entire row or column is empty. '''
        sums_columns = self.M.sum(axis=0)
        sums_rows = self.M.sum(axis=1)
                    
        # Assert none of the rows or columns are entirely unknown values
        for i,c in enumerate(sums_rows):
            assert c != 0, "Fully unobserved row in R, row %s." % i
        for j,c in enumerate(sums_columns):
            assert c != 0, "Fully unobserved column in R, column %s." % j


    def train(self,init_FG,init_S,iterations):
        ''' Initialise and run the sampler. '''
        self.initialise(init_FG=init_FG, init_S=init_S)
        self.run(iterations)


    def initialise(self,init_FG='random',init_S='random'):
        ''' Initialise F, S, G, tau, and lambdaFk, lambdaGl (if ARD). '''
        assert init_FG in OPTIONS_INIT_FG, "Unknown initialisation option for F and G: %s. Should be in %s." % (init_FG, OPTIONS_INIT_FG)
        assert init_S in OPTIONS_INIT_S, "Unknown initialisation option for S: %s. Should be in %s." % (init_S, OPTIONS_INIT_S)
        
        # Initialise lambdaFk, lambdaGl, and compute expectations
        if self.ARD:
            self.alphaFk_s, self.betaFk_s = numpy.zeros(self.K), numpy.zeros(self.K)
            self.alphaGl_s, self.betaGl_s = numpy.zeros(self.L), numpy.zeros(self.L)
            self.exp_lambdaFk, self.exp_loglambdaFk = numpy.zeros(self.K), numpy.zeros(self.K)
            self.exp_lambdaGl, self.exp_loglambdaGl = numpy.zeros(self.L), numpy.zeros(self.L)
            for k in range(self.K):
                self.alphaFk_s[k] = self.alpha0
                self.betaFk_s[k] = self.beta0
                self.update_exp_lambdaFk(k)
            for l in range(self.L):
                self.alphaGl_s[l] = self.alpha0
                self.betaGl_s[l] = self.beta0
                self.update_exp_lambdaGl(l)
                
        # Initialise parameters F, G
        self.mu_F, self.tau_F = numpy.zeros((self.I,self.K)), numpy.zeros((self.I,self.K))
        self.mu_G, self.tau_G = numpy.zeros((self.J,self.L)), numpy.zeros((self.J,self.L))
        self.mu_S, self.tau_S = numpy.zeros((self.K,self.L)), numpy.zeros((self.K,self.L))
        
        if init_FG == 'kmeans':
            print "Initialising F using KMeans."
            kmeans_F = KMeans(self.R,self.M,self.K)
            kmeans_F.initialise()
            kmeans_F.cluster()
            self.mu_F = kmeans_F.clustering_results    
            
            for i,k in itertools.product(range(self.I),range(self.K)):  
                self.tau_F[i,k] = 1.       
            
            print "Initialising G using KMeans."
            kmeans_G = KMeans(self.R.T,self.M.T,self.L)   
            kmeans_G.initialise()
            kmeans_G.cluster()
            self.mu_G = kmeans_G.clustering_results
            
            for j,l in itertools.product(range(self.J),range(self.L)):
                self.tau_G[j,l] = 1.
        else:
            # 'random' or 'exp'
            for i,k in itertools.product(range(self.I),range(self.K)):  
                self.tau_F[i,k] = 1.
                hyperparam = self.exp_lambdaFk[k] if self.ARD else self.lambdaF[i,k]
                self.mu_F[i,k] = exponential_draw(hyperparam) if init_FG == 'random' else 1.0/hyperparam
            for j,l in itertools.product(range(self.J),range(self.L)):
                self.tau_G[j,l] = 1.
                hyperparam = self.exp_lambdaGl[l] if self.ARD else self.lambdaG[j,l]
                self.mu_G[j,l] = exponential_draw(hyperparam) if init_FG == 'random' else 1.0/hyperparam
            
        # Initialise parameters S
        for k,l in itertools.product(range(self.K),range(self.L)):
            self.tau_S[k,l] = 1.
            hyperparam = self.lambdaS[k,l]
            self.mu_S[k,l] = exponential_draw(hyperparam) if init_S == 'random' else 1.0/hyperparam
        
        # Compute expectations and variances F, G, S
        self.exp_F, self.var_F = numpy.zeros((self.I,self.K)), numpy.zeros((self.I,self.K))
        self.exp_G, self.var_G = numpy.zeros((self.J,self.L)), numpy.zeros((self.J,self.L))
        self.exp_S, self.var_S = numpy.zeros((self.K,self.L)), numpy.zeros((self.K,self.L))
        
        for k in range(self.K):
            self.update_exp_F(k)
        for l in range(self.L):
            self.update_exp_G(l)
        for k,l in itertools.product(range(self.K),range(self.L)):
            self.update_exp_S(k,l)

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
            # Update lambdaFk and lambdaGl
            if self.ARD:
                for k in range(self.K):
                    self.update_lambdaFk(k)
                    self.update_exp_lambdaFk(k)
                for l in range(self.L):
                    self.update_lambdaGl(l)
                    self.update_exp_lambdaGl(l)
            
            # Update F
            for k in range(self.K):
                self.update_F(k)
                self.update_exp_F(k)
                
            # Update S
            for k,l in itertools.product(range(self.K),range(self.L)):
                self.update_S(k,l)
                self.update_exp_S(k,l)
                
            # Update G
            for l in range(0,self.L):
                self.update_G(l)
                self.update_exp_G(l)
            
            # Update tau
            self.update_tau()
            self.update_exp_tau()
            
            # Store expectations
            self.all_exp_tau.append(self.exp_tau)
            
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
        total_elbo += self.size_Omega / 2. * ( self.exp_logtau - math.log(2*math.pi) ) \
                      - self.exp_tau / 2. * self.exp_square_diff()
                      
        # Prior lambdaFk and lambdaGl, if using ARD, and prior F,G
        if self.ARD:
            total_elbo += self.alpha0 * math.log(self.beta0) - scipy.special.gammaln(self.alpha0) \
                          + (self.alpha0 - 1.)*self.exp_loglambdaFk.sum() - self.beta0 * self.exp_lambdaFk.sum()
            total_elbo += self.alpha0 * math.log(self.beta0) - scipy.special.gammaln(self.alpha0) \
                          + (self.alpha0 - 1.)*self.exp_loglambdaGl.sum() - self.beta0 * self.exp_lambdaGl.sum()
            
            total_elbo += self.I * numpy.log(self.exp_lambdaFk).sum() - ( self.exp_lambdaFk * self.exp_F ).sum()
            total_elbo += self.J * numpy.log(self.exp_lambdaGl).sum() - ( self.exp_lambdaGl * self.exp_G ).sum()
            
        else:
            total_elbo += numpy.log(self.lambdaF).sum() - ( self.lambdaF * self.exp_F ).sum()
            total_elbo += numpy.log(self.lambdaG).sum() - ( self.lambdaG * self.exp_G ).sum()
        
        # Prior S
        total_elbo += numpy.log(self.lambdaS).sum() - ( self.lambdaS * self.exp_S ).sum()
        
        # Prior tau
        total_elbo += self.alphatau * math.log(self.betatau) - scipy.special.gammaln(self.alphatau) \
                      + (self.alphatau - 1.)*self.exp_logtau - self.betatau * self.exp_tau
        
        # q for lambdaFk and lambdaGl, if using ARD
        if self.ARD:
            total_elbo += - sum([v1*math.log(v2) for v1,v2 in zip(self.alphaFk_s,self.betaFk_s)]) + sum([scipy.special.gammaln(v) for v in self.alphaFk_s]) \
                          - ((self.alphaFk_s - 1.)*self.exp_loglambdaFk).sum() + (self.betaFk_s * self.exp_lambdaFk).sum()
            total_elbo += - sum([v1*math.log(v2) for v1,v2 in zip(self.alphaGl_s,self.betaGl_s)]) + sum([scipy.special.gammaln(v) for v in self.alphaGl_s]) \
                          - ((self.alphaGl_s - 1.)*self.exp_loglambdaGl).sum() + (self.betaGl_s * self.exp_lambdaGl).sum()
            
        # q for F, G, S
        total_elbo += - .5*numpy.log(self.tau_F).sum() + self.I*self.K/2.*math.log(2*math.pi) \
                      + numpy.log(0.5*scipy.special.erfc(-self.mu_F*numpy.sqrt(self.tau_F)/math.sqrt(2))).sum() \
                      + ( self.tau_F / 2. * ( self.var_F + (self.exp_F - self.mu_F)**2 ) ).sum()
        total_elbo += - .5*numpy.log(self.tau_G).sum() + self.J*self.L/2.*math.log(2*math.pi) \
                      + numpy.log(0.5*scipy.special.erfc(-self.mu_G*numpy.sqrt(self.tau_G)/math.sqrt(2))).sum() \
                      + ( self.tau_G / 2. * ( self.var_G + (self.exp_G - self.mu_G)**2 ) ).sum()
        total_elbo += - .5*numpy.log(self.tau_S).sum() + self.K*self.L/2.*math.log(2*math.pi) \
                      + numpy.log(0.5*scipy.special.erfc(-self.mu_S*numpy.sqrt(self.tau_S)/math.sqrt(2))).sum() \
                      + ( self.tau_S / 2. * ( self.var_S + (self.exp_S - self.mu_S)**2 ) ).sum()
        
        # q for tau
        total_elbo += - self.alpha_s * math.log(self.beta_s) + scipy.special.gammaln(self.alpha_s) \
                      - (self.alpha_s - 1.)*self.exp_logtau + self.beta_s * self.exp_tau
        
        return total_elbo        
        
        
    def triple_dot(self,M1,M2,M3):
        ''' Triple matrix multiplication: M1*M2*M3. 
            If the matrices have dimensions I,K,L,J, then the complexity of M1*(M2*M3) 
            is ~IJK, and (M1*M2)*M3 is ~IJL. So if K < L, we use the former. '''
        K,L = M2.shape
        if K < L:
            return numpy.dot(M1,numpy.dot(M2,M3))
        else:
            return numpy.dot(numpy.dot(M1,M2),M3)
        
        
    ''' Update the parameters for the distributions. '''
    def update_tau(self):   
        ''' Parameter updates tau. '''
        self.alpha_s = self.alphatau + self.size_Omega/2.0
        self.beta_s = self.betatau + 0.5*self.exp_square_diff()
        
    def exp_square_diff(self): 
        ''' Compute: sum_Omega E_q(F,S,G) [ ( Rij - Fi S Gj )^2 ]. '''
        return (self.M*( self.R - self.triple_dot(self.exp_F,self.exp_S,self.exp_G.T) )**2).sum() + \
               (self.M*( self.triple_dot(self.var_F+self.exp_F**2, self.var_S+self.exp_S**2, (self.var_G+self.exp_G**2).T ) - self.triple_dot(self.exp_F**2,self.exp_S**2,(self.exp_G**2).T) )).sum() + \
               (self.M*( numpy.dot(self.var_F, ( numpy.dot(self.exp_S,self.exp_G.T)**2 - numpy.dot(self.exp_S**2,self.exp_G.T**2) ) ) )).sum() + \
               (self.M*( numpy.dot( numpy.dot(self.exp_F,self.exp_S)**2 - numpy.dot(self.exp_F**2,self.exp_S**2), self.var_G.T ) )).sum()
    
    def update_lambdaFk(self,k):   
        ''' Parameter updates lambdaFk. '''
        self.alphaFk_s[k] = self.alpha0 + self.I
        self.betaFk_s[k] = self.beta0 + self.exp_F[:,k].sum()
        
    def update_lambdaGl(self,l):   
        ''' Parameter updates lambdaFk. '''
        self.alphaGl_s[l] = self.alpha0 + self.J
        self.betaGl_s[l] = self.beta0 + self.exp_G[:,l].sum()
        
    def update_F(self,k):  
        ''' Parameter updates F. ''' 
        var_SkG = numpy.dot( self.var_S[k]+self.exp_S[k]**2 , (self.var_G+self.exp_G**2).T ) - numpy.dot( self.exp_S[k]**2 , (self.exp_G**2).T ) # Vector of size J
        self.tau_F[:,k] = self.exp_tau * numpy.dot( var_SkG + ( numpy.dot(self.exp_S[k],self.exp_G.T) )**2 , self.M.T ) 
        
        lamb = self.exp_lambdaFk[k] if self.ARD else self.lambdaF[:,k]
        diff_term = (self.M * ( (self.R-self.triple_dot(self.exp_F,self.exp_S,self.exp_G.T)+numpy.outer(self.exp_F[:,k],numpy.dot(self.exp_S[k],self.exp_G.T)) ) * numpy.dot(self.exp_S[k],self.exp_G.T) )).sum(axis=1)        
        cov_term = ( self.M * ( ( numpy.dot(self.exp_S[k]*numpy.dot(self.exp_F,self.exp_S), self.var_G.T) - numpy.outer(self.exp_F[:,k], numpy.dot( self.exp_S[k]**2, self.var_G.T )) ) ) ).sum(axis=1)
        self.mu_F[:,k] = 1./self.tau_F[:,k] * (
            - lamb
            + self.exp_tau * diff_term
            - self.exp_tau * cov_term
        ) 
        
    def update_S(self,k,l):   
        ''' Parameter updates S. '''     
        self.tau_S[k,l] = self.exp_tau*(self.M*( numpy.outer( self.var_F[:,k]+self.exp_F[:,k]**2 , self.var_G[:,l]+self.exp_G[:,l]**2 ) )).sum()
        
        diff_term = (self.M * ( (self.R-self.triple_dot(self.exp_F,self.exp_S,self.exp_G.T)+self.exp_S[k,l]*numpy.outer(self.exp_F[:,k],self.exp_G[:,l]) ) * numpy.outer(self.exp_F[:,k],self.exp_G[:,l]) )).sum()
        cov_term_G = (self.M * numpy.outer( self.exp_F[:,k] * ( numpy.dot(self.exp_F,self.exp_S[:,l]) - self.exp_F[:,k]*self.exp_S[k,l] ), self.var_G[:,l] )).sum()
        cov_term_F = (self.M * numpy.outer( self.var_F[:,k], self.exp_G[:,l]*(numpy.dot(self.exp_S[k],self.exp_G.T) - self.exp_S[k,l]*self.exp_G[:,l]) )).sum()        
        self.mu_S[k,l] = 1./self.tau_S[k,l] * (
            - self.lambdaS[k,l] 
            + self.exp_tau * diff_term
            - self.exp_tau * cov_term_G
            - self.exp_tau * cov_term_F
        ) 
        
    def update_G(self,l):  
        var_FSl = numpy.dot( self.var_F+self.exp_F**2 , self.var_S[:,l]+self.exp_S[:,l]**2 ) - numpy.dot( self.exp_F**2 , self.exp_S[:,l]**2 ) # Vector of size I
        self.tau_G[:,l] = self.exp_tau * numpy.dot( ( var_FSl + ( numpy.dot(self.exp_F,self.exp_S[:,l]) )**2 ).T, self.M) #sum over i, so columns        
        
        lamb = self.exp_lambdaGl[l] if self.ARD else self.lambdaG[:,l]
        diff_term = (self.M * ( (self.R-self.triple_dot(self.exp_F,self.exp_S,self.exp_G.T)+numpy.outer(numpy.dot(self.exp_F,self.exp_S[:,l]), self.exp_G[:,l]) ).T * numpy.dot(self.exp_F,self.exp_S[:,l]) ).T ).sum(axis=0)
        cov_term = (self.M * ( numpy.dot(self.var_F, (self.exp_S[:,l]*numpy.dot(self.exp_S,self.exp_G.T).T).T) - numpy.outer(numpy.dot(self.var_F,self.exp_S[:,l]**2), self.exp_G[:,l]) )).sum(axis=0)
        self.mu_G[:,l] = 1./self.tau_G[:,l] * (
            - lamb
            + self.exp_tau * diff_term
            - self.exp_tau * cov_term
        )


    ''' Update the expectations and variances. '''
    def update_exp_tau(self):
        ''' Update expectation tau. '''
        self.exp_tau = gamma_expectation(self.alpha_s,self.beta_s)
        self.exp_logtau = gamma_expectation_log(self.alpha_s,self.beta_s)
        
    def update_exp_lambdaFk(self,k):
        ''' Update expectation lambdaFk. '''
        self.exp_lambdaFk[k] = gamma_expectation(self.alphaFk_s[k],self.betaFk_s[k])
        self.exp_loglambdaFk[k] = gamma_expectation_log(self.alphaFk_s[k],self.betaFk_s[k])
        
    def update_exp_lambdaGl(self,l):
        ''' Update expectation lambdaGl. '''
        self.exp_lambdaGl[l] = gamma_expectation(self.alphaGl_s[l],self.betaGl_s[l])
        self.exp_loglambdaGl[l] = gamma_expectation_log(self.alphaGl_s[l],self.betaGl_s[l])
        
    def update_exp_F(self,k):
        ''' Update expectation F. '''
        self.exp_F[:,k] = TN_vector_expectation(self.mu_F[:,k],self.tau_F[:,k])
        self.var_F[:,k] = TN_vector_variance(self.mu_F[:,k],self.tau_F[:,k])
        
    def update_exp_S(self,k,l):
        ''' Update expectation S. '''
        self.exp_S[k,l] = TN_expectation(self.mu_S[k,l],self.tau_S[k,l])
        self.var_S[k,l] = TN_variance(self.mu_S[k,l],self.tau_S[k,l])
        
    def update_exp_G(self,l):
        ''' Update expectation G. '''
        self.exp_G[:,l] = TN_vector_expectation(self.mu_G[:,l],self.tau_G[:,l])
        self.var_G[:,l] = TN_vector_variance(self.mu_G[:,l],self.tau_G[:,l])


    def predict(self,M_pred):
        ''' Predict missing values in R. '''
        R_pred = self.triple_dot(self.exp_F,self.exp_S,self.exp_G.T)
        MSE = self.compute_MSE(M_pred,self.R,R_pred)
        R2 = self.compute_R2(M_pred,self.R,R_pred)    
        Rp = self.compute_Rp(M_pred,self.R,R_pred)        
        return {'MSE':MSE,'R^2':R2,'Rp':Rp}
        
        
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
        assert metric in ['loglikelihood','BIC','AIC','MSE','ELBO'], 'Unrecognised metric for model quality: %s.' % metric
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
            R_pred = self.triple_dot(self.exp_F,self.exp_S,self.exp_G.T)
            return self.compute_MSE(self.M,self.R,R_pred)
        elif metric == 'ELBO':
            return self.elbo()
        
    def log_likelihood(self):
        ''' Return the likelihood of the data given the trained model's parameters. '''
        return self.size_Omega / 2. * ( self.exp_logtau - math.log(2*math.pi) ) \
             - self.exp_tau / 2. * (self.M*( self.R - self.triple_dot(self.exp_F,self.exp_S,self.exp_G.T) )**2).sum()
             
    def number_parameters(self):
        ''' Return the number of free variables in the model. '''
        return (self.I*self.K + self.K*self.L + self.J*self.L + 1) + (self.K+self.L if self.ARD else 0)