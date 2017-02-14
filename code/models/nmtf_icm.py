"""
Iterated Conditional Modes for MAP non-negative matrix tri-factorisation, with ARD.

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
    F,G: K-means ('kmeans'), expectation ('exp'), or random ('random')
    S:   expectation ('exp') or random ('random')
    tau: using updates, and then random draw
    lambdaFk, lambdaGl: expectation
We initialise the values of F and G according to the given argument 'init_FG',
and S according to 'init_S'. 

Usage of class:
    BNMTF = bnmf_gibbs(R, M, K, L, ARD, hyperparameters)
    BNMTF.initisalise(init_FG, init_S)
    BNMTF.run(iterations)
Or:
    BNMTF = bnmf_gibbs(R,M,K,L,hyperparameters)
    BNMTF.train(init_FG, init_S, iterations)
    
The draws for all iterations are stored in: all_F, all_S, all_G, all_lambdaFk, all_lambdaGl, all_tau.
    
The expectation can be computed by specifying a burn-in and thinning rate, and using:
    BNMTF.approx_expectation(burn_in,thinning)
This returns a tuple (exp_F, exp_S, exp_G, exp_tau, exp_lambdaFk, exp_lambdaGl).

We can test the performance of our model on a test dataset, specifying our test set with a mask M. 
    performance = BNMTF.predict(M_pred,burn_in,thinning)
This gives a dictionary of performances,
    performance = { 'MSE', 'R^2', 'Rp' }
    
The performances of all iterations are stored in BNMTF.all_performances, which 
is a dictionary from 'MSE', 'R^2', or 'Rp' to a list of performances.
    
Finally, we can return the goodness of fit of the data using the quality(metric) function:
- metric = 'loglikelihood' -> return p(D|theta)
         = 'BIC'           -> return Bayesian Information Criterion
         = 'AIC'           -> return Afaike Information Criterion
         = 'MSE'           -> return Mean Square Error
         = 'ELBO'          -> N/A (only for VB)
"""

from kmeans.kmeans import KMeans
from distributions.exponential import exponential_draw
from distributions.gamma import gamma_mode
from distributions.truncated_normal import TN_mode
from distributions.truncated_normal_vector import TN_vector_mode

import numpy, itertools, math, time

ALL_METRICS = ['MSE','R^2','Rp']
ALL_QUALITY = ['loglikelihood','BIC','AIC','MSE','ELBO']
OPTIONS_INIT_FG = ['kmeans', 'random', 'exp']
OPTIONS_INIT_S = ['random', 'exp']

class nmtf_icm:
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
            self.lambdaS = self.lambdaV * numpy.ones((self.K,self.L))
        assert self.lambdaS.shape == (self.K,self.L), "Prior matrix lambdaS has the wrong shape: %s instead of (%s, %s)." % (self.lambdaS.shape,self.K,self.L)
            
        if self.ARD:
            self.alpha0, self.beta0 = float(hyperparameters['alpha0']), float(hyperparameters['beta0'])
        else:
            self.lambdaF, self.lambdaG = float(hyperparameters['lambdaF']), float(hyperparameters['lambdaG'])
            # Make lambdaF/G into a numpy array if they are a float
            if self.lambdaF.shape == ():
                self.lambdaF = self.lambdaF * numpy.ones((self.I,self.K))
            if self.lambdaG.shape == ():
                self.lambdaG = self.lambdaV * numpy.ones((self.J,self.L))
        
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
        
        self.F = numpy.zeros((self.I,self.K))
        self.S = numpy.zeros((self.K,self.L))
        self.G = numpy.zeros((self.J,self.L))
        self.lambdaFk = numpy.zeros(self.K)  
        self.lambdaGl = numpy.zeros(self.L)  
        
        # Initialise lambdaFk, lambdaGl
        if self.ARD:
            for k in range(self.K):
                self.lambdaFk[k] = self.alpha0 / self.beta0
            for l in range(self.L):
                self.lambdaGl[l] = self.alpha0 / self.beta0
        
        # Initialise F, G
        if init_FG == 'kmeans':
            print "Initialising F using KMeans."
            kmeans_F = KMeans(self.R,self.M,self.K)
            kmeans_F.initialise()
            kmeans_F.cluster()
            self.F = kmeans_F.clustering_results + 0.2  
            
            print "Initialising G using KMeans."
            kmeans_G = KMeans(self.R.T,self.M.T,self.L)   
            kmeans_G.initialise()
            kmeans_G.cluster()
            self.mu_G = kmeans_G.clustering_results + 0.2
        else:
            # 'random' or 'exp'
            for i,k in itertools.product(range(self.I),range(self.K)):    
                hyperparam = self.lambdaFk[k] if self.ARD else self.lambdaF[i,k]
                self.F[i,k] = exponential_draw(hyperparam) if init_FG == 'random' else 1.0/hyperparam
            for j,l in itertools.product(range(self.J),range(self.L)):
                hyperparam = self.lambdaGl[l] if self.ARD else self.lambdaF[j,l]
                self.G[j,l] = exponential_draw(hyperparam) if init_FG == 'random' else 1.0/hyperparam
            
        # Initialise S
        for k,l in itertools.product(range(self.K),range(self.L)):
            hyperparam = self.lambdaS[k,l] 
            self.S[k,l] = exponential_draw(hyperparam) if init_FG == 'random' else 1.0/hyperparam
        
        # Initialise tau
        self.tau = gamma_mode(self.alpha_s(),self.beta_s())


    def run(self,iterations):
        ''' Run the Gibbs sampler. '''
        self.all_F = numpy.zeros((iterations,self.I,self.K))  
        self.all_S = numpy.zeros((iterations,self.K,self.L))   
        self.all_G = numpy.zeros((iterations,self.J,self.L))  
        self.all_tau = numpy.zeros(iterations)
        self.all_lambdaFk = numpy.zeros((iterations,self.K))
        self.all_lambdaGl = numpy.zeros((iterations,self.L))
        self.all_times = [] # to plot performance against time
        
        self.all_performances = {} # for plotting convergence of metrics
        for metric in ALL_METRICS:
            self.all_performances[metric] = []
        
        time_start = time.time()
        for it in range(0,iterations):   
            # Update lambdaFk, lambdaGl
            if self.ARD:
                for k in range(self.K):
                    self.lambdaFk[k] = gamma_mode(self.alphaFk_s(k),self.betaFk_s(k))
                for l in range(self.L):
                    self.lambdaGl[l] = gamma_mode(self.alphaGl_s(l),self.betaGl_s(l))
            
            # Update F
            for k in range(0,self.K):
                tauFk = self.tauF(k)
                muFk = self.muF(tauFk,k)
                self.F[:,k] = TN_vector_mode(muFk,tauFk)
                #self.F[:,k] = numpy.maximum(self.F[:,k],minimum_TN*numpy.ones(self.I))
                
            # Update S
            for k,l in itertools.product(xrange(0,self.K),xrange(0,self.L)):
                tauSkl = self.tauS(k,l)
                muSkl = self.muS(tauSkl,k,l)
                self.S[k,l] = TN_mode(muSkl,tauSkl)
                #self.S[k,l] = max(self.S[k,l],minimum_TN)
                
            # Update G
            for l in range(0,self.L):
                tauGl = self.tauG(l)
                muGl = self.muG(tauGl,l)
                self.G[:,l] = TN_vector_mode(muGl,tauGl)
                #self.G[:,l] = numpy.maximum(self.G[:,l],minimum_TN*numpy.ones(self.J))
                
            # Update tau
            self.tau = gamma_mode(self.alpha_s(),self.beta_s())
            
            # Store draws
            self.all_F[it], self.all_S[it], self.all_G[it], self.all_tau[it] = numpy.copy(self.F), numpy.copy(self.S), numpy.copy(self.G), self.tau
            if self.ARD:
                self.all_lambdaFk[it] = numpy.copy(self.lambdaFk)
                self.all_lambdaGl[it] = numpy.copy(self.lambdaGl)
            
            # Store and print performances
            perf = self.predict_while_running()
            for metric in ALL_METRICS:
                self.all_performances[metric].append(perf[metric])
                
            print "Iteration %s. MSE: %s. R^2: %s. Rp: %s." % (it+1,perf['MSE'],perf['R^2'],perf['Rp'])
        
            # Store time taken for iteration
            time_iteration = time.time()
            self.all_times.append(time_iteration-time_start)            
            

    def triple_dot(self,M1,M2,M3):
        ''' Triple matrix multiplication: M1*M2*M3. 
            If the matrices have dimensions I,K,L,J, then the complexity of M1*(M2*M3) 
            is ~IJK, and (M1*M2)*M3 is ~IJL. So if K < L, we use the former. '''
        K,L = M2.shape
        if K < L:
            return numpy.dot(M1,numpy.dot(M2,M3))
        else:
            return numpy.dot(numpy.dot(M1,M2),M3)
        
        
    ''' Compute the parameters for the distributions we sample from. '''
    def alpha_s(self):   
        ''' alpha* for tau. '''
        return self.alpha + self.size_Omega/2.0
    
    def beta_s(self):   
        ''' beta* for tau. '''
        return self.beta + 0.5*(self.M*(self.R-self.triple_dot(self.F,self.S,self.G.T))**2).sum()
        
    def alphaFk_s(self,k):   
        ''' alphaFk* for lambdaFk. '''
        return self.alpha0 + self.I
    
    def betaFk_s(self,k):   
        ''' betak* for lambdaFk. '''
        return self.beta0 + self.F[:,k].sum()
        
    def alphaGl_s(self,l):   
        ''' alphaFk* for lambdaFk. '''
        return self.alpha0 + self.J
    
    def betaGl_s(self,l):   
        ''' betak* for lambdaFk. '''
        return self.beta0 + self.G[:,l].sum()
        
    def tauF(self,k):      
        ''' tauFk for Fk. ''' 
        return self.tau * ( self.M * numpy.dot(self.S[k],self.G.T)**2 ).sum(axis=1)
        
    def muF(self,tauFk,k):
        ''' muFk for Fk. '''
        lamb = self.lambdaFk[k] if self.ARD else self.lambdaF[:,k]
        return 1./tauFk * (-lamb + self.tau*(self.M * ( (self.R-self.triple_dot(self.F,self.S,self.G.T)+numpy.outer(self.F[:,k],numpy.dot(self.S[k],self.G.T)))*numpy.dot(self.S[k],self.G.T) )).sum(axis=1)) 
        
    def tauS(self,k,l):  
        ''' tauSkl for Skl. '''     
        return self.tau * ( self.M * numpy.outer(self.F[:,k]**2,self.G[:,l]**2) ).sum()
        
    def muS(self,tauSkl,k,l):
        ''' muSkl for Skl. '''
        return 1./tauSkl * (-self.lambdaS[k,l] + self.tau*(self.M * ( (self.R-self.triple_dot(self.F,self.S,self.G.T)+self.S[k,l]*numpy.outer(self.F[:,k],self.G[:,l]))*numpy.outer(self.F[:,k],self.G[:,l]) )).sum()) 
        
    def tauG(self,l):  
        ''' tauGl for Gl. '''     
        return self.tau * ( self.M.T * numpy.dot(self.F,self.S[:,l])**2 ).T.sum(axis=0)
        
    def muG(self,tauGl,l):
        ''' muGl for Gl. '''
        lamb = self.lambdaGl[l] if self.ARD else self.lambdaG[:,l]
        return 1./tauGl * (-lamb + self.tau*(self.M * ( (self.R-self.triple_dot(self.F,self.S,self.G.T)+numpy.outer(numpy.dot(self.F,self.S[:,l]),self.G[:,l])).T * numpy.dot(self.F,self.S[:,l]) ).T).sum(axis=0)) 
        

    def approx_expectation(self,burn_in,thinning):
        ''' Return our expectation of U, V, tau, lambdak. '''
        indices = range(burn_in,len(self.all_U),thinning)
        exp_F = numpy.array([self.all_F[i] for i in indices]).sum(axis=0) / float(len(indices))      
        exp_S = numpy.array([self.all_S[i] for i in indices]).sum(axis=0) / float(len(indices))     
        exp_G = numpy.array([self.all_G[i] for i in indices]).sum(axis=0) / float(len(indices))  
        exp_tau = sum([self.all_tau[i] for i in indices]) / float(len(indices))
        exp_lambdaFk = None if not self.ARD else sum(
            [self.all_lambdaFk[i] for i in indices]) / float(len(indices))
        exp_lambdaGl = None if not self.ARD else sum(
            [self.all_lambdaGl[i] for i in indices]) / float(len(indices))
        return (exp_F, exp_S, exp_G, exp_tau, exp_lambdaFk, exp_lambdaGl)


    def predict(self,M_pred,burn_in,thinning):
        ''' Compute the expectation of U and V, and use it to predict missing values. '''
        (exp_F,exp_S,exp_G,_,_,_) = self.approx_expectation(burn_in,thinning)
        R_pred = self.triple_dot(exp_F,exp_S,exp_G.T)
        MSE = self.compute_MSE(M_pred,self.R,R_pred)
        R2 = self.compute_R2(M_pred,self.R,R_pred)    
        Rp = self.compute_Rp(M_pred,self.R,R_pred)        
        return {'MSE':MSE,'R^2':R2,'Rp':Rp}
        
        
    def predict_while_running(self):
        ''' Predict the training error while running. '''
        R_pred = self.triple_dot(self.F,self.S,self.G.T)
        MSE = self.compute_MSE(self.M,self.R,R_pred)
        R2 = self.compute_R2(self.M,self.R,R_pred)    
        Rp = self.compute_Rp(self.M,self.R,R_pred)        
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
        
        
    def quality(self,metric,burn_in,thinning):
        ''' Return the model quality, either as log likelihood, BIC, AIC, MSE, or ELBO. '''
        assert metric in ALL_QUALITY, 'Unrecognised metric for model quality: %s.' % metric
        
        (exp_F, exp_S, exp_G, exp_tau, _, _) = self.approx_expectation(burn_in,thinning)
        log_likelihood = self.log_likelihood(exp_F, exp_S, exp_G, exp_tau)
        
        if metric == 'loglikelihood':
            return log_likelihood
        elif metric == 'BIC':
            # -2*loglikelihood + (no. free parameters * log(no data points))
            return - 2 * log_likelihood + self.number_parameters() * math.log(self.size_Omega)
        elif metric == 'AIC':
            # -2*loglikelihood + 2*no. free parameters
            return - 2 * log_likelihood + 2 * self.number_parameters()
        elif metric == 'MSE':
            R_pred = self.triple_dot(exp_F, exp_S, exp_G.T)
            return self.compute_MSE(self.M, self.R, R_pred)
        elif metric == 'ELBO':
            return 0.
        
    def log_likelihood(self, exp_F, exp_S, exp_G, exp_tau):
        ''' Return the likelihood of the data given the trained model's parameters. '''
        explogtau = math.log(exp_tau)
        return self.size_Omega / 2. * ( explogtau - math.log(2*math.pi) ) \
             - exp_tau / 2. * (self.M*( self.R - self.triple_dot(exp_F,exp_S,exp_G.T) )**2).sum()
             
    def number_parameters(self):
        ''' Return the number of free variables in the model. '''
        return (self.I*self.K + self.K*self.L + self.J*self.L + 1) + (self.K+self.L if self.ARD else 0)
    