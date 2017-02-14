"""
Non-probabilistic non-negative matrix factorisation, as presented in
"Algorithms for Non-negative Matrix Factorization" (Lee and Seung, 2001).

We change the notation to match ours: R = UV.T instead of V = WH.

The updates are then:
- Uik <- Uik * (sum_j Vjk * Rij / (Ui dot Vj)) / (sum_j Vjk)
- Vjk <- Vjk * (sum_i Uik * Rij / (Ui dot Vj)) / (sum_i Uik)
Or more efficiently using matrix operations:
- Uik <- Uik * (Mi dot [V.k * Ri / (Ui dot V.T)]) / (Mi dot V.k)
- Vjk <- Vjk * (M.j dot [U.k * R.j / (U dot Vj)]) / (M.j dot U.k)
And realising that elements in each column in U and V are independent:
- U.k <- U.k * sum(M * [V.k * (R / (U dot V.T))], axis=1) / sum(M dot V.k, axis=1)
- V.k <- V.k * sum(M * [U.k * (R / (U dot V.T))], axis=0) / sum(M dot U.k, axis=0)

We expect the following arguments:
- R, the matrix
- M, the mask matrix indicating observed values (1) and unobserved ones (0)
- K, the number of latent factors
    
Initialisation can be done by running the initialise(init,tauUV) function. We initialise as follows:
- init_UV = 'ones'        -> U[i,k] = V[j,k] = 1
          = 'random'      -> U[i,k] ~ U(0,1), V[j,k] ~ U(0,1), 
          = 'exponential' -> U[i,k] ~ Exp(expo_prior), V[j,k] ~ Exp(expo_prior) 
  where expo_prior is an additional parameter (default 1).
"""

from distributions.exponential import exponential_draw

import numpy, math, itertools, time

ALL_METRICS = ['MSE','R^2','Rp']
OPTIONS_INIT_UV = ['ones', 'random', 'exponential']

class NMF:
    def __init__(self,R,M,K):
        ''' Set up the class and do some checks on the values passed. '''
        self.R = numpy.array(R,dtype=float)
        self.M = numpy.array(M,dtype=float)
        self.K = K                     
        
        self.metrics = ['MSE','R^2','Rp']
                
        assert len(self.R.shape) == 2, "Input matrix R is not a two-dimensional array, " \
            "but instead %s-dimensional." % len(self.R.shape)
        assert self.R.shape == self.M.shape, "Input matrix R is not of the same size as " \
            "the indicator matrix M: %s and %s respectively." % (self.R.shape,self.M.shape)
            
        (self.I,self.J) = self.R.shape
        
        self.check_empty_rows_columns() 
        
        # For computing the I-div it is easier if unknown values are 1's, not 0's, to avoid numerical issues
        self.R_excl_unknown = numpy.empty((self.I,self.J))
        for i,j in itertools.product(range(0,self.I),range(0,self.J)):
            self.R_excl_unknown[i,j] = self.R[i,j] if self.M[i,j] else 1.
                 
      
    def check_empty_rows_columns(self):
        ''' Raise an exception if an entire row or column is empty. '''
        sums_columns = self.M.sum(axis=0)
        sums_rows = self.M.sum(axis=1)
                    
        # Assert none of the rows or columns are entirely unknown values
        for i,c in enumerate(sums_rows):
            assert c != 0, "Fully unobserved row in R, row %s." % i
        for j,c in enumerate(sums_columns):
            assert c != 0, "Fully unobserved column in R, column %s." % j
        
        
    def train(self,iterations,init_UV='random',expo_prior=1.):
        ''' Initialise and run the algorithm. '''
        self.initialise(init_UV=init_UV,expo_prior=expo_prior) 
        self.run(iterations=iterations)     


    def initialise(self,init_UV='random',expo_prior=1.):
        ''' Initialise U and V. '''
        assert init_UV in OPTIONS_INIT_UV, "Unrecognised init option for U,V: %s. Should be one in %s." % (init_UV, OPTIONS_INIT_UV)
        
        if init_UV == 'ones':
            self.U = numpy.ones((self.I,self.K))
            self.V = numpy.ones((self.J,self.K))
        elif init_UV == 'random':
            self.U = numpy.random.rand(self.I,self.K)
            self.V = numpy.random.rand(self.J,self.K)
        elif init_UV == 'exponential':
            self.U = numpy.empty((self.I,self.K))
            self.V = numpy.empty((self.J,self.K))
            for i,k in itertools.product(xrange(0,self.I),xrange(0,self.K)):        
                self.U[i,k] = exponential_draw(expo_prior)
            for j,k in itertools.product(xrange(0,self.J),xrange(0,self.K)):
                self.V[j,k] = exponential_draw(expo_prior)
    
    
    def run(self,iterations):
        ''' Run the algorithm. '''
        assert hasattr(self,'U') and hasattr(self,'V'), "U and V have not been initialised - please run NMF.initialise() first."        
        
        self.all_times = [] # to plot performance against time
        self.all_performances = {} # for plotting convergence of metrics
        for metric in ALL_METRICS:
            self.all_performances[metric] = []
            
        time_start = time.time()
        for it in range(1,iterations+1):
            for k in range(self.K):
                self.update_U(k)
            for k in range(self.K):
                self.update_V(k)
            
            self.give_update(it)
            
            time_iteration = time.time()
            self.all_times.append(time_iteration-time_start)       
            

    ''' Updates for U and V. '''
    def update_U(self,k):
        ''' Update values for U. '''
        self.U[:,k] = self.U[:,k] * (self.M * (self.V[:,k] * ( self.R / numpy.dot(self.U,self.V.T) ) )).sum(axis=1) / (self.M * self.V[:,k]).sum(axis=1)
        
    def update_V(self,k):
        ''' Update values for V. '''
        self.V[:,k] = self.V[:,k] * ( (self.U[:,k] * ( self.R / numpy.dot(self.U,self.V.T) ).T ).T * self.M ).sum(axis=0) / (self.U[:,k] * self.M.T).T.sum(axis=0)
        
        
    def predict(self,M_pred):
        ''' Predict missing values in R. '''
        R_pred = numpy.dot(self.U,self.V.T)
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
        
        
    def compute_I_div(self):  
        ''' Return the I-divergence. '''  
        R_pred = numpy.dot(self.U, self.V.T)
        return (self.M * ( self.R_excl_unknown * numpy.log( self.R_excl_unknown / R_pred ) - self.R_excl_unknown + R_pred ) ).sum()        
        
        
    def give_update(self,iteration):    
        ''' Print and store the I-divergence and performances. '''
        perf = self.predict(self.M)
        i_div = self.compute_I_div()
        
        for metric in ALL_METRICS:
            self.all_performances[metric].append(perf[metric])
               
        print "Iteration %s. I-divergence: %s. MSE: %s. R^2: %s. Rp: %s." % (iteration,i_div,perf['MSE'],perf['R^2'],perf['Rp'])
