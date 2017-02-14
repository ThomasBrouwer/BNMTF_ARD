"""
Non-probabilistic non-negative matrix tri-factorisation, as presented in
"Probabilistic Matrix Tri-Factorisation" (Yoo and Choi, 2009).

We change the notation to match ours: R = FSG.T instead of V = USV.T.

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
- K, the number of row latent factors
- L, the number of column latent factors
    
Initialisation can be done by running the initialise(init,tauUV) function. We initialise as follows:
- init_FG = 'ones'          -> F[i,k] = G[j,k] = 1
          = 'random'        -> F[i,k] ~ U(0,1), G[j,l] ~ G(0,1), 
          = 'exponential'   -> F[i,k] ~ Exp(expo_prior), G[j,l] ~ Exp(expo_prior) 
          = 'kmeans'        -> F = KMeans(R,rows)+0.2, G = KMeans(R,columns)+0.2
  where expo_prior is an additional parameter (default 1)
- init_S = 'ones'          -> S[i,k] = 1
         = 'random'        -> S[i,k] ~ U(0,1)
         = 'exponential'   -> S[i,k] ~ Exp(expo_prior)
"""

from kmeans.kmeans import KMeans
from distributions.exponential import exponential_draw

import numpy,itertools,math,time

ALL_METRICS = ['MSE','R^2','Rp']
OPTIONS_INIT_FG = ['kmeans', 'ones', 'random', 'exponential']
OPTIONS_INIT_S = ['ones', 'random', 'exponential']

class NMTF:
    def __init__(self,R,M,K,L):
        ''' Set up the class and do some checks on the values passed. '''
        self.R = numpy.array(R,dtype=float)
        self.M = numpy.array(M,dtype=float)
        self.K = K            
        self.L = L    
        
        self.metrics = ['MSE','R^2','Rp']
                
        assert len(self.R.shape) == 2, "Input matrix R is not a two-dimensional array, " \
            "but instead %s-dimensional." % len(self.R.shape)
        assert self.R.shape == self.M.shape, "Input matrix R is not of the same size as " \
            "the indicator matrix M: %s and %s respectively." % (self.R.shape,self.M.shape)
        
        (self.I,self.J) = self.R.shape
        
        self.check_empty_rows_columns() 
        
        # For computing the I-div it is better if unknown values are 1's, not 0's, to avoid numerical issues
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
        
        
    def train(self,iterations,init_FG='random',init_S='random',expo_prior=1.):
        ''' Initialise and run the algorithm. '''
        self.initialise(init_FG=init_FG, init_S=init_S, expo_prior=expo_prior) 
        self.run(iterations=iterations)     



    def initialise(self,init_FG='random',init_S='random',expo_prior=1.):
        ''' Initialise F, S and G. '''
        assert init_FG in OPTIONS_INIT_FG, "Unrecognised init option for F,G: %s. Should be one in %s." % (init_FG, OPTIONS_INIT_FG)
        assert init_S in OPTIONS_INIT_S, "Unrecognised init option for S: %s. Should be one in %s." % (init_S, OPTIONS_INIT_S)
        
        if init_S == 'ones':
            self.S = numpy.ones((self.K,self.L))
        elif init_S == 'random':
            self.S = numpy.random.rand(self.K,self.L)
        elif init_S == 'exponential':
            self.S = numpy.empty((self.K,self.L))
            for k,l in itertools.product(xrange(0,self.K),xrange(0,self.L)):        
                self.S[k,l] = exponential_draw(expo_prior)
        
        if init_FG == 'ones':
            self.F = numpy.ones((self.I,self.K))
            self.G = numpy.ones((self.J,self.L))
        elif init_FG == 'random':
            self.F = numpy.random.rand(self.I,self.K)
            self.G = numpy.random.rand(self.J,self.L)
        elif init_FG == 'exponential':
            self.F = numpy.empty((self.I,self.K))
            self.G = numpy.empty((self.J,self.L))
            for i,k in itertools.product(xrange(0,self.I),xrange(0,self.K)):        
                self.F[i,k] = exponential_draw(expo_prior)
            for j,l in itertools.product(xrange(0,self.J),xrange(0,self.L)):
                self.G[j,l] = exponential_draw(expo_prior)
        elif init_FG == 'kmeans':
            print "Initialising F using KMeans."
            kmeans_F = KMeans(self.R,self.M,self.K)
            kmeans_F.initialise()
            kmeans_F.cluster()
            self.F = kmeans_F.clustering_results + 0.2            
            
            print "Initialising G using KMeans."
            kmeans_G = KMeans(self.R.T,self.M.T,self.L)   
            kmeans_G.initialise()
            kmeans_G.cluster()
            self.G = kmeans_G.clustering_results + 0.2
        
        
    def run(self,iterations):
        ''' Run the algorithm. '''
        assert hasattr(self,'F') and hasattr(self,'S') and hasattr(self,'G'), \
            "F, S and G have not been initialised - please run NMTF.initialise() first."        
        
        self.all_times = [] # to plot performance against time
        self.all_performances = {} # for plotting convergence of metrics
        for metric in ALL_METRICS:
            self.all_performances[metric] = []
            
        time_start = time.time()
        for it in range(1,iterations+1):
            for k in range(self.K):
                self.update_F(k)
                
            for k,l in itertools.product(range(self.K),range(self.L)):
                self.update_S(k,l)
                    
            for l in range(self.L):
                self.update_G(l)
               
            self.give_update(it)
            
            time_iteration = time.time()
            self.all_times.append(time_iteration-time_start)  
        
                
    ''' Updates for F, G, S. '''             
    def triple_dot(self,M1,M2,M3):
        ''' Triple matrix multiplication: M1*M2*M3. 
            If the matrices have dimensions I,K,L,J, then the complexity of M1*(M2*M3) 
            is ~IJK, and (M1*M2)*M3 is ~IJL. So if K < L, we use the former. '''
        K,L = M2.shape
        if K < L:
            return numpy.dot(M1,numpy.dot(M2,M3))
        else:
            return numpy.dot(numpy.dot(M1,M2),M3)
        
    def update_F(self,k):
        ''' Update values for F. '''
        R_pred = self.triple_dot(self.F,self.S,self.G.T)
        SG = numpy.dot(self.S[k],self.G.T)
        numerator = (self.M * self.R / R_pred * SG).sum(axis=1)
        denominator = (self.M * SG).sum(axis=1)
        self.F[:,k] = self.F[:,k] * numerator / denominator
        
    def update_G(self,l):
        ''' Update values for G. '''
        R_pred = self.triple_dot(self.F,self.S,self.G.T)
        FS = numpy.dot(self.F,self.S[:,l])
        numerator = ((self.M * self.R / R_pred).T * FS).T.sum(axis=0)
        denominator = (self.M.T * FS).T.sum(axis=0)
        self.G[:,l] = self.G[:,l] * numerator / denominator
        
    def update_S(self,k,l):
        ''' Update values for S. '''
        R_pred = self.triple_dot(self.F,self.S,self.G.T)
        F_times_G = self.M * numpy.outer(self.F[:,k], self.G[:,l])   
        numerator = (self.R * F_times_G / R_pred).sum()
        denominator = F_times_G.sum()
        self.S[k,l] = self.S[k,l] * numerator / denominator
           
           
    def predict(self,M_pred):
        ''' Predict missing values in R. '''
        R_pred = self.triple_dot(self.F,self.S,self.G.T)
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
        R_pred = self.triple_dot(self.F,self.S,self.G.T)
        return (self.M * ( self.R_excl_unknown * numpy.log( self.R_excl_unknown / R_pred ) - self.R_excl_unknown + R_pred ) ).sum()        
        
        
    def give_update(self,iteration):    
        ''' Print and store the I-divergence and performances. '''
        perf = self.predict(self.M)
        i_div = self.compute_I_div()
        
        for metric in self.metrics:
            self.all_performances[metric].append(perf[metric])
               
        print "Iteration %s. I-divergence: %s. MSE: %s. R^2: %s. Rp: %s." % (iteration,i_div,perf['MSE'],perf['R^2'],perf['Rp'])
