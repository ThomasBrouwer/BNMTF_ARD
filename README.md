# BNMTF_ARD
Bayesian Non-negative Matrix Tri-Factorisation with Automatic Relevance Determination

Implementations of BNMTF-ARD model. Inference using Gibbs sampling or variational Bayes.

We have three variations, defined by the prior for S. The main model is:
- We decompose a given matrix R into matrices F, S, G, s.t. R = F*S*G.T
- The mask matrix M indicates the observed entries in R (Mij = 1 if Rij is observed, 0 otherwise)
- The model assumptions are: 
  - Gaussian noise/likelihood
  - Exponential priors over entries in F, with parameter lambdaFk shared per factor k (ARD per row factor)
  - Exponential priors over entries in G, with parameter lambdaGl shared per factor l (ARD per column factor)
  - The lambda parameters have common Gamma prior with parameters alpha0,beta0
- The model is conjugate: posteriors are truncated normal distributions.

The variations are:
1. Exponential priors over individual entries in S, with parameter lambdaSkl (element-wise ARD)
2. Exponential priors over individual entries in S, with parameter lambdaFk*lambdaGl (shared ARD)
3. Exponential priors over individual entries in S, with parameter lambdaS (no ARD)
We found that model TODO gives the best sparsity:
- In model 1 it is kind of possible to find the active factors k and l, but not super obvious (1/lambda of 0.4 for active versus 0.2 for inactive).
- TODO
- TODO

The model automatically chooses the correct number of components K (rows) and L (columns), by pushing unused components towards 0:
- If sum_i Fik = 0, the posterior over lambdaFk has a peak at a very high value, so lambdaFk gets a high value.
- The posterior of the Fik (i=1..I) is TN(mu,tau), and the mu parameter has a term -lambdaFk.
- So if lambdaFk is high, mu is very negative, resulting in a posterior that is approximately exponential with parameter mu*sqrt(tau).
- The mean of that posterior is then 1/(mu*sqrt(tau)) = 0.

Usage:
TODO
