"""
Recover the toy dataset using ICM. 

Measure the convergence over iterations and time.
We run the algorithm 10 times with the same seed, and take the average timestamps.
"""

project_location = "/Users/thomasbrouwer/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

from BNMTF_ARD.code.models.nmf_icm import nmf_icm

import numpy
import scipy
import random
import matplotlib.pyplot as plt


''' Location of toy data, and where to store the performances. '''
input_folder = project_location+"BNMTF_ARD/data/toy/bnmf/"
output_folder = project_location+"BNMTF_ARD/experiments/experiments_toy/convergence/results/"
output_file_performances = output_folder+'nmf_icm_all_performances.txt'
output_file_times = output_folder+'nmf_icm_all_times.txt'


''' Model settings. '''
iterations = 2000

init_UV = 'random'
I, J, K = 100, 80, 10
ARD = False
repeats = 10

lambdaU = 0.1
lambdaV = 0.1
alphatau, betatau = 1., 1.
alpha0, beta0 = 1., 1.
hyperparams = { 'alphatau':alphatau, 'betatau':betatau, 'alpha0':alpha0, 'beta0':beta0, 'lambdaU':lambdaU, 'lambdaV':lambdaV }


''' Load in data. '''
R = numpy.loadtxt(input_folder+"R.txt")
M = numpy.ones((I,J))


''' Run the algorithm, :repeats times, and average the timestamps. '''
times_repeats = []
performances_repeats = []
for i in range(0,repeats):
    # Set all the seeds
    numpy.random.seed(0), random.seed(0), scipy.random.seed(0)
    
    # Run the classifier
    BNMF = nmf_icm(R,M,K,ARD,hyperparams) 
    BNMF.initialise(init_UV)
    BNMF.run(iterations)

    # Extract the performances and timestamps across all iterations
    times_repeats.append(BNMF.all_times)
    performances_repeats.append(BNMF.all_performances)


''' Check whether seed worked: all performances should be the same. '''
assert all([numpy.array_equal(p, performances_repeats[0]) for p in performances_repeats]), \
    "Seed went wrong - performances not the same across repeats!"


''' Print out the performances, and the average times, and store them in a file. '''
all_times_average = list(numpy.average(times_repeats, axis=0))
all_performances = performances_repeats[0]
print "all_times_average = %s" % all_times_average
print "all_performances = %s" % all_performances
open(output_file_times,'w').write("%s" % all_times_average)
open(output_file_performances,'w').write("%s" % all_performances)


''' Plot the average time plot, and performance vs iterations. '''
plt.figure()
plt.title("Performance against average time")
plt.plot(all_times_average, all_performances['MSE'])
plt.ylim(0,10)

plt.figure()
plt.title("Performance against iteration")
plt.plot(all_performances['MSE'])
plt.ylim(0,10)