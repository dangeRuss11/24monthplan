
# Goal: Estimate P(X > 4) where X ~ N(0,1)

# Part 1: Naive Monte Carlo
# - Draw N samples from N(0,1)
# - Estimate probability as fraction exceeding 4
# - Compute estimate and standard error for N = 1000, 10000, 100000

# Part 2: Importance Sampling
# - Use proposal distribution N(4, 1)
# - Implement importance weights: w(x) = p(x) / q(x) where p is N(0,1), q is N(4,1)
# - Compute estimate and standard error for same N values

# Part 3: Comparison
# - Print a table comparing estimates and standard errors
# - Add a comment explaining why importance sampling reduces variance here



import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import norm



# Part 1:

ns = [1000,10000,100000]

for n in ns:
    random_samples = np.random.normal(0 ,1, n)

    p_indicator = np.sum(random_samples > 4)

    prob = (p_indicator / n)
    #print(f'the probability of X_i > 4 = {prob} ')

    # SE - spread of estimated probabiity (prob) based off of your random samples
    # Formula i googled
    standard_error = np.sqrt((prob*(1-prob) / n))

    print(f'simple prob for {n}: {prob}, and SE : {standard_error}')


    # incorporate importance sampling
    helper_samples = np.random.normal(4,1, n)

    q_indicator = helper_samples > 4

    # p(x) / q(x) are pfd of each indvidual sample, not the value of the sample
    p_x = norm.pdf(helper_samples, 0 , 1)
    q_x = norm.pdf(helper_samples, 4, 1)

    w_x = p_x / q_x

    weighted_prob = np.mean(w_x * q_indicator)
    weighted_standard_error = np.sqrt((np.var(w_x * q_indicator)) / n)
    
    print(f'weighted prob for {n}: {weighted_prob}, and weighted SE : {weighted_standard_error}')

    # reduces variance because samples more of the important region, thus estimator is more stable







