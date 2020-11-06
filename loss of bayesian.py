from posteriors import Posterior
from finding_estimates import EstimatorClass
from scipy.stats import gamma
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

n = 3
rho = 0.5

S1 = np.random.gamma(shape=n / 2, scale=4 * (1 + rho))  # 2*n*(1+rho)
S2 = np.random.gamma(shape=n / 2, scale=4 * (1 - rho))  # 2*n*(1-rho)

T1 = 1 / 2 * (S1 + S2)
T2 = 1 / 4 * (S1 - S2)

jef = Posterior("jeffrey")
est_obj = EstimatorClass()

lim = 50

#print(gamma.ppf(0.999,n/2,0,4*(1+rho)))

est_func = est_obj.e_estimator
def S1_int_func(S1,n,S2,rho):
    T1 = 1 / 2 * (S1 + S2)
    T2 = 1 / 4 * (S1 - S2)
    x2 = est_obj.expt(jef.distribution,lambda x: x**2, n, T1,T2,jef.normalization(n,T1,T2))
    return x2*gamma.pdf(S1,a=n/2,scale=4*(1+rho))

def S2_int_func(S2,n,rho):
    return quad(S1_int_func,0,lim,args=(n,S2,rho))[0]

test = quad(S2_int_func,0,lim,args=(n,rho))
print(test)

