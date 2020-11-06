import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

def f(rho,alpha,A,B,limits= True):
    if limits and (rho <= -1 or rho >= 1):
        return 0
    return (1-rho**2)**(-alpha)*np.exp(-alpha/2*(A/(1+rho)+B/(1-rho)))

def jeffreys(rho,n,T1,T2):
    if rho < -1 or rho > 1:
        return 0
    temp = 1-rho**2
    return np.sqrt(1+rho**2)/((2*np.pi)**n*temp**(n/2+1))*np.exp(-T1/(2*temp)+rho*T2/temp)

def PC(rho, n, T1, T2,lam):
    if rho < -1 or rho > 1:
        return 0
    temp = 1-rho**2
    return lam*np.abs(rho)/(temp**(n/2+1)*np.sqrt(-np.log(temp)))*\
           np.exp(-T1/(2*temp)+rho*T2/temp-lam*np.sqrt(-np.log(temp)))

def uniformprior(rho, n, T1, T2):
    if rho < -1 or rho > 1:
        return 0
    temp = 1-rho**2
    return 1/((2*np.pi)**n*2*temp**(n/2))*np.exp(-T1/(2*temp)+rho*T2/temp)

def arcsine(rho, n, T1, T2):
    if rho < -1 or rho > 1:
        return 0
    temp = 1-rho**2
    return 1/((2*np.pi)**n*np.pi*temp**(n/2+1/2))*np.exp(-T1/(2*temp)+rho*T2/temp)


def quad_mean_estimate(a,T1,T2,distr):
    moment_func = lambda x: x*distr(x,a,T1,T2)
    return quad(moment_func,-1,1)[0]/quad(distr,-1,1,args=(a,T1,T2))[0]

def MSE_signle_estimator(rho,a,distr,n):
    SSE = 0
    estimates = []
    for i in range(n):
        S1 = np.random.gamma(shape=a,scale=2*(1+rho))
        S2 = np.random.gamma(shape=a,scale=2*(1-rho))
        T1 = 1/2*(S1+S2)
        T2 = 1/4*(S1-S2)

        estimate = quad_mean_estimate(a,T1,T2,distr)
        estimates.append(estimate)
        SSE = (estimate-rho)**2

    MSE = 1/len(rhos)*SSE

    return MSE,np.array(estimates)

def MSE_estimator(rhos,a,distr,n):
    MSEs = np.empty(len(rhos))
    estimates = np.empty((len(rhos),n))
    counter = 0
    for rho in rhos:
        print(counter,rho)
        MSE,all_estimates = MSE_signle_estimator(rho,a,distr,n)
        MSEs[counter] = MSE
        estimates[counter,:] = all_estimates

        counter += 1

    return MSEs,estimates



n = 10 ** 3

rhos = np.linspace(-1+1/500,1-1/500,1000)

lam = 10**(-1)
MSE,estimates = MSE_estimator(rhos,50,jeffreys,n)

plt.figure()
plt.subplot(2,1,1)
plt.scatter(rhos,MSE)
plt.subplot(2,1,2)
for i in range(n):
    plt.scatter(rhos,estimates[:,i])
plt.show()











