import numpy as np
import matplotlib.pyplot as plt

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

def one_simulation(a,T1,T2,distr,sigma2,n_samples):
    data = np.empty(n_samples)
    norms = np.random.normal(0, sigma2, n_samples)
    unifs = np.random.uniform(0, 1, n_samples)
    data[0] = norms[0]

    for i in range(1, n_samples):
        y = norms[i] + data[i - 1]
        limit = min(1, distr(y, a, T1, T2) / distr(data[i - 1], a, T1, T2))
        if unifs[i] < limit:
            data[i] = y
        else:
            data[i] = data[i - 1]
    return data

def MCMC_mean_estimate(a,T1,T2,distr,sigma2,n_samples):
    data = one_simulation(a,T1,T2,distr,sigma2,n_samples)
    return np.mean(data)

def MSE_estimator(rhos,a,distr,sigma2,n_samples):
    SSE = 0
    estimates = []
    counter = 1
    for rho in rhos:
        print(counter,rho)
        S1 = np.random.gamma(shape=a,scale=2*(1+rho))
        S2 = np.random.gamma(shape=a,scale=2*(1-rho))
        T1 = 1/2*(S1+S2)
        T2 = 1/4*(S1-S2)

        estimate = MCMC_mean_estimate(a,T1,T2,distr,sigma2,n_samples)
        estimates.append(estimate)
        SSE = (estimate-rho)**2
        counter += 1

    MSE = 1/len(rhos)*SSE

    return MSE,np.array(estimates)

n = 10 ** 5
sigma2 = 0.1

rhos = np.random.uniform(-1,1,1000)

MSE, estimates = MSE_estimator(rhos,5,jeffreys,sigma2,n)

plt.figure()
plt.plot([-1,1],[-1,1])
plt.scatter(rhos,estimates)
plt.show()

if False:
    rho = 0.0

    a = 10
    A = 2*a*(1+rho)
    B = 2*a*(1-rho)

    T1 = 1/2*(A+B)
    T2 = 1/4*(A-B)
    lam = 10**(-1)

    n = 10**5
    sigma2 = 0.1

    priors = [jeffreys,
              lambda x, a, T1, T2: PC(x,a,T1,T2,lam),
              uniformprior,
              arcsine]
    data = np.empty((len(priors),n))
    norms = np.empty((len(priors),n))
    unifs = np.empty((len(priors),n))

    for i in range(len(priors)):
        data[i] = np.empty(n)
        norms[i] = np.random.normal(0,sigma2,n)
        unifs[i] = np.random.uniform(0,1,n)
        data[i,0] = norms[i,0]


    for i in range(1,n):
        for j in range(len(priors)):
            y = norms[j,i] + data[j,i-1]
            limit = min(1,priors[j](y,a,T1,T2)/priors[j](data[j,i-1],a,T1,T2))
            if unifs[j,i] < limit:
                data[j,i] = y
            else:
                data[j,i] = data[j,i-1]


    rhos = np.linspace(-1+1/n,1-1/n,n)

    for i in range(len(priors)):
        print(str(priors[i]),np.mean(data[i,:]))

    plt.figure()
    for i in range(len(priors)):
        plt.subplot(len(priors),1,(i+1))
        plt.hist(data[i,1000:],bins=50, density=True)
    plt.show()









