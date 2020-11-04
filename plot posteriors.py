import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad


def jeffreys(rho,T1,T2,n):
    if type(rho)==type(0.1):
        if rho >= 1 or rho <= -1:
            return 0
    temp = 1-rho**2
    return np.sqrt(1+rho**2)/((2*np.pi)**n*temp**(n/2+1))*np.exp(-T1/(2*temp)+rho*T2/temp)

def PC(rho, lam, T1, T2, n):
    if type(rho)==type(0.1):
        if rho >= (1) or rho <= (-1):
            return 0
        elif rho == 0:
            return lam
    temp = 1-rho**2
    return lam*np.abs(rho)/(temp**(n/2+1)*np.sqrt(-np.log(temp)))*\
           np.exp(-T1/(2*temp)+rho*T2/temp-lam*np.sqrt(-np.log(temp)))

def uniform(rho,T1,T2,n):
    if type(rho)==type(0.1):
        if rho >= 1 or rho <= -1:
            return 0
    temp = 1-rho**2
    return 1/(temp**(n/2))*np.exp(-T1/(2*temp)+rho*T2/temp)

def arcsine(rho, T1, T2, n):
    if type(rho)==type(0.1):
        if rho >= 1 or rho <= -1:
            return 0
    temp = 1-rho**2
    return 1/(np.pi*temp**(n/2+1/2))*np.exp(-T1/(2*temp)+rho*T2/temp)

lam = 5.36

n = 5
rho = 0.5
S1 = 2*n*(1+rho)
S2 = 2*n*(1-rho)

T1 = 1/2*(S1+S2)
T2 = 1/4*(S1-S2)

n_rhos = 100
rhos = np.linspace(-1+1/n_rhos,1-1/n_rhos,n_rhos)

j_p = jeffreys(rhos,T1,T2,n)
pc_p = PC(rhos,lam,T1,T2,n)
u_p = uniform(rhos,T1,T2,n)
as_p = arcsine(rhos,T1,T2,n)

print(quad(jeffreys,-1,1,args=(T1,T2,n)))
j_p /= quad(jeffreys,-1,1,args=(T1,T2,n))[0]
print(quad(PC,-1,1,args=(lam,T1,T2,n)))
pc_p /= quad(PC,-1,1,args=(lam,T1,T2,n))[0]
print(quad(uniform,-1,1,args=(T1,T2,n)))
u_p /= quad(uniform,-1,1,args=(T1,T2,n))[0]
print(quad(arcsine,-1,1,args=(T1,T2,n)))
as_p /= quad(arcsine,-1,1,args=(T1,T2,n))[0]


plt.figure()
plt.title(r"$S_1$")
plt.plot(rhos,j_p,label = "jeffreys")
plt.plot(rhos,pc_p, label = r"PC: $\lambda=$"+str(lam))
plt.plot(rhos,u_p, label = "uniform")
plt.plot(rhos,as_p, label = "arcsine")
plt.legend(loc=2)
plt.show()


