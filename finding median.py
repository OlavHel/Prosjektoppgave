import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.integrate import quad
from scipy.optimize import fsolve
from scipy.optimize import minimize


def normaldistr(x,mu=0,sigma2=1):
    return (2*np.pi*sigma2)**(-1/2)*np.exp(-1/(2*sigma2)*(x-mu)**2)

def f(x):
    if x <= -1 or x >= 1:
        return 0
    return np.sqrt(2)*np.arctanh(np.sqrt(2)*x/np.sqrt(1+x**2))-np.arcsinh(x)

def df(x):
    if x<= -1 or x >= 1:
        return np.infty
    return np.sqrt(1+x**2)/(1-x**2)

def uniformprior(rho, n, T1, T2):
    if type(rho)==type(np.array([0.1])) and len(rho) == 1:
        rho = rho[0]
    temp = 1-rho**2
    if type(rho) == type(np.array([])):
        rho[rho<=-1] = 0
        rho[rho>= 1] = 0
    elif (rho <= (-1) or rho >= (1)):
        return 0
    return np.exp(n/1.1)*temp**(-n/2)*np.exp(-T1/(2*temp)+rho*T2/temp)

def jeffreys(rho,n,T1,T2):
    if type(rho)==type(np.array([0.1])) and len(rho) == 1:
        rho = rho[0]
    temp = 1-rho**2
    if type(rho) == type(np.array([])):
        rho[rho<=-1] = 0
        rho[rho>= 1] = 0
    elif (rho <= (-1) or rho >= (1)):
        return 0
    return np.sqrt(1+rho**2)/((2*np.pi)**n*temp**(n/2+1))*np.exp(-T1/(2*temp)+rho*T2/temp)

def PC(rho, n, T1, T2,lam):
    if np.isnan(rho):
        print("hei")
    if type(rho)==type(np.array([0.1])) and len(rho) == 1:
        rho = rho[0]
    if type(rho) == type(np.array([])):
        rho[rho<=-1] = 0
        rho[rho>= 1] = 0
    elif (rho <= (-1) or rho >= (1)):
        return 0
    temp = 1-rho**2
    print("temp:",temp,rho)
    return lam*np.abs(rho)/(temp**(n/2+1)*np.sqrt(-np.log(temp)))* \
           np.exp(-T1/(2*temp)+rho*T2/temp-lam*np.sqrt(-np.log(temp)))

def arcsine(rho, n, T1, T2):
    if type(rho)==type(np.array([0.1])) and len(rho) == 1:
        rho = rho[0]
    if type(rho) == type(np.array([])):
        rho[rho<=-1] = 0
        rho[rho>= 1] = 0
    elif (rho <= (-1) or rho >= (1)):
        return 0
    temp = 1-rho**2
    return np.exp(n)/(temp**(n/2+1/2))*np.exp(-T1/(2*temp)+rho*T2/temp)

def kullback_leibler(rho,rho0):
#    if rho>=1 or rho <=-1 or rho0 >=1 or rho0 <=-1:
#        return 0
    return (1-rho*rho0)/(1-rho0**2)-1/2*np.log((1-rho**2)/(1-rho0**2))-1

def fisher_information(rho):
    return np.sqrt(2)*np.arctanh(np.sqrt(2)*rho/np.sqrt(1+rho**2))-np.arcsinh(rho)


def fisher_information_metric(rho,rho0):
    return np.abs(fisher_information(rho)-fisher_information(rho0))

from posteriors import Posterior

n = 3
rho = -0.2
lam = 10**(-6)

post = Posterior("jeffrey",lam=lam)
distr = post.distribution#lambda x,n,T1,T2: PC(x,n,T1,T2,lam)

S1 = np.random.gamma(shape=n/2,scale=4*(1+rho))#2*n*(1+rho)
S2 = np.random.gamma(shape=n/2,scale=4*(1-rho))#2*n*(1-rho)


T1 = 1/2*(S1+S2)
T2 = 1/4*(S1-S2)
print(S1,S2, 2*T2/T1)

#print("val:",distr(rho,n,T1,T2))

c,c_error = quad(distr,-1,1,args=(n,T1,T2))
print("c:",c,c_error)

initials = np.tanh(2*T2/T1)
print("start",initials)

xfunc = lambda x,n,T1,T2: x*distr(x,n,T1,T2)/c
Erho = quad(xfunc,-1,1,args=(n,T1,T2))[0]

initial_map = initials

MAP = minimize(lambda x,n,T1,T2: -np.log(distr(x,n,T1,T2)),initial_map,args=(n,T1,T2)).x[0]

func = lambda x,n,T1,T2: \
    quad(distr,-1,x,args=(n,T1,T2))[0]/c-1/2
median = fsolve(func,initials,args=(n,T1,T2))[0]

Ef = quad(lambda x: fisher_information(x)*distr(x,n,T1,T2)/c, -1,1)[0]
fi2 = fsolve(lambda x: Ef-fisher_information(x),np.tanh(Ef))[0]

s_time = time.time()
kl2func = lambda x,n,T1,T2,y: kullback_leibler(x,y)**(2)*distr(x,n,T1,T2)/c
minkl2func = lambda y: quad(kl2func,-1,1,args=(n,T1,T2,y))[0]
print("kl2 test",minimize(minkl2func,initials).x[0])
print("kl2 test time",time.time()-s_time)

s_time = time.time()
elog = quad(lambda x: np.log(1-x**2)*distr(x,n,T1,T2)/c,-1,1)[0]
exlog = quad(lambda x: x*np.log(1-x**2)*distr(x,n,T1,T2)/c,-1,1)[0]
Erho2 = quad(lambda x: x**2*distr(x,n,T1,T2)/c,-1,1)[0]
print(elog,exlog,Erho2)
kl2 = fsolve(lambda x:
             -x*(1-Erho*x)/(1-x**2)+1/2*x*elog-1/2*x*np.log(1-x**2)+x+
             (Erho-x*Erho2)/(1-x**2)+1/2*Erho*np.log(1-x**2)-1/2*exlog-Erho
             ,Erho)[0]
print("kl2 test time",time.time()-s_time)

kl12func = lambda x,n,T1,T2,y: np.sqrt(kullback_leibler(x,y))*distr(x,n,T1,T2)/c
minkl12func = lambda y: quad(kl12func,-1,1,args=(n,T1,T2,y))[0]
kl12 = minimize(minkl12func,initials).x[0]




#fifunc = lambda x,n,T1,T2,y: fisher_information_metric(x,y)*distr(x,n,T1,T2)
#minfifunc = lambda y: quad(fifunc,-1,1,args=(n,T1,T2,y))[0]/c
#fi = minimize(minfifunc,MAP).x[0]


print("Ef",Ef,"tanh(Ef)",np.tanh(Ef))

print("kl2",kl2)
#print("kl2 test",kl2_test)
print("E(rho)",Erho)
print("kl1/2",kl12)
print("fi2",fi2)
print("median",median)
print("map", MAP)

print("rel dist:",(fi2-median)/median)

rhos = np.linspace(-1+1/500,1-1/500,1000)

plt.figure()
plt.plot(rhos,distr(rhos,n,T1,T2)/c,label = "posterior")
plt.plot([kl2,kl2],[0,distr(kl2,n,T1,T2)/c],label = "kl2")
plt.plot([Erho,Erho],[0,distr(Erho,n,T1,T2)/c],label = "E(rho)")
#plt.plot([kl12,kl12],[0,distr(kl12,n,T1,T2)/c],label = "kl1/2")
plt.plot([fi2,fi2],[0,distr(fi2,n,T1,T2)/c],label = "fi2")
plt.plot([median,median],[0,distr(median,n,T1,T2)/c],label = "median")
plt.plot([MAP,MAP],[0,distr(MAP,n,T1,T2)/c],label = "MAP")
plt.legend()
plt.show()







