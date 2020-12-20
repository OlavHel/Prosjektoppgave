import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.integrate import quad
from scipy.optimize import fsolve
from scipy.optimize import minimize

### FOR TESTING ESTIMATIONS OF BAYESIAN ESTIMATORS ##

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
from finding_estimates import EstimatorClass

est_class = EstimatorClass()

n = 10
rho = 0.8
lam = 10**(-6)

post = Posterior("arcsine",lam=lam)
distr = post.distribution#lambda x,n,T1,T2: PC(x,n,T1,T2,lam)

#S1 = np.random.gamma(shape=n/2,scale=4*(1+rho))#2*n*(1+rho)
#S2 = np.random.gamma(shape=n/2,scale=4*(1-rho))#2*n*(1-rho)

data = np.random.multivariate_normal(
    np.array([0, 0]),
    np.array([[1, rho], [rho, 1]]),
    size=n
)
Xs = data[:, 0]
Ys = data[:, 1]
S1 = np.sum((Xs+Ys)**2)
S2 = np.sum((Xs-Ys)**2)

T1 = 1/2*(S1+S2)
T2 = 1/4*(S1-S2)
print(S1,S2, 2*T2/T1)

#print("val:",distr(rho,n,T1,T2))

MLE = est_class.mle(Xs,Ys,n)
samp = est_class.sample_corr(Xs,Ys,n)
samp_var = est_class.c_sample_corr(Xs,Ys,n)
samp_var_trunc = est_class.t_c_sample_corr(Xs,Ys,n)

c,c_error = quad(distr,-1,1,args=(n,T1,T2))
print("c:",c,c_error)

initials = 2*T2/T1#np.tanh(2*T2/T1)
if initials >= 0.97:
    initials = 0.97
elif initials <= -0.97:
    initials = -0.97
print("start",initials)

xfunc = lambda x,n,T1,T2: x*distr(x,n,T1,T2)/c
Erho = quad(xfunc,-1,1,args=(n,T1,T2))[0]

def map_func(x,n,T1,T2):
    if x[0]<=-1 or x[0]>=1:
        return np.infty
    temp = distr(x,n,T1,T2)/c
    if temp < 10**(-5):
        return np.infty
    return -temp#np.log(temp)

t_start = time.time()
temp_rhos = np.linspace(-1+1/100,1-1/100,50)
temp = distr(temp_rhos,n,T1,T2)/c
initial_map = temp_rhos[np.argmax(temp)]
initial_map = initials
print("map start",initial_map)
MAP = minimize(map_func,initial_map,args=(n,T1,T2)).x[0]
print("MAP time:",time.time()-t_start)


t_start = time.time()
temp_rhos = np.linspace(-1+1/100,1-1/100,100)
temp = distr(temp_rhos,n,T1,T2)/c
temp2 = np.cumsum(temp)
temp3 = temp2/temp2[-1]
initial_med = temp_rhos[np.argmin(np.abs(temp3-1/2))]
print("med start",initial_med)
def func(x,n,T1,T2):
    if x[0]<=-1 or x[0]>=1:
        return np.infty
    return quad(distr,-1,x,args=(n,T1,T2))[0]-1/2*c

median = fsolve(func,initial_med,args=(n,T1,T2))[0]
print("med time",time.time()-t_start)

Ef = quad(lambda x: fisher_information(x)*distr(x,n,T1,T2)/c, -1,1)[0]
fi2 = fsolve(lambda x: Ef-fisher_information(x),np.tanh(Ef))[0]
print("med i fi2",Ef-fisher_information(median),"fi2 i fi2",Ef-fisher_information(fi2))

#s_time = time.time()
#kl2func = lambda x,n,T1,T2,y: kullback_leibler(x,y)**(2)*distr(x,n,T1,T2)/c
#minkl2func = lambda y: quad(kl2func,-1,1,args=(n,T1,T2,y))[0]
#print("kl2 test",minimize(minkl2func,initials).x[0])
#print("kl2 test time",time.time()-s_time)

s_time = time.time()
elog = quad(lambda x: np.log(1-x**2)*distr(x,n,T1,T2)/c,-1,1)[0]
exlog = quad(lambda x: x*np.log(1-x**2)*distr(x,n,T1,T2)/c,-1,1)[0]
Erho2 = quad(lambda x: x**2*distr(x,n,T1,T2)/c,-1,1)[0]
print(elog,exlog,Erho2)
def kl2_func(x):
    return -x * (1 - Erho * x) / (1 - x ** 2) + 1 / 2 * x * elog - 1 / 2 * x * np.log(1 - x ** 2) + x +\
    (Erho - x * Erho2) / (1 - x ** 2) + 1 / 2 * Erho * np.log(1 - x ** 2) - 1 / 2 * exlog - Erho
kl2 = fsolve(kl2_func,initials)[0]
print("kl2 test time",time.time()-s_time)

#kl12func = lambda x,n,T1,T2,y: np.sqrt(kullback_leibler(x,y))*distr(x,n,T1,T2)/c
#minkl12func = lambda y: quad(kl12func,-1,1,args=(n,T1,T2,y))[0]
#kl12 = minimize(minkl12func,initials).x[0]




#fifunc = lambda x,n,T1,T2,y: fisher_information_metric(x,y)*distr(x,n,T1,T2)
#minfifunc = lambda y: quad(fifunc,-1,1,args=(n,T1,T2,y))[0]/c
#fi = minimize(minfifunc,MAP).x[0]


print("Ef",Ef,"tanh(Ef)",np.tanh(Ef))

print("kl2",kl2)
print("kl2 val",kl2_func(kl2))
#print("kl2 test",kl2_test)
print("E(rho)",Erho)
#print("kl1/2",kl12)
print("fi2",fi2)
print("median",median)
print("med val",quad(distr,-1,median,args=(n,T1,T2))[0]/c)
print("map", MAP)
print("MLE",MLE)
print("samp corr",samp)
print("Samc corr var",samp_var)

print("rel dist:",(fi2-median)/median)

rhos = np.linspace(-1+1/500,1-1/500,1000)

plt.figure()
plt.title(r"$n=$"+str(n)+r" $S_1$="+str(np.round(S1,2))+r" $S_2$="+str(np.round(S2,2)))
plt.plot(rhos,distr(rhos,n,T1,T2)/c,label = "posterior")
#plt.plot([kl2,kl2],[0,distr(kl2,n,T1,T2)/c],label = "kl2")
plt.plot([Erho,Erho],[0,distr(Erho,n,T1,T2)/c],label = "E(rho)")
#plt.plot([kl12,kl12],[0,distr(kl12,n,T1,T2)/c],label = "kl1/2")
plt.plot([fi2,fi2],[0,distr(fi2,n,T1,T2)/c],label = "fi2")
plt.plot([median,median],[0,distr(median,n,T1,T2)/c],label = "median")
plt.plot([MAP,MAP],[0,distr(MAP,n,T1,T2)/c],label = "MAP")
plt.plot([MLE,MLE],[0,distr(MLE,n,T1,T2)/c],label = "MLE")
plt.plot([samp,samp],[0,distr(samp,n,T1,T2)/c],label = "Sample corr")
if np.abs(samp_var) >=1:
    plt.plot([samp_var,samp_var],[0,distr(MAP,n,T1,T2)/c],label = "C Sample corr")
else:
    plt.plot([samp_var,samp_var],[0,distr(samp_var,n,T1,T2)/c],label = "C Sample corr")
plt.legend()
plt.show()






