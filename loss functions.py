import numpy as np
import matplotlib.pyplot as plt

def kullback_leibler(rho,rho0):
    return (1-rho*rho0)/(1-rho0**2)-1/2*np.log((1-rho**2)/(1-rho0**2))-1

def fisher_information_metric(rho,rho0):
    def f(x):
        return np.sqrt(2)*np.arctanh(np.sqrt(2)*x/np.sqrt(1+x**2))-np.arcsinh(x)
    return np.abs(f(rho)-f(rho0))

n = 1000
rho0 = 0.0
rhos = np.linspace(-1+2/n,1-2/n,n)

kls = kullback_leibler(rhos,rho0)
fis = fisher_information_metric(rhos,rho0)

plt.figure()
plt.subplot(2,2,1)
plt.title(r"Kullback-Leibler($\rho,$"+str(rho0)+")")
plt.plot(rhos,kls)
plt.subplot(2,2,2)
plt.title(r"Fisher-information-metric($\rho,$"+str(rho0)+")")
plt.plot(rhos,fis)
plt.subplot(2,2,3)
plt.title(r"Kullback-Leibler$^2$($\rho,$"+str(rho0)+")")
plt.plot(rhos,kls**2)
plt.subplot(2,2,4)
plt.title(r"Fisher-information-metric$^2$($\rho,$"+str(rho0)+")")
plt.plot(rhos,fis**2)
#plt.subplot(3,2,5)
#plt.title(r"Kullback-Leibler$^{1/2}$($\rho,$"+str(rho0)+")")
#plt.plot(rhos,np.sqrt(kls))
plt.show()








