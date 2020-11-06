import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

def graph(rho,a,A,B):
    return (1-rho**2)**(-a)*np.exp(-a/2*(A/(1+rho)+B/(1-rho)))

def areas_of_a(alphas,A,B,rho):
    areas = []
    for a in alphas:
        areas.append(integrate.quad(graph,-1,1,args=(a,A,B)))
    return np.array(areas)

n = 1000
rhos = np.linspace(-1+1/n,1-1/n,n)

print(np.max(-np.log(1-rhos**2)*(1-rhos**2)))

#plt.figure()
#plt.plot(rhos,-np.log(1-rhos**2)*(1-rhos**2))
#plt.plot(rhos,np.log(1/(1-rhos**2)),label="uten A")
#plt.plot(rhos,1/(1-rhos**2),label="med A")
#plt.legend()
#plt.show()

a = 4
A = 1#1/np.exp(1)
B = 1#A
C = A+B
D = B-A

curve = graph(rhos,a,A,B)

alphas = np.linspace(0.0001,10,100)

areas = areas_of_a(alphas,A,B,rhos)

print(areas)
plt.figure()
plt.plot(alphas,areas[:,0])
plt.show()









