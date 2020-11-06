import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

def jeffreys(rho,T1,T2,n):
    temp = 1-rho**2
    return np.sqrt(1+rho**2)/((2*np.pi)**n*temp**(n/2+1))*np.exp(-T1/(2*temp)+rho*T2/temp)

def lnjeffreys(rho,T1,T2,n):
    temp = 1-rho**2
    return 1/2*np.log(1+rho**2)-n*np.log(2*np.pi)-(n/2+1)*np.log(temp)-T1/(2*temp)+rho*T2/temp

def PC(lam, rho, T1, T2, n):
    temp = 1-rho**2
    return lam*np.abs(rho)/(temp**(n/2+1)*np.sqrt(-np.log(temp)))*\
           np.exp(-T1/(2*temp)+rho*T2/temp-lam*np.sqrt(-np.log(temp)))

def lnPC(lam, rho, T1, T2, n):
    temp = 1-rho**2
    return np.log(lam)+np.log(np.abs(rho))-(n/2+1)*np.log(temp)-1/2*np.log(-np.log(temp))\
           -T1/(2*temp)+rho*T2/temp-lam*np.sqrt(-np.log(temp))

def ratio(lam, rho):
    temp = np.sqrt(-np.log(1-rho**2))
    return lam*np.abs(rho)/(temp*np.sqrt(1+rho**2))*np.exp(-lam*temp)


n = 100

lams = [10**(0),10**(-2),10**(-3)]
line_types = ["--","-.",":"]
rho = 0.0
data = np.random.multivariate_normal(np.zeros(2),np.array([[1,rho],[rho,1]]),n)

T1 = np.sum(data[:,0]**2+data[:,1]**2)
T2 = np.sum(data[:,0]*data[:,1])

print("T1",T1)
print("T2",T2)

print(T1,T2,n)
print(integrate.quad(lambda x: jeffreys(x,T1,T2,n),-0.9,0.9))

rho_data = np.linspace(-0.99,0.99,10000)

jeffrey_post = np.exp(lnjeffreys(rho_data,T1,T2,n))
PC_posts = [np.exp(lnPC(lam,rho_data,T1,T2,n)) for lam in lams]


#jeffrey_post /= 1/np.max(jeffrey_post)
#for posterior in PC_posts:
#    posterior /= np.max(posterior)

#jeffrey_post /= np.sum(jeffrey_post)*1/10000
#for posterior in PC_posts:
#    posterior /= np.sum(posterior)*1/10000

#1/0

#post_ratio = ratio(lam,rho_data)

plt.figure(1)
plt.subplot(1,2,1)
for i in range(len(PC_posts)):
    plt.plot(rho_data, PC_posts[i], label="PC: "+str(lams[i]), linestyle=line_types[i])
plt.plot(rho_data, jeffrey_post, label="jeffrey")
plt.legend(loc=1)
plt.subplot(1,2,2)
for i in range(len(PC_posts)):
    plt.plot(rho_data, PC_posts[i]/jeffrey_post, label="PC/jeffrey: "+str(lams[i]), linestyle=line_types[i])
#plt.plot(rho_data,post_ratio*1.5/np.max(post_ratio),label="thr PC/jeffrey")
plt.legend(loc=1)
plt.show()





