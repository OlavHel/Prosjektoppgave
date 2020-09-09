import numpy as np
import matplotlib.pyplot as plt

def jeffreys(rho,T1,T2,n):
    temp = 1-rho**2
    return np.sqrt(1+rho**2)/((2*np.pi)**n*temp**(n/2+1))*np.exp(-T1/(2*temp)+rho*T2/temp)

def PC(lam, rho, T1, T2, n):
    temp = 1-rho**2
    return lam*np.abs(rho)/(temp**(n/2+1)*np.sqrt(-np.log(temp)))*\
           np.exp(-T1/(2*temp)+rho*T2/temp-lam*np.sqrt(-np.log(temp)))

def ratio(lam, rho):
    temp = np.sqrt(-np.log(1-rho**2))
    return lam*np.abs(rho)/(temp*np.sqrt(1+rho**2))*np.exp(-lam*temp)

class data_pack:
    def __init__(self, S1=None,S2=None, n=10, real_rho = 0.0, resolution = 200):
        if S1 is not None and S2 is not None:
            self.S1 = S1
            self.S2 = S2
        else:
            self.S1 = np.random.gamma(shape=n/2,scale=4*(1+real_rho),size=1)[0]
            self.S2 = np.random.gamma(shape=n/2,scale=4*(1-real_rho),size=1)[0]
        self.n = n
        self.rhos = np.linspace(-0.99,0.99,resolution)
        self.T1 = 1/2*(self.S1+self.S2)
        self.T2 = 1/4*(self.S1-self.S2)
        self.resolution = resolution


    def get_normalized_jeffrey(self):
        post = jeffreys(self.rhos,self.T1,self.T2,self.n)
        return post/(np.sum(post)/self.resolution)

    def get_normalized_PC(self,lam):
        post = PC(lam,self.rhos,self.T1,self.T2,self.n)
        return post/(np.sum(post)/self.resolution)

    def create_plot(self,lambdas):
        plt.figure()
        plt.title("S1="+str(np.around(self.S1,2))+", S2="+str(np.around(self.S2,2))+", n="+str(self.n))
        for lam in lambdas:
            plt.plot(self.rhos,self.get_normalized_PC(lam), label = "PC: "+str(lam))
        plt.plot(self.rhos,self.get_normalized_jeffrey(), label = "jeffrey")
        plt.legend()
        plt.show()


pack1 = data_pack(n=100,S1=9,S2=3)

pack1.create_plot([1,10**(-2),10**(-4)])



