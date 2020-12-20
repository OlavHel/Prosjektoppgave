import numpy as np
from scipy.integrate import quad


class Posterior:
    def __init__(self,posterior=None,lam=10**(-1)):
        self.posterior = posterior
        self.lam = lam
        self.distr = {
            "jeffrey":self.jeffreys,
            "PC":lambda x,n,T1,T2: self.PC(x,n,T1,T2,self.lam),
            "uniform":self.uniformprior,
            "arcsine":self.arcsine}

    def get_list_of_posteriors(self):
        return [[0,"jeffrey"],
                [1,"PC"],
                [2,"uniform"],
                [3,"arcsine"]]

    def set_posterior(self,i,lam=10**(-1)):
        self.posterior = {0:"jeffrey",
                1:"PC",
                2:"uniform",
                3:"arcsine"}[i]

    def distribution(self,rho,n,T1,T2):
        return self.distr[self.posterior](rho,n,T1,T2)

    def norm_distribution(self,rho,n,T1,T2):
        return self.distribution(rho,n,T1,T2)/self.normalization(n,T1,T2)

    def normalization(self,n,T1,T2):
        distr = self.distr[self.posterior]
        c, c_error = quad(distr, -1, 1, args=(n, T1, T2))
        return c        

    def uniformprior(self,rho, n, T1, T2):
        if type(rho) == type(np.array([0.1])) and len(rho) == 1:
            rho = rho[0]
        temp = 1 - rho ** 2
        if type(rho) == type(np.array([])):
            rho[rho <= -1] = 0
            rho[rho >= 1] = 0
        elif (rho <= (-1) or rho >= (1)):
            return 0
        return np.exp(n / 1.1) * temp ** (-n / 2) * np.exp(-T1 / (2 * temp) + rho * T2 / temp)

    def jeffreys(self,rho, n, T1, T2):
        if type(rho) == type(np.array([0.1])) and len(rho) == 1:
            rho = rho[0]
        temp = 1 - rho ** 2
        if type(rho) == type(np.array([])):
            rho[rho <= -1] = 0
            rho[rho >= 1] = 0
        elif (rho <= (-1) or rho >= (1)):
            return 0
        return np.sqrt(1 + rho ** 2) / ((2 * np.pi) ** n * temp ** (n / 2 + 1)) * np.exp(
            -T1 / (2 * temp) + rho * T2 / temp)

    def PC(self,rho, n, T1, T2, lam): ## noe galt med PC
        if type(rho) == type(np.array([0.1])) and len(rho) == 1:
            rho = rho[0]
        if type(rho) == type(np.array([])):
            rho[rho <= -1] = 0
            rho[rho >= 1] = 0
            rho[rho == 0] = lam
        elif (rho <= (-1) or rho >= (1)):
            return 0
        elif rho == 0:
            return lam
        temp = 1 - rho ** 2
        return lam * np.abs(rho) / (temp ** (n / 2 + 1) * np.sqrt(-np.log(temp))) * \
               np.exp(-T1 / (2 * temp) + rho * T2 / temp - lam * np.sqrt(-np.log(temp)))

    def arcsine(self,rho, n, T1, T2):
        if type(rho) == type(np.array([0.1])) and len(rho) == 1:
            rho = rho[0]
        if type(rho) == type(np.array([])):
            rho[rho <= -1] = 0
            rho[rho >= 1] = 0
        elif (rho <= (-1) or rho >= (1)):
            return 0
        temp = 1 - rho ** 2
        return np.exp(n) / (temp ** (n / 2 + 1 / 2)) * np.exp(-T1 / (2 * temp) + rho * T2 / temp)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from finding_estimates import EstimatorClass

    est_class = EstimatorClass()

    n = 3
    rho = 0.9
    S1 = 2*n*(1+rho)#np.random.gamma(n/2,4*(1+rho))#
    S2 = 2*n*(1-rho)#np.random.gamma(n/2,4*(1-rho))#

    Xs = np.array([0.2, -0.39, -1.95])
    Ys = np.array([1.08, 0.35, -0.4])

    T1 = np.sum(Xs**2+Ys**2)
    T2 = np.sum(Xs*Ys)

    S1 = T1+2*T2
    S2 = T1-2*T2

    print(S1,S2)

    mle = est_class.mle(Xs,Ys,n)

    jeff = Posterior("jeffrey")
    PC = Posterior("PC",lam=10**(-4))
    unif = Posterior("uniform")
    arcs = Posterior("arcsine")

    distrs = [jeff,PC,unif,arcs]
    labels = ["Jeffreys","PC, $\lambda=10^{-4}$","uniform","arcsine"]

    rhos = np.linspace(-1+1/500,1-1/500,1000)

    plt.figure()
    plt.title("n: "+str(n)+", S1: "+str(S1)+", S2: "+str(S2))
    for i in range(len(distrs)):
        distr = distrs[i]
        plt.plot(rhos,distr.norm_distribution(rhos,n,T1,T2),label=labels[i])
    plt.plot([mle,mle],[0,np.max([d.norm_distribution(mle,n,T1,T2) for d in distrs])],label="MLE")
    plt.legend()
    plt.show()










