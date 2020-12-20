import numpy as np
import matplotlib.pyplot as plt
from finding_estimates import EstimatorClass
from posteriors import Posterior
import pickle

n = 10
n_samples = 100000
rho = 0.3
unif_post = Posterior("uniform")
jeff_post = Posterior("jeffrey")
arc_post = Posterior("arcsine")
PC_post = Posterior("PC",lam=10**(-4))

est_class = EstimatorClass()

if False:
    data = pickle.load(open("distr_test8.p","rb"))
    print(data)

#    mean_jeff_samples = data["jeff"]
#    mean_sampvar1 = data["sampvar1"]
    mean_med_samples = data["Med"]
    mean_fish_samples = data["FI2"]
#    mean_mle= data["mle"]
#    mean_unif_samples = data["mean_unif"]
#    mean_arc_samples = data["arc"]
#    mean_PC_samples = data["PC"]

    bins = np.linspace(-1, 1, 100)

#    print("MEAN UNIFORM",np.mean(mean_unif_samples-rho),"UNIFORM VAR",np.mean((mean_unif_samples-rho)**2))
#    print("MEAN SAMPLEVAR1",np.mean(mean_sampvar1-rho),"UNIFORM VAR",np.mean((mean_sampvar1-rho)**2))
#    print("MEAN JEFFREY",np.mean(mean_jeff_samples-rho),"JEFFREY VAR",np.mean((mean_jeff_samples-rho)**2))
#    print("MEAN ARCSINE",np.mean(mean_arc_samples),"ARCSINE VAR",np.var(mean_arc_samples))
#    print("MEAN PC",np.mean(mean_PC_samples),"PC VAR",np.var(mean_PC_samples))


    plt.title(r"$n=3$, $\rho=0.3$",fontsize=16)
#    plt.hist(mean_unif_samples, bins, alpha=0.5, label='uniform')
    plt.hist(mean_fish_samples, bins, alpha=0.5, label='FI2')
    plt.hist(mean_med_samples, bins, alpha=0.5, label='Median')
#    plt.hist(mean_mle, bins, alpha=0.5, label='MLE')
#    plt.hist(mean_sampvar1, bins, alpha=0.5, label='Sample corr, var=1')
#    plt.hist(mean_arc_samples, bins, alpha=0.5, label='arcsine')
#    plt.hist(mean_PC_samples, bins, alpha=0.5, label='PC')
#    plt.hist(mean_jeff_samples, bins, alpha=0.5, label='jeffrey')
    plt.axvline(x=np.mean(mean_fish_samples),ymin=0,ymax=250,label="mean FI2",color="red")
    plt.axvline(x=np.mean(mean_med_samples),ymin=0,ymax=250,label="mean Median",color="red")
#    plt.axvline(x=np.mean(mean_unif_samples),ymin=0,ymax=250,label="mean uniform",color="red")
#    plt.axvline(x=np.mean(mean_mle),ymin=0,ymax=250,label="mean MLE",color="green")
#    plt.axvline(x=np.mean(mean_sampvar1),ymin=0,ymax=250,label="Sample corr, var=1",color="green")
#    plt.axvline(x=np.mean(mean_arc_samples),ymin=0,ymax=250,label="mean arcsine",color="blue")
#    plt.axvline(x=np.mean(mean_PC_samples),ymin=0,ymax=250,label="mean PC",color="yellow")
    plt.legend(loc='upper left',fontsize=16)
    plt.show()


if True:
#    mle_samples = np.empty(n_samples)
#    samp_var1 = np.empty(n_samples)
#    mean_fish_samples = np.empty(n_samples)
    mean_med_samples = np.empty(n_samples)
#    mean_unif_samples = np.empty(n_samples)
#    mean_jeff_samples = np.empty(n_samples)
#    mean_arc_samples = np.empty(n_samples)
#    mean_PC_samples = np.empty(n_samples)

    for i in range(n_samples):
        if i%1000==0:
            print(i)
        data = np.random.multivariate_normal(
            np.array([0, 0]),
            np.array([[1, rho], [rho, 1]]),
            size=n
        )
        Xs = data[:, 0]
        Ys = data[:, 1]

        T1 = np.sum(Xs**2+Ys**2)
        T2 = np.sum(Xs*Ys)


#        mean_fish_samples[i] = est_class.fi2_estimator(unif_post,n,T1,T2)
        mean_med_samples[i] = est_class.m_estimator(unif_post,n,T1,T2)
#        samp_var1[i] = est_class.t_c_sample_corr(Xs,Ys,n)
#        mean_jeff_samples[i] = est_class.e_estimator(jeff_post,n,T1,T2)
#        mean_unif_samples[i] = est_class.e_estimator(unif_post,n,T1,T2)
#        mean_arc_samples[i] = est_class.e_estimator(arc_post,n,T1,T2)
#        mean_PC_samples[i] = est_class.e_estimator(PC_post,n,T1,T2)

    pickle.dump({#"FI2":mean_fish_samples,
                 "Med":mean_med_samples
        #"unif":mean_unif_samples,
#                 "sampvar1":samp_var1
#                 "jeff":mean_jeff_samples
#                 "arc":mean_arc_samples,
#                 "PC":mean_PC_samples
                 },open("distr_test7.p","wb"))

#    mle_mean = np.mean(mle_samples)
#    mean_mean = np.mean(mean_unif_samples)
#    mle_var = np.var(mle_samples)
#    var_mean = np.var(mean_unif_samples)

    bins = np.linspace(-1, 1, 100)

#    print("MEAN MLE",mle_mean,"MLE VAR",mle_var)
#    print("MEAN POSTERIOR MEAN UNIFORM",mean_mean, "VARIANCE POSTERIOR MEAN UNIFORM",var_mean)

    plt.hist(mean_jeff_samples, bins, alpha=0.5, label='MLE')
    plt.hist(mean_unif_samples, bins, alpha=0.5, label='mean uniform')
    plt.legend(loc='upper right')
    plt.show()










