import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import KFold
import time

def random_normalized (d1,d2):
    # to create a valid random markov matrix by dim d1&d2
    x = np.random.random((d1,d2))
    # ensure the sum of each rows are equal to 1
    return x/x.sum(axis=1 ,keepdims=True)

# training data
data = []
for line in open('train534.dat','r'):
    item = line.rstrip()
    data.append(item.split(" "))
ddata = np.empty((1000, 40), dtype=int)
for i in range(1000):
    for j in range(40):
        ddata[i][j] = int(data[i][j])

# testing data
data_t = []
for line2 in open('test1_534.dat','r'):
    item2 = line2.rstrip()
    data_t.append(item2.split(" "))
data_test = np.empty((50, 40), dtype=int)
for i in range(50):
    for j in range(40):
        data_test[i][j] = int(data_t[i][j])


def alpha_beta_not_scale(M,X):
        # max iteration during EM
        np.random.seed(123)
        
        v = 4
        #N = len(X) # number of sequences
        
        # initialize A,B,pi
        pi = np.ones(M) / M # uniform distrbution
        pi = np.array(pi)
        A = random_normalized(M , M )
        A = np.array(A)
        B = random_normalized(M , v )
        B = np.array(B)
        # loop for each obs
        
        T = 40
        alpha = np.zeros((T , M))
        alpha[0] = pi * B[:,X[0]]
        for t in range(1,T):
            alpha[t] = alpha[t-1].dot(A) * B[:,X[t]]        
                
        beta = np.zeros((T,M))
        beta[-1] = 1
        for t in range(T-2,-1,-1):
            beta[t] = A.dot(B[:,X[t+1]])*beta[t+1]
   
        return alpha, beta


al, be = alpha_beta_not_scale(5,ddata[10])
al.shape
t = np.arange(1, 41, 1)
# red dashes, blue squares and green triangles
plt.plot(t, al[:,0], 'r--',label="qi=1")
plt.plot( t, al[:,1], '-s',label="qi=2")
plt.plot( t, al[:,2], ':^',label="qi=3")
plt.plot(t,al[:,3],'dy',label="qi=4")
plt.plot(t,al[:,4],':+m',label="qi=5")

plt.xlabel('time t')
plt.ylabel('alpha')
plt.title('alpha i (i=1,2,3,4,5) for a single sequence of observation')

plt.legend(loc='upper right')
plt.show()


# before prdicting, we need gamma!
def forward(A,B,pi,observtion):
        T = len(observtion)
        n = A.shape[0]
        alpha = np.zeros((T, n))
        for i in range(n):
                index = observtion[0]
                alpha[0, i] = pi[0, i] * B[i, index]
        
        for t in range(1,T,1):
            ot = observtion[t]
            for i in range(n):
                        for j in range(n):
                            alpha[t, i] = alpha[t, i]+alpha[t-1, j] * A[j, i] * B[i, ot] 
        return alpha

def backward(A,B,pi,observtion):
        T = len(observtion)
        n = A.shape[0]
        beta = np.zeros((T,n))
        beta[T-1,:] = 1
        for t in range(T - 2, -1, -1):
            ot = observtion[t + 1]
            for i in range(n):
                for j in range(n):
                    beta[t, i] = beta[t, i]+ A[i, j] * B[j, ot] * beta[t+1, j]
        return beta

def gamma(alpha, beta,observation):
        T = len(observation)
        n = A.shape[0]
        gamma = np.zeros((T, n))
        for t in range(T):
            total=0
            for j in range(n):
                total=total+alpha[t, j] * beta[t, j]
            for i in range(n):
                gamma[t, i] = alpha[t, i] * beta[t, i] / total
        return gamma



def fit_BW_withscale(M,X, max_iter ,eps=1):
        # M is the number of hidden states
        # X is the obs data
        # max_ite is max iteration during EM
        
        v = max(max(x) for x in X) + 1 # number of obs states
        N = len(X) # number of sequences
        
        # initialize A,B,pi
        #np.random.seed(1)
        pi = np.ones(M) / M # uniform distrbution
        # apply random_normalized to create initial A and B
        A = random_normalized(M , M )
        B = random_normalized(M , v )
        ## use confined init
        #A = A_opt8
        #B = B_opt8
        #pi = pi_opt8

        cost = 0
        # main loop
        # cost is the summation of all costs, where each cost is the logP(O|lambda)
        costs = []
        costs.append(cost)
        tstart = time.time()
        for it in range(max_iter):
            if it % 10 == 0:
                print ("it:", it)
            
            # alphas and betas are alpha and beta (forward and backword variables from all obs)
            alphas  = []
            betas = []
            scales = []
            logP = np.zeros(N)
  
            # loop for each obs
            for n in range(N):
                x = X[n]
                T = len(x)
                
                # compute hat_alpha and hat_beta by scaling
                scale = np.zeros(T)
                alpha = np.zeros((T , M))
                alpha[0] = pi * B[:,x[0]]
                # introduce a scale at each time by summation all alpha's at time t
                scale[0] = alpha[0].sum()
                # hat_alpha = alpha * 1/sum_alpha i.e. normalization factor
                alpha[0] /= scale[0]
                for t in range(1,T):
                    obs1 = x[t]
                    alpha_t_prime = alpha[t-1].dot(A)*B[:,obs1]
                    scale[t] = alpha_t_prime.sum()
                    alpha[t] = alpha_t_prime / scale[t]
                
                # we denote c_{t} as scale at time t, and prod(c_1,...,c_t)=C_t, so prod(c_1,...,c_T)=C_T (easily prove)
                # 
                # so in each obs k, we have P * C_T = 1 
                # here we previouly use 1/scale as scale, hence P = C_T
                # and log{P(O|lambda)} = -sum_{t=1}^T logc_t to be sum_{t=1}^T logc_t, hence summation all scale here.
                logP[n] = np.log(scale).sum()  
                
                alphas.append(alpha)
                scales.append(scale)
                
                beta = np.zeros((T,M))
                beta[-1] = 1
                for t in range(T-2,-1,-1):
                    obs2 = x[t+1]
                    beta[t] = A.dot(B[:,obs2]*beta[t+1]) / scale[t+1]     
                betas.append(beta)
            
            cost = np.sum(logP)
            costs.append(cost)
        
            if abs( costs[-1]-costs[-2] ) < eps:
                tend2 = time.time()
                print("convergence time:",tend2-tstart)
                print("A:",A)
                print("B:",B)
                print("pi:",pi)
                it_all = len(costs)
                iterations = np.arange(1, it_all, 1)
                plt.plot(iterations,costs[1:],'-s')
                plt.xlabel('iterations')
                plt.ylabel('log likelihood')
                plt.title('log loikelihood with iterations')
                return A, B, pi,costs
            
            else:
                pass
            
            # re-estimate A, B, pi
            pi = np.sum((alphas[n][0]*betas[n][0]  for n in range(N) ))/N
            
            # create denominators
            den1 = np.zeros((M,1))
            den2 = np.zeros((M,1))
            a_num = np.zeros((M,M))
            b_num = np.zeros((M,v))
            for n in range(N):
                x = X[n]
                T = len(x)
                
                den1 += (alphas[n][:-1]*betas[n][:-1]).sum(axis = 0 ,keepdims = True).T  # sum until T-1
                den2 += (alphas[n]*betas[n]).sum(axis = 0 ,keepdims = True).T  # sum all values
                
                # A numerator
                for i in range(M):
                    for j in range(M):
                        for t in range(T-1):
                            obs3 = x[t+1]
                            a_num[i,j] += alphas[n][t,i]* betas[n][t+1,j]* A[i,j]* B[j,obs3] / scales[n][t+1]
                
                # B numerator
                for i in range(M):
                    for j in range(v):
                        for t in range(T):
                            if x[t] == j:
                                b_num[i,j] += alphas[n][t,i] * betas[n][t,i]
            
            A = a_num / den1
            B = b_num / den2

        tend2 = time.time()
        print("convergence time:",tend2-tstart)
        print("A:",A)
        print("B:",B)
        print("pi:",pi)
        it_all = len(costs)
        iterations = np.arange(1, it_all, 1)
        plt.plot(iterations,costs[1:],'-s')
        plt.xlabel('iterations')
        plt.ylabel('log likelihood')
        plt.title('log loikelihood with iterations')

        plt.show()
        
        return A, B, pi,costs

a,b,c,d= fit_BW_withscale(8,ddata, max_iter = 1,eps =0.01) # max_iter = 1000


# Write code that calculates the log-likelihood of the test set (by Forward-Backward).
def log_likelihood(M,A,B,pi,x):
    T  = len(x)
    scale = np.zeros(T)
    alpha = np.zeros((T,M))
    alpha[0] = pi*B[:,x[0]]
    scale[0] = alpha[0].sum()
    alpha[0] /= scale[0] 
    for t in range(1,T):
        obs = x[t]
        alpha_t_prime = alpha[t-1].dot(A)*B[:,obs]
        scale[t] = alpha_t_prime.sum()
        alpha[t] = alpha_t_prime / scale[t]
        # we denote c_{t} as scale at time t, and prod(c_1,...,c_t)=C_t, so prod(c_1,...,c_T)=C_T (easily prove)
        # so in each obs k, we have P * C_T = 1 
        # here we previouly use 1/scale as scale, hence P = C_T
        # and log{P(O|lambda)} = -sum_{t=1}^T logc_t to be sum_{t=1}^T logc_t, hence summation all scale here.
    return np.log(scale).sum()

def log_likelihood_multi(M,A,B,pi,X):
    return np.array([log_likelihood(M,A,B,pi,x) for x in X])

logprob_8 = log_likelihood_multi(8,a,b,c,data_test)
logprob_8.sum()
print("The log likelihood is:",logprob_8.sum())


#  choose the best number of hidden state
def train_choose_M (M,X_train,X_test,max_iter):
    # X data
    A_hat, B_hat, pi_hat ,d= fit_BW_withscale(M,X_train,max_iter,eps=0.1)
    prob = log_likelihood_multi(M,A_hat, B_hat, pi_hat,X_test)
    return prob.sum()

# cross validation
# 5-fold data
def cv_train (M, X_all , n_splits,max_iter ):
    kf = KFold(n_splits,random_state=True,shuffle=True )
    kf.split(X_all)
    logProb = []
    for train_index,test_index in kf.split(X_all):
        X_train,X_test=X_all[train_index],X_all[test_index]
        logProb.append(train_choose_M (M,X_train,X_test,max_iter))
    return sum(logProb)


# calculate all of the CVs and plot, find the max logP
M = range(0,5)
n = len(M)
cv = np.zeros(n)
for m in M:
    cv[m] = cv_train (m+3,ddata,n_splits=4,max_iter=50)
    print ("the number of hidden state is:",m+3)


def get_state_seq(M,A,B,pi,x):
    T = len(x)
    delta = np.zeros((T,M))
    psi = np.zeros((T,M))
    delta[0] = np.log(pi) + np.log(B[:,x[0]]) 
    for t in range(1,T):
        for j in range(M):
            delta[t,j] = np.max(delta[t-1] + np.log(A[:,j])) + np.log(B[j,x[t]]) 
            psi[t,j] = np.argmax(delta[t-1]+ np.log(A[:,j]))
    states = np.zeros(T,dtype=np.int32)  
    maxlogp = np.max(delta[T-1])
    maxp = np.exp(maxlogp)
    states[T-1] = np.argmax(delta[T-1])
    for t in range(T-2,-1,-1):
        states[t] = psi[t+1,states[t+1]]
    # states is the most possible states at each time
    # maxp is the maximum prob
    return states # ,maxp, np.exp(delta)
        


# multiple obs (predict for training)
A = np.matrix(A_opt8)
B = np.matrix(B_opt8)
pi = np.matrix(pi_opt8)
error_num = 0 
for index in range(len(ddata)):
    obs = ddata[index]
    obs_given = obs[0:39]
    obs_true = obs[-1]
    alpha = forward(A,B,pi,obs_given)
    beta = backward(A,B,pi,obs_given)
    ga = gamma(alpha, beta,obs_given)
    mul = ga[-1,:]*A
    hidden_next = np.argmax(np.array(mul)) 
    obs_next = np.argmax(mul*B)
    if obs_next != obs_true:
        error_num += 1
error_rate = error_num / len(ddata)
print("error rate for training:",error_rate)


# multiple obs testing (test 40)
A = np.matrix(A_opt8)
B = np.matrix(B_opt8)
pi = np.matrix(pi_opt8)
error_num = 0 
for index in range(len(data_test)):
    obs = data_test[index]
    obs_given = obs[0:39]
    obs_true = obs[-1]
    alpha = forward(A,B,pi,obs_given)
    beta = backward(A,B,pi,obs_given)
    ga = gamma(alpha, beta,obs_given)
    mul = ga[-1,:]*A
    hidden_next = np.argmax(np.array(mul)) 
    obs_next = np.argmax(mul*B)
    if obs_next != obs_true:
        error_num += 1
error_rate = error_num / len(data_test)
error_rate
print("error rate for testing:",error_rate)


# output the most likely sequence
A = a
B =b
pi = c
states_all = []
for i in range(len(data_test)):
    states = get_state_seq(8,A,B,pi,data_test[i])
    states_all.append(states)




    
