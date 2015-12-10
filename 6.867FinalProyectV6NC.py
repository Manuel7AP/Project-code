
import numpy as np
import itertools as itert
from numpy import random
from scipy import stats
from itertools import product


def sigmoid(x):
    x[x<-500] = -500 #### control for overflow
    return 1 / (1 + np.exp(-x))


def drawer(W, offset, conditional):
    probs = sigmoid(np.dot(W.T, conditional) + offset) 
    draw = np.matrix(random.binomial(1, probs))
    return draw

def Gradient_approx(W, a, b, H, T):
    xTdraws = np.empty((H.shape[0],1))
    for x0 in H.T: 
        xt = x0.T
        for t in range(T):
            yt = drawer(W, a, xt)
            xt = sigmoid(np.dot(W, yt) + b)
        xTdraws = np.hstack((xTdraws,xt))
    xTdraws = np.matrix(xTdraws[:,1:])
    sigmoidsx0 = [sigmoid(np.dot(W.T, x0.T) + a) for x0 in H.T] 
    sigmoidsxT = [sigmoid(np.dot(W.T, xT.T) + a) for xT in xTdraws.T]
    outerprodx0 = [np.dot(x0j.T,sigma0k.T) for x0j, sigma0k in zip(H.T,sigmoidsx0)]
    outerprodxT = [np.dot(xTj.T,sigmaTk.T) for xTj, sigmaTk in zip(xTdraws.T,sigmoidsxT)]
    gradlistW = [b - a for a,b in zip(outerprodx0, outerprodxT)]
    gradlista = [b - a for a,b in zip(sigmoidsx0, sigmoidsxT)]
    gradlistb = xTdraws - H
    gradmatW = sum(gradlistW)
    gradmata =  sum(gradlista)
    gradmatb = np.sum(gradlistb, axis = 1)

    return gradmatW, gradmata, gradmatb

def Gradient_aprox_incomplete(data_ind, data, T, W, a, b):
    N = len(data_ind)
    M, K = W.shape
    indy = np.array(range(K))
    gW_save = np.zeros((M,K))
    ga_save = np.zeros((K,1))
    gb_save = np.zeros((M,1))
    R  = 1
    for j in range(R):
        i = np.random.randint(0, N)
        M_red = len(data_ind[i])
        indx = np.array(data_ind[i])
        W_subset = W[indx[:,None], indy]
        b_subset = b[indx[:,None],0]
        x = np.reshape(np.matrix(data[i]), (M_red,1))
        
        gW, ga, gb = Gradient_approx(W_subset, a, b_subset, x, T)

        gW_save[indx[:,None], indy] = gW_save[indx[:,None], indy] + gW
        gb_save[indx[:,None],0] = gb_save[indx[:,None],0] + gb
        ga_save = ga_save + ga

    return np.multiply(gW_save,1/float(R)), np.multiply(ga_save,1/float(R)), np.multiply(gb_save,1/float(R))


def train_BRBM(data_ind, data, M, K, T, alpha, epsilon, maxit, init):
    N = len(data_ind)
    def grad_temp(W_complete):
        W = W_complete[0:(M*K),:]
        W = W.reshape((M,K))
        a = W_complete[(M*K):(M*K + K),:]
        b = W_complete[(M*K + K)::,:]
        gW, ga, gb = Gradient_aprox_incomplete(data_ind, data, T, W, a, b)
        full_grad = np.vstack((np.reshape(gW,(M*K,1)),np.reshape(ga, (K,1)), np.reshape(gb, (M, 1))))
        return full_grad
    W_opt, iteration = graddesc(grad_temp, init, alpha, epsilon, maxit)
    print iteration
    W = W_opt[0:(M*K),:] 
    W = W.reshape((M,K))
    a = W_opt[(M*K):(M*K + K),:]
    b = W_opt[(M*K + K)::,:]
    return W, a, b


def computemarginals(data_ind, data, W, a, b):
    M, K = W.shape
    VisibleToHidden = np.ndarray(shape=(M,K), dtype=dict)
    for k in range(K):
        for j in range(M):
            if (j in data_ind): 
                VisibleToHidden[j,k] = {'0': np.exp(b[j,0]*data[data_ind.index(j)]) , '1': np.exp((W[j,k] + b[j,0])*data[data_ind.index(j)] + a[k,0])}
            else:
                VisibleToHidden[j,k] = {'0': 1 + np.exp(b[j,0]), '1': np.exp(a[k,0]) + np.exp(W[j,k] + b[j,0] + a[k,0])}
    HiddenToFactor = np.ndarray(shape=(M,K), dtype=dict)
    for k in range(K):
        for j in range(M):
            HiddenToFactor[j,k] = {'0': 1, '1': 1}
            for i in range(M):
                if (i != j):
                    HiddenToFactor[j,k]['0'] = HiddenToFactor[j,k]['0']*VisibleToHidden[i,k]['0']
                    HiddenToFactor[j,k]['1'] = HiddenToFactor[j,k]['1']*VisibleToHidden[i,k]['1']
    FactorToVisible = np.ndarray(shape=(M,K), dtype=dict)
    for k in range(K):
        for j in range(M):
            FactorToVisible[j,k] = {'0': HiddenToFactor[j,k]['0'] + np.exp(a[k,0])*HiddenToFactor[j,k]['1'],
                                    '1': np.exp(b[j,0])*HiddenToFactor[j,k]['0'] + np.exp(W[j,k] + b[j,0] + a[k,0])*HiddenToFactor[j,k]['1']}
    Marginals = np.ndarray(shape = (M, 1), dtype = dict)
    for j in range(M):
        if (j in data_ind): ### If I observed the node, I want to send out a degenerate probability mass function
            if ( data[data_ind.index(j)] == 0):
                Marginals[j] = {'0':1, '1':0}
            else:
                Marginals[j] = {'0':0, '1':1}
        else: ## If I did not observe the node, then I have to get the marginal
            aux = {'0':1, '1':1} ## I will use aux to compute the unstandardized probabilities
            for k in range(K):
                aux['0'] = aux['0']*FactorToVisible[j,k]['0']
                aux['1'] = aux['1']*FactorToVisible[j,k]['1']
            Marginals[j] = {'0': aux['0']/(aux['0']+aux['1']), '1': aux['1']/(aux['0']+aux['1'])}
    return Marginals[:]

def predict(W, a, b, data_ind, data):
    Marg = computemarginals(data_ind, data, W, a, b)
    M = Marg.shape[0]
    pred = np.zeros(shape = (M,), dtype = float)
    p0 = np.zeros(shape = (M,), dtype = float)
    for k in range(M):
        if (Marg[k,0]['0']  < Marg[k,0]['1']):
            pred[k] = 1
        p0[k] = Marg[k,0]['0']
    rank = stats.rankdata(p0)
    return pred, rank

def graddesc(gradient, initial, alpha, epsilon, maxiter = 1000):
    x_new = np.reshape(np.matrix(initial), (max(np.matrix(initial).shape),1))
    x_old = np.multiply(x_new, 100)
    iteration = 1
    g = 1
    while (np.linalg.norm(g)**2 > epsilon and iteration <= maxiter):
        x_old = x_new
        g = gradient(x_old)
        x_new = x_old - np.multiply(alpha,g)
        iteration += 1
    return np.matrix(x_new), iteration

def toy_complete_data(M, ps_lexicographic, N):      
    list_el = list(product((0,1), repeat = M))
    p = ps_lexicographic ### must match dimension of 2**M
    data_matrix = np.matrix(np.zeros((N,M)))
    for i in range(N):
        x = list(np.random.multinomial(1,p)).index(1)
        data_matrix[i,:] = list_el[x]
    return data_matrix

def toy_hidden_data(data_matrix, percentage):
    N, M = data_matrix.shape
    keep_num = np.floor(float(percentage)*M)
    data_ind = []
    data = []
    for i in range(N):
        aux = np.sort(np.random.choice(range(M),keep_num, replace = False))
        aux = tuple(aux)
 #       aux = (0,1,2) #### this is as seeing complete data 
        data_ind.append(aux)
        chosen = tuple(np.array(data_matrix[i,:][:, aux])[0])
        data.append(chosen)

    return data_matrix, data_ind, data



K = 2  ### hidden units
N = 50 ### sample points
M = 3 ### 3 visible units
T = 1 ### number of iterations of Gibbs sampling
alpha = .001 ### Step size for gradient descent
epsilon = .0001 ### Not really important in this case
maxit = 5000 ### Number of iterations for gradient descent, original was 50,000
init = np.matrix(np.zeros((M*K+K+M,1))) ### initialization for gradient descent
#init = np.matrix(np.random.normal(0, 0.01, (M*K + K + M,1)))

##### Testing cases

p = [1./3., 1./3., 0, 1./3., 0, 0, 0.,0] 
#p = [0,0,1./2.,0,0,0,0,1./2.]
#p = [1./4., 0, 1./4., 0, 1./4., 0, 1./4.,0]
#p = [1./8., 1./8.,1./8., 1./8.,1./8., 1./8.,1./8., 1./8.]


#### I generate the toy data
data_matrix, data_ind, data =  toy_hidden_data(toy_complete_data(M,p,N), .7)

##array = np.array(data_matrix)
##print array[array[:,2] == 1][:,1]
##print sum(array[array[:,2] == 1][:,1])/len(array[array[:,2] == 1][:,1])



###### Train BRBM with my methods
W, a, b = train_BRBM(data_ind, data, M, K, T, alpha, epsilon, maxit, init)

###### Getting marginals
#data_ind = tuple()
#data = tuple()
#print computemarginals(data_ind, data, W, a, b)

##### Getting conditional
data_ind = (2,)
data = (1,)
print computemarginals(data_ind, data, W, a, b)


######### BENCHMARKING #################

##### Train BRBM with SKlearn
##from sklearn.neural_network import BernoulliRBM
##model = BernoulliRBM(n_components=K)
##model.fit(data_matrix)
##BernoulliRBM(batch_size=10, learning_rate=0.1, n_components=K, n_iter=20, random_state=None, verbose=0)
##
##a = np.reshape(model.intercept_hidden_,(K,1))
##b = np.reshape(model.intercept_visible_,(M,1))
##W = model.components_
##
######## Getting marginals
##data_ind = tuple()
##data = tuple()
##print computemarginals(data_ind, data, W.T, a, b)
##
####### Getting conditional
##data_ind = (1,)
##data = (1,)
##print computemarginals(data_ind, data, W.T, a, b)
