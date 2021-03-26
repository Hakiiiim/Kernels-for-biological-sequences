import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import timeit
from scipy.spatial import distance_matrix
from scipy import optimize
import cvxopt
from itertools import product
import timeit
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
from sklearn.model_selection import train_test_split

print("- libraries imported\n")


################################################################################
# Mismatch kernel
################################################################################

def make_corres(corpus,k):
    '''
    make_corres
    
    Args:
        corpus: the letters that compose the sequences
        k: length of the sequences
        
    Returns:
        corres: dictionnary of correspondances between sequences of length k and indexes
    '''
    corres = dict()
    count = 0
    for u in product(corpus,repeat=k):
        s = ''
        for i in range(k):
            s += u[i]
        corres[s] = count
        count += 1
    return corres

def mismatches(k,m,corres):
    '''
    mismatches
    
    Args:
        k: length of the sequences
        m: number of allowed mismatches
        corres: dictionnary of correspondances between sequences of length k and indexes
        
    Returns:
        mms: matrix of mismatches among all k-length sequences 
    '''
    N = 4**k
    #L = lil_matrix((N,N), dtype="int16")
    L = np.zeros((N,N))
    seqs = list(corres.keys())
    for i in range(N):
        seq1 = seqs[i]
        for j in range(N):
            seq2 = seqs[j]
            mm = 0
            count = 0
            while (mm <= m) and (count <= k-1):
                if seq1[count] != seq2[count]:
                    mm += 1
                count += 1
            if mm <= m:
                L[i,j] = 1
    return L

def gram_mismatch(X, X_test, corres, k, mismatches):
    '''
    gram_spectrum
    
    Args:
        X, X_test: training and test data
        corres: dictionnary of correspondances between sequences of length k and indexes
        k: length of the sequences
        mismatches: matrix of mismatches among all k-length sequences
        
    Returns:
        K: Gram matrix of X and X_test concatenated
    '''
    
    X = np.concatenate((X,X_test),axis=0)
    
    N = X.shape[0]
    m = len(X[0][0])
    n = 4**k
    
    
    #features = csr_matrix((N,n), dtype="int8")
    features = np.zeros((N,n))
    for sample in range(N):
        for i in range(m-k+1):
            segment = X[sample][0][i:i+k]
            features[sample] += mismatches[corres[segment]]
        features[sample] = features[sample]/np.sqrt(features[sample].dot(features[sample]))
        
    #features = csr_matrix(features, shape=(N,n), dtype="int8")
    
    return features.dot(features.T)

print("- kernel defined\n")


################################################################################
# Kernel Ridge Regression
################################################################################

class KernelRidgeRegScratch_mismatch():
    '''
    This class is an implementation of the kernel ridge regression method
    
    Inputs:
        - param: Regularization value (lambda)
        - k: the width of the sequences
        - corres: the matrix of the different sequences
        - mms: the matrix of mismatches
    
    Outputs:
        - fit: the learned parameters
        - predict: the prediction using the learned parameters
    
    '''
    
    def __init__(self, param=1.0, solver='closed_form', corres=None,  k = 6, mms =None):
        self.param = param
        self.solver = solver
        self.corres = corres
        self.k = k
        self.mms = mms
        
    def fit(self, X, y, X_test):
        
        self.X = X
        self.y = y
        self.X_test = X_test
        
        
        # number of rows  in matrix of X 
        n = self.X.shape[0]
        
        # Identity matrix of dimension compatible with our X_intercept Matrix
        I = np.identity(n)
        
        
        # We create a bias term corresponding to param for each column of X 
        I_biased = self.param * I
        
        n = self.X.shape[0]
        
        K = gram_mismatch(self.X, self.X_test, self.corres, self.k, self.mms)
        
        threshold = n
        
        self.K = K[:threshold,:threshold]
        self.K_test = K[threshold:,:threshold]
        
        betas = np.linalg.inv(self.K + I_biased).dot(y)
        self.betas = betas
        
    def predict(self):
        betas = self.betas
        K_test = self.K_test.transpose()
        
        K_predictor = np.dot(betas,K_test)
        
        self.predictions = K_predictor
        return self.predictions
    
print("- kernel ridge regression defined\n")



################################################################################
# Data
################################################################################
    
#1 

Xtr0 = pd.read_csv("Xtr0.csv", sep=",", index_col=0)
Xtr0 = np.array(Xtr0)

N = Xtr0.shape[0]

ytr0 = pd.read_csv("Ytr0.csv", index_col=0)
ytr0 = np.array(ytr0).reshape(N)
ytr0 = 2*ytr0 - 1

#2

Xtr1 = pd.read_csv("Xtr1.csv", sep=",", index_col=0)
Xtr1 = np.array(Xtr1)

N = Xtr1.shape[0]

ytr1 = pd.read_csv("Ytr1.csv", index_col=0)
ytr1 = np.array(ytr1).reshape(N)
ytr1 = 2*ytr1 - 1

# 3

Xtr2 = pd.read_csv("Xtr2.csv", sep=",", index_col=0)
Xtr2 = np.array(Xtr2)

N = Xtr2.shape[0]

ytr2 = pd.read_csv("Ytr2.csv", index_col=0)
ytr2 = np.array(ytr2).reshape(N)
ytr2 = 2*ytr2 - 1

# test

Xte0 = pd.read_csv("Xte0.csv", sep=",", index_col=0)
Xte0 = np.array(Xte0)

Xte1 = pd.read_csv("Xte1.csv", sep=",", index_col=0)
Xte1 = np.array(Xte1)

Xte2 = pd.read_csv("Xte2.csv", sep=",", index_col=0)
Xte2 = np.array(Xte2)

print("- data imported\n")


################################################################################
# Training
################################################################################

# 1

print("start training for dataset 1")
print("warning: This might take a looooooooong time :D")

corpus = ['A','C','G','T']
k = 7
m = 1

corres = make_corres(corpus,k)
mms = mismatches(k,m,corres)
param = 1

Ridge = KernelRidgeRegScratch_mismatch(param = param, corres = corres, k = k, mms = mms)

Ridge.fit(Xtr0,ytr0, Xte0)

ypred = np.sign(Ridge.predict())

rep = ypred.astype("int8")
df_rep1 = pd.DataFrame(data=np.array([range(1000),rep]).T, columns=["id", "Bound"])

# 2

print("start training for dataset 2")

# corpus = ['A','C','G','T']
# k = 7
# m = 1

# corres = make_corres(corpus,k)
# mms = mismatches(k,m,corres)
# param = 1

Ridge = KernelRidgeRegScratch_mismatch(param = param, corres = corres, k = k, mms = mms)

Ridge.fit(Xtr1,ytr1, Xte1)

ypred = np.sign(Ridge.predict())

rep = ypred.astype("int8")
df_rep2 = pd.DataFrame(data=np.array([range(1000,2000),rep]).T, columns=["id", "Bound"])

# 3

print("start training for dataset 3")

# corpus = ['A','C','G','T']
# k = 7
m = 3

# corres = make_corres(corpus,k)
mms = mismatches(k,m,corres)
param = 0.07

Ridge = KernelRidgeRegScratch_mismatch(param = param, corres = corres, k = k, mms = mms)

Ridge.fit(Xtr2,ytr2, Xte2)

ypred = np.sign(Ridge.predict())

rep = ypred.astype("int8")
df_rep3 = pd.DataFrame(data=np.array([range(2000,3000),rep]).T, columns=["id", "Bound"])

################################################################################
# Submission
################################################################################

df_sub = pd.concat([df_rep1,df_rep2,df_rep3])
df_sub = df_sub.replace(-1,0)

df_sub.to_csv('Yte.csv', index=False)

################################################################################
# The end
#By: Ichraq Lemghari, Abdelhakim Benechehab
################################################################################
