import os
import numpy as np
import random
def normalize(X):

    m,n=X.shape
    for i in range(n):
        MAX=max(X[:,i])
        MIN=min(X[:,i])
        if(MAX==MIN):
            continue
        X[:,i]=(X[:,i]-MIN)/(MAX-MIN)
    return X

def read_names():
    filepath="McTwo/"
    tmp=os.listdir(filepath)
    for i in range(len(tmp)):
        tmp[i]=filepath+tmp[i]
    return tmp


def get_data(path):
    tmp=np.loadtxt(path,dtype=str)
    #np.random.shuffle(tmp)
    m,n=tmp.shape
    dic={}
    cnt=0
    X=tmp[:,0:n-1]
    y=tmp[:,n-1]
    X=np.array(X,dtype=float)
    X=normalize(X)
    for i in range(len(y)):
        if dic.__contains__(y[i]):
            y[i]=dic[y[i]]
        else:
            dic[y[i]]=cnt
            y[i] = cnt
            cnt+=1
    y=np.array(y,dtype=int)
    return X,y

