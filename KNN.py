from copy import deepcopy
import numpy as np

class MyKNN:
    def __init__(self,n_neighbors=5):
        self.n_neighbors=n_neighbors
        self.X=[]
        self.y=[]
    def fit(self,X,y):
        self.X=deepcopy(X)
        self.y=deepcopy(y)
        tmp=np.unique(y)
        count=[0]*len(tmp)
        for i in range(len(tmp)):
            count[i]=sum(self.y==i)
        cnt=len(y)
        for i in range(len(count)):
            count[i]/=cnt
        self.percentage=deepcopy(count)

    def predict(self,X):
        m,n=X.shape
        ret=[]
        for i in range(m):
            dis=[]
            for j in range(len(self.X)):
                dist1 = np.linalg.norm(X[i, :] - self.X[j, :])
                dis.append(dist1)
            neighbors = []
            tmp = dis
            for k in range(self.n_neighbors):
                MIN = np.argmin(tmp)
                neighbors.append(MIN)
                tmp[MIN] = 1000000000
            labels = self.y[neighbors]
            #print("neighbors:",labels)
            tmp = max(labels)+1
            count = [0] * tmp
            for i in range(tmp):
                count[i] = sum(labels == i)
                count[i]/=self.percentage[i]
            ret.append(np.argmax(count,axis=0))
        return ret

    def score(self,X,y):
        predict=self.predict(X)
        cnt=0
        for i in range(len(predict)):
            if predict[i]==y[i]:
                cnt+=1
        cnt/=len(predict)
        return cnt