from sklearn.model_selection import StratifiedShuffleSplit,cross_val_score
from sklearn.model_selection import KFold,cross_val_score,GridSearchCV
from sklearn.metrics import f1_score
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as KNN

def BAcc(X, y):
    m,n=X.shape
    if n==0:
        return 0
    acc=[]
    knn=KNN(n_neighbors=5,weights='distance',p=2)
    return cross_val_score(knn,X,y,cv=5).mean()



def BAcc1(X,y,train,test):    #训练集acc
    m,n=X.shape
    if n==0:
        return 0
    knn=KNN(n_neighbors=5,weights='distance',p=2)
    knn.fit(X[train], y[train])
    return knn.score(X[test, :], y[test])


def get_key(x):
    return x[1]


def Run(X,y,train,test):
    m,n=X.shape
    num_of_feature=int(n/5)
    data=[]
    for i in range(1000):
        num=int(np.random.normal(num_of_feature,10))
        tmp=np.array(range(n))
        np.random.shuffle(tmp)
        feature=tmp[0:num]
        train_acc=BAcc(X[train,:][:,feature],y[train])
        data.append([feature,train_acc])
    data.sort(key=get_key,reverse=True)
    features=data[0:100]
    features=[x[0] for x in features]
    ans=[]
    select=[0]*n
    count=[]
    for i in range(len(features)):
        for j in range(len(features[i])):
            select[features[i][j]]+=1

    for i in range(n):
        count.append([i,select[i]])
    count.sort(key=get_key,reverse=True)
    count=np.array(count)
    length=0
    for i in range(len(count)):
        if count[i][1]<30:
            length=i
            break
    count = count[:, 0]

    ans=count[0:length]
    train_acc=BAcc(X[train,:][:,ans],y[train])
    test_acc=BAcc1(X[:,ans],y,train,test)
    print("pos",len(ans), train_acc, test_acc)
    return ans,train_acc,test_acc




def Run_opposite(X, y, train, test):
    m, n = X.shape
    num_of_feature = int(n / 2)
    data = []
    for i in range(1000):
        num = int(np.random.normal(num_of_feature, 10))
        tmp = np.array(range(n))
        np.random.shuffle(tmp)
        feature = tmp[0:num]
        train_acc = BAcc(X[train, :][:, feature], y[train])
        data.append([feature, train_acc])
    data.sort(key=get_key, reverse=False)
    features = data[0:100]
    features = [x[0] for x in features]
    ans = []
    select = [0] * n
    count = []
    for i in range(len(features)):
        for j in range(len(features[i])):
            select[features[i][j]] += 1

    for i in range(n):
        count.append([i, select[i]])
    count.sort(key=get_key, reverse=True)
    count = np.array(count)
    length = 0
    for i in range(len(count)):
        if count[i][1] < 50:
            length = i
            break
    count = count[:, 0]

    ans = count[0:length]

    all=set(range(n))
    ans=set(ans)
    ans=all-ans
    ans=list(ans)
    train_acc = BAcc(X[train, :][:, ans], y[train])
    test_acc = BAcc1(X[:, ans], y, train, test)
    print("oppo:",len(ans), train_acc, test_acc)

    return ans, train_acc, test_acc


def Random(X,y):
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
    feature=[]
    m,n=X.shape
    for train,test in split.split(X,y):
        return Run_opposite(X,y,train,test)