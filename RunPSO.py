from PSO import PSO
import random
from sklearn.neighbors import KNeighborsClassifier as KNN
from PSO import BAcc1
from sklearn.model_selection import StratifiedShuffleSplit,cross_val_score
import numpy as np
from Mic_Relevance import Mic_filter
from PSO2 import PSO2

def BAcc(X, y, feature):
    acc=[]
    knn=KNN(n_neighbors=5)
    return cross_val_score(knn,X,y,cv=5).mean()


def f(X,begin=0):   #获取所选特征 [0 0 1 0 1]变为[2 4]
    res = []
    length = len(X)
    for i in range(length):
        res.append(X[i]+begin)
    return res

def instance_selection(X,y):
    tmp=np.unique(y)
    classes=len(tmp)
    m,n=X.shape
    dis=np.zeros((m,m))
    for i in range(m):
        dis[i][i]=10000000
        for j in range(i+1,m):
            dist1 = np.linalg.norm(X[i,:]-X[j,:])
            dis[i][j]=dis[j][i]=dist1
    K_neighbors=classes*5
    res=[]
    for i in range(m):
        neighbors=[]
        tmp=dis[i,:]
        for j in range(K_neighbors):
            MIN=np.argmin(tmp)
            neighbors.append(MIN)
            tmp[MIN]=1000000000
        labels=y[neighbors]
        cnt=0
        for j in range(len(labels)):
            if(labels[j]==y[i]):
                cnt+=1
        if cnt<=1:
            res.append(i)
    return res

def multi_population(X,y,select,mic):
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=0) #区分训练集、测试集
    accs=[]
    num_of_groups=1
    m,n=X.shape
    span=int(n/num_of_groups)
    begin=0
    end=span
    ans=[]
    for train,test in split.split(X,y):
        while(True):
            feature, train_acc, mbest, select_acc, test_acc=PSO(X[train,:][:,begin:end],y[train],select,mic[begin:end]).PSO(X,y,train,test)
            accs.append(test_acc)
            ans+=f(feature,begin)
            print("ans==",f(feature,begin))
            begin+=span
            end+=span
            if begin>=n:
                break
            if end>n:
                end=n
                if(end-begin<20):
                    break
        print(ans)
        test_acc1=BAcc1(X[:,ans],y,train,test)     #获取测试集acc
        accs.append(test_acc1)

        return ans, 0, test_acc1, 0








def RunPSO(X,y,select,mic):
    m,n=X.shape
    randomint=np.random.randint(0,10000)
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
    for train,test in split.split(X,y):





        #train=deepcopy(MaxTrain)
        #test=deepcopy(MaxTest)
        Train_X=X[train,:]
        Train_Y=y[train]
        Test_X=X[test,:]
        Test_Y=y[test]
        MAXACC=0
        threshold=0
        for i in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]:
            length=int(n*i)
            acc=BAcc(X[train][:,0:length],y[train],list(range(length)))
            print("acc",acc)
            if acc>=MAXACC:
                MAXACC=acc
                threshold=i
        threshold=1

        X=X[:,0:int(threshold*n)]
        mic = mic[0:int(threshold * n)]
        print("threshold:",threshold)






        remove=[]
        #remove=instance_selection(Train_X,Train_Y)
        #print("len==",len(remove))
        #train=np.delete(train,remove,axis=0)
        print("train")
        tmp=np.unique(y[train])
        for i in range(len(tmp)):
            print(sum(y[train]==i))
        print("test")
        tmp=np.unique(y[test])
        for i in range(len(tmp)):
            print(sum(y[test]==i))
        ans=[]
        feature,train_acc,mbest,select_acc,test_acc=PSO(X[train,:],y[train],select,mic).PSO(X,y,train,test)  #运行PSO
        ans.append(test_acc)
        for i in range(3):
            #X=X[:,feature]
            #feature,train_acc,mbest,select_acc,test_acc=PSO2(X[train, :], y[train], select).PSO(X, y, train, test)
            ans.append(test_acc)
        test_acc1=BAcc1(X[:,feature],y,train,test)     #获取测试集acc

    return feature,train_acc,test_acc1,select_acc


