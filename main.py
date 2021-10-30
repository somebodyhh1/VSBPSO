from read_data import read_names
from read_data import get_data
from RunPSO import RunPSO,multi_population
from copy import deepcopy
from Mic_Relevance import Mic_filter
from Random import Random



README="hybrid reset particles  hybrid_init mdlp  v/=4:\n"
with open("ans.txt", "a") as f:
    f.write(README)

for path in read_names():     #读取数据集路径

    allbests=[]
    print(path)
    X, y = get_data(path)  # 获取数据
    _,n=X.shape
    tmp=deepcopy(X)
    for i in range(1): #循环执行次数
        X=deepcopy(tmp)
        _, mic = Mic_filter(X, y) #计算相关性
        _,n=X.shape
        gbest,train_acc,test_acc1,test_acc2=multi_population(X,y,[0]*n,mic)  #运行PSO
        num=len(gbest)
        m,n=X.shape
        s=str(path)+"  "+str(num)+"/"+str(n)+"  "+str(test_acc1)+"  "+str(test_acc2) + "\n"
        with open("ans.txt","a") as f:
            f.write(s)

README="\n"
with open("ans.txt", "a") as f:
    f.write(README)
