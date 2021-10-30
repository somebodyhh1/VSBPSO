import numpy as np
from sklearn.neighbors import KNeighborsClassifier as KNN
from copy import deepcopy
import time
import random
import math
from sklearn.model_selection import KFold,cross_val_score
from Mic_Relevance import Mic
from sklearn.cluster import MeanShift, estimate_bandwidth
from PSO import BAcc,BAcc1


Balance=1
alpha=0.9
belta=1-alpha
MIC=[]





def cmp(x):
    return x[1]



def Manhattan_distance(a, b):
    ret = 0
    for i in range(len(a)):
        if a[i] != b[i]:
            ret += 1
    return ret


def sum(a):
    threshold = 0.6
    length = len(a)
    ret = 0
    for i in range(length):
        if (a[i] >= threshold):
            ret += 1
    return ret

Total_Particles = 30
num_of_groups = 2

class PSO2:
    def __init__(self, F, c,select):
        global MIC
        m, n = F.shape

        self.data=np.zeros((m,n+1))
        for i in range(m):
            self.data[i,0:n]=deepcopy(F[i])
            self.data[i,n]=c[i]

        self.X = F
        self.y = c
        self.F=F
        self.c=c
        self.num_of_not_updates = 0
        self.num_of_feature = n
        self.num_of_particles = 30
        self.MaxIter = 10
        self.iter = 0
        self.weight = 1
        self.particles = []
        self.v = []
        if n > 50:
            possibility = 4 / n
            tmp = math.log(1 / possibility - 1, math.e)  # 用于获取Max_V
        else:
            tmp = 4
        self.MaxV_pos = tmp
        self.MaxV_neg = -tmp
        self.BAcc_of_Gbest = 0
        self.BAcc_of_Pbest = [0] * self.num_of_particles
        self.BAcc_of_Particle = [0] * self.num_of_particles
        self.gbest = [1] * self.num_of_feature
        self.pbest = [[0] * self.num_of_feature] * self.num_of_particles
        self.num_of_particles_fins_equal_to_gbest = 0
        MIC = np.array(Mic(F, c))
        MAX=max(MIC)
        MIN=min(MIC)
        for i in range(len(MIC)):
            MIC[i]=(MIC[i]-MIN)/(MAX-MIN)
        self.mask = [1] * self.num_of_feature
        self.BEST = [1] * self.num_of_feature
        self.BAcc_of_BEST = 0
        self.select=deepcopy(select)
        feature=self.f(self.BEST)
        acc=BAcc(self.X[:,feature],self.y,feature)
        self.BAcc_of_BEST=acc
        self.BAcc_of_Gbest=acc



    def McTwo(self):
        mic=deepcopy(MIC)
        feature=[]
        feature_t=[]
        MAX=0
        cnt=0
        while True:
            tmp=np.argmax(mic)
            mic[tmp]=-1
            feature_t.append(tmp)
            acc=BAcc(self.X[:,feature_t],self.y,feature_t)
            if MAX<acc:
                MAX=acc
                feature=deepcopy(feature_t)
            else:
                cnt+=1
            if cnt>=50:
                break
            feature_t=deepcopy(feature)
        print(feature)
        print("McTwo acc==",MAX)
        res=[]
        feature=[9, 15, 23, 50, 52, 81, 83, 84, 86, 95, 150, 279, 297]
        acc = BAcc(self.X[:, feature], self.y,feature)
        for i in range(self.num_of_feature):
            if i in feature:
                res.append(1)
            else:
                res.append(0)
        self.gbest=deepcopy(res)
        self.BAcc_of_Gbest=acc
        self.BEST=deepcopy(res)
        self.BAcc_of_BEST=acc
        print("acc==",acc)
        return res


    def f(self, X):  # 获取所选特征 [0 0 1 0 1]变为[2 4]
        threshold = 0.6
        res = []
        length = len(X)
        for i in range(length):
            if (X[i] >= threshold):
                res.append(i)
        return res

    def update_mask(self):
        for j in range(self.num_of_feature):
            self.mask[j] = 0
            for i in range(self.num_of_particles):
                if self.pbest[i][j] == 1:
                    self.mask[j] = 1
                    break

    def hybrid_init(self):  # 粒子初始化
        for i in range(self.num_of_particles):
            tmp = []
            t = []
            if (2 * i < self.num_of_particles):
                limit = self.num_of_feature * 0.2 / self.num_of_feature * 1000
            else:
                limit = self.num_of_feature * 0.8 / self.num_of_feature * 1000
            for j in range(self.num_of_feature):
                t.append(0)
                if np.random.uniform(0, 1000) >= limit:
                    tmp.append(0)
                else:
                    tmp.append(1)
            self.particles.append(tmp)
            self.v.append(t)
            self.cmp(i)

    def MIC_init(self):
        MAX = max(MIC)
        MIN = min(MIC)
        for i in range(self.num_of_particles):
            tmp = []
            t = []
            for j in range(self.num_of_feature):
                t.append(0)
                p = (MIC[j] - MIN) / (MAX - MIN)
                limit = p * 1000
                if np.random.uniform(0, 1000) >= limit:
                    tmp.append(0)
                else:
                    tmp.append(1)
            self.particles.append(tmp)
            self.v.append(t)
            self.cmp(i)

    def sigmoid(self, inx):
        if inx >= 0:
            return 1.0 / (1 + np.exp(-inx))
        else:
            return np.exp(inx) / (1 + np.exp(inx))

    def dis(self, a, b):  # 用于粒子速度更新

        return (a - b)

    def cmp(self, index):  # 计算粒子acc并与pbest比较
        tmp = self.f(self.particles[index])
        BAcc1 = BAcc(self.X[:, tmp], self.y,tmp)


        #print(BAcc1)
        self.BAcc_of_Particle[index] = BAcc1
        #print(index, BAcc1, len(tmp))
        if (BAcc1 > self.BAcc_of_Pbest[index] or (BAcc1==self.BAcc_of_Pbest[index] and len(tmp)>sum(self.pbest[index]))):
            #self.pbest[index],self.BAcc_of_Pbest[index]=self.local_search(self.particles[index])
            self.pbest[index] = deepcopy(self.particles[index])
            self.BAcc_of_Pbest[index]= BAcc1
            #print("update-pbest")
            if (BAcc1 > self.BAcc_of_Gbest or (
                    BAcc1 == self.BAcc_of_Gbest and len(tmp) > sum(self.gbest))):
                self.gbest=deepcopy(self.particles[index])
                self.BAcc_of_Gbest=BAcc1
                #print("update-gbest")
                self.num_of_not_updates = 0
                if (BAcc1 > self.BAcc_of_BEST or (BAcc1==self.BAcc_of_BEST and len(tmp)>sum(self.BEST))):
                    self.BEST = deepcopy(self.particles[index])
                    self.BAcc_of_BEST = BAcc1

        '''
        if np.random.uniform(0,100)<1:
            self.gbest=self.mbest()
            tmp = self.f(self.gbest)
            BAcc1 = BAcc(self.X[:, tmp], self.y)
            self.BAcc_of_Gbest=BAcc1
            print("BACC",BAcc1)
            if(BAcc1>self.BAcc_of_BEST):
                self.BEST=deepcopy(self.gbest)
                self.BAcc_of_BEST=BAcc1
                print("BEST mbest")
        '''

    def update(self):  # 更新粒子位置
        self.weight = 0.7298
        c1 = 1.49618
        c2 = 1.49618
        for i in range(self.num_of_particles):
            for j in range(self.num_of_feature):
                self.v[i][j] = self.v[i][j] * self.weight + c1 * np.random.uniform(0, 1) * \
                               self.dis(self.gbest[j], self.particles[i][j]) + \
                               c2 * np.random.uniform(0, 1) * self.dis(self.pbest[i][j],
                                                                       self.particles[i][j])
                if (self.v[i][j] > self.MaxV_pos):
                    self.v[i][j] = self.MaxV_pos
                if (self.v[i][j] < self.MaxV_neg):
                    self.v[i][j] = self.MaxV_neg
                if (self.sigmoid(self.v[i][j]) > np.random.uniform(0, 1)):
                    self.particles[i][j] = 1
                else:
                    self.particles[i][j] = 0
                if self.select[j]==1:
                    self.particles[i][j] = 1
                if self.mask[j]==0:
                    self.particles[i][j] = 0

            self.cmp(i)
        self.num_of_not_updates += 1

    def hybrid_reset_particles(self):  # 重置粒子位置
        randomlist = random.sample(range(0, self.num_of_particles), int(self.num_of_particles / 1.5))  # 随机选择部分粒子重置
        randomlist=list(range(self.num_of_particles))

        tmp = round(time.time()) % 1000
        np.random.seed(tmp)
        for i in randomlist:
            self.particles[i] = [0] * self.num_of_feature
            if (2 * i < self.num_of_particles):
                limit = self.num_of_feature * 0.8 / self.num_of_feature * 1000  # 取gbest 0.8的特征数
            else:
                limit = self.num_of_feature * 0.2 / self.num_of_feature * 1000  # 取gbest 1.2的特征数
            for j in range(self.num_of_feature):
                if np.random.uniform(0, 1000) >= limit:
                    self.particles[i][j] = 0
                else:
                    self.particles[i][j] = 1
            self.BAcc_of_Pbest[i]=0

            # self.BAcc_of_Pbest[i]=0

            for j in range(self.num_of_feature):
                self.v[i][j] /= self.MaxV_pos * 1.5  # 减小其速度
            self.cmp(i)  # 比较

        self.BAcc_of_Gbest = 0


    def MIC_reset_particles(self):  # 重置粒子位置
        randomlist = random.sample(range(0, self.num_of_particles), int(self.num_of_particles / 1.5))  # 随机选择部分粒子重置
        tmp = round(time.time()) % 1000
        np.random.seed(tmp)
        MAX = max(MIC)
        MIN = min(MIC)
        self.BAcc_of_Gbest=0
        for i in randomlist:
            self.particles[i] = [0] * self.num_of_feature
            for j in range(self.num_of_feature):

                p = (MIC[j] - MIN) / (MAX - MIN)
                limit = p * 1000

                if np.random.uniform(0, 1000) >= limit:
                    self.particles[i][j] = 0
                else:
                    self.particles[i][j] = 1
            self.pbest[i]=deepcopy(self.particles[i])
            tmp=self.f(self.particles[i])
            self.BAcc_of_Pbest[i]=BAcc(self.X[:, tmp], self.y,tmp)
            if self.BAcc_of_Gbest<self.BAcc_of_Pbest[i]:
                self.gbest=deepcopy(self.particles[i])
                self.BAcc_of_Gbest = self.BAcc_of_Pbest[i]
            for j in range(self.num_of_feature):
                self.v[i][j] /= self.MaxV_pos * 1.5  # 减小其速度
            self.cmp(i)  # 比较

    def resetParticles(self):  # 判断何时进行粒子重置
        avg_dis_of_gbest = 0
        for i in range(self.num_of_particles):
            tmp = self.particles[i]
            temp = 0
            for j in range(self.num_of_feature):
                if tmp[j] != self.gbest[j]:
                    temp += 1
            avg_dis_of_gbest += temp
        avg_dis_of_gbest /= self.num_of_particles  # 粒子与gbest的距离

        dis = []
        for i in range(1, self.num_of_particles):
            dis.append(Manhattan_distance(self.particles[0], self.particles[i]))  # 粒子与0号粒子的距离
        variance = np.var(dis)  # 粒子与0号粒子距离的方差，方差小则认为粒子趋于一致，进行重置
        print("variance==", variance)
        variance_of_acc = np.var(self.BAcc_of_Particle)  # 各粒子bacc的方差，同上
        threshold_avg = 3
        print("variance_of_acc=", variance_of_acc)
        threshold_var = max(3, sum(self.gbest) * 0.1)

        threshold_acc = 0.0007
        print(avg_dis_of_gbest, self.num_of_particles_fins_equal_to_gbest, threshold_var)
        if (
                avg_dis_of_gbest < threshold_avg or variance_of_acc < threshold_acc or variance < threshold_var or self.num_of_particles_fins_equal_to_gbest > self.num_of_particles / 3):
            print("particles reset\n")
            self.num_of_particles_fins_equal_to_gbest = 0
            self.hybrid_reset_particles()

    def crossover(self):
        pc = 0.9
        tmp = range(self.num_of_particles)
        tmp = list(tmp)
        random.shuffle(tmp)
        for i in range(0, self.num_of_particles, 2):
            if (np.random.uniform(0, 1) > pc):
                continue
            point = np.random.randint(0, self.num_of_feature)
            tmp1 = deepcopy(self.pbest[tmp[i]])
            tmp2 = deepcopy(self.pbest[tmp[i + 1]])
            for j in range(point):
                tmp2[j] = self.pbest[tmp[i]][j]
            for j in range(point, self.num_of_feature):
                tmp1[j] = self.pbest[tmp[i + 1]][j]
            self.pbest[tmp[i]] = deepcopy(tmp1)
            self.pbest[tmp[i + 1]] = deepcopy(tmp2)

    def mutate(self):
        pm = 1 / (sum(self.mask))
        for i in range(self.num_of_particles):
            for j in range(self.num_of_feature):
                if (np.random.uniform(0, 1) > pm):
                    continue
                self.pbest[i][j] = 1 - self.pbest[i][j]

            tmp = self.f(self.pbest[i])
            BAcc1 = BAcc(self.X[:, tmp], self.y,tmp)
            BAcc1=alpha*BAcc1+(1-len(tmp)/self.num_of_feature)*belta
            self.BAcc_of_Pbest[i] = BAcc1
            if (BAcc1 > self.BAcc_of_Gbest):
                self.gbest = deepcopy(self.pbest[i])
                self.BAcc_of_Gbest = BAcc1

    def cluster_num(self):
        ms = MeanShift()
        ms.fit(self.particles)
        labels = ms.labels_
        unique = np.unique(labels)
        print("n_clusster==", len(unique))
        return len(unique)

    def local_search(self, pos):
        res = deepcopy(pos)
        tmp=self.f(pos)
        fins = BAcc(self.X[:, tmp], self.y,tmp)
        length = sum(pos)
        for i in range(20):
            for j in range(math.floor(0.02 * self.num_of_feature)):
                tmp = np.random.randint(0, self.num_of_feature)
                pos[tmp] = 1 - pos[tmp]
            feature=self.f(pos)
            t = BAcc(self.X[:, feature], self.y,feature)
            if (t > fins or (t==fins and sum(pos)>sum(res))):
                res = deepcopy(pos)
                fins = t
            else:
                pos = deepcopy(res)
        return res, fins

    def mbest(self):
        gbest = 3
        total = self.num_of_particles + gbest
        res = [0] * self.num_of_feature
        for j in range(self.num_of_feature):
            cnt = 0
            for i in range(self.num_of_particles):
                if self.pbest[i][j] == 1:
                    cnt += 1
            if (self.gbest[j] == 1):
                cnt += gbest
            if np.random.uniform(0, 1) < cnt / total:
                res[j] = 1
        return res

    def select_filter(self,gbests,length):
        count=len(gbests)
        res=[]
        for j in range(self.num_of_feature):
            cnt=0
            for i in range(count):
                cnt+=gbests[i][j]
            res.append([j,cnt])
        res.sort(key=cmp, reverse=True)
        res=np.array(res)
        res=res[0:length][:,0]
        ret=[0]*self.num_of_feature
        for i in range(self.num_of_feature):
            if i in res:
                ret[i]=1
        return ret


    def PSO(self,X,y,train,test):
        print("PSO running")
        tmp = round(time.time()) % 1000
        np.random.seed(tmp)
        after_reset=0
        self.MIC_init()  # 初始化
        mu = 0.1
        mask_update_iter = mu * self.MaxIter
        mask_update_iter=10
        select_acc = 0
        gbests=[]
        count=0
        test_acc=0
        for self.iter in range(1, self.MaxIter + 1):

            if self.iter==50:
                #self.select=self.select_filter(gbests,int(sum(self.BEST)*0.8))
                self.select=deepcopy(self.BEST)
                print(sum(self.select))
                select_acc=BAcc1(X[:,self.f(self.select)],y,train,test)
                print(select_acc)
                self.select=[0]*self.num_of_feature

            tmp = round(time.time()) % 1000
            np.random.seed(tmp)

            print("stage 2",self.iter)
            self.update()  # 更新位置
            # self.resetParticles() #粒子重置判定


            if self.iter % mask_update_iter == 0:
                if sum(self.mask)>sum(self.BEST)*5:
                    self.update_mask()
                self.crossover()
                self.mutate()

            if (self.num_of_not_updates >= 20):  # 数次未更新gbest则重置粒子
                count+=1
                gbests.append(self.BEST)
                self.num_of_not_updates = 0
                self.MIC_reset_particles()
                after_reset=0
                self.gbest=[0]*self.num_of_feature
                self.BAcc_of_Gbest=0
            after_reset+=1

            self.BEST,self.BAcc_of_BEST=self.local_search(self.BEST)
            tmp=self.f(self.BEST)
            tmp=MIC[tmp]
            tmp=np.average(tmp)
            Acc=(self.BAcc_of_BEST-belta*(1-sum(self.BEST)/self.num_of_feature))/alpha
            train_acc=self.BAcc_of_BEST
            test_acc=BAcc1(X[:,self.f(self.BEST)],y,train,test)
            print("BEST",self.BAcc_of_BEST, sum(self.BEST), len(self.gbest), sum(self.mask))
            print("gbest",self.BAcc_of_Gbest)
            print("particle",np.average(self.BAcc_of_Particle),np.min(self.BAcc_of_Particle))
            print("test",test_acc,tmp*10)
            print("select",select_acc)
            feature=self.f(self.BEST)
            print(BAcc(X[test][:,feature],y[test],feature))
            print("gbest==", self.f(self.gbest))
        #for i in range(20):
        #    self.BEST, self.BAcc_of_BEST = self.local_search(self.BEST)
        return self.f(self.BEST), self.BAcc_of_BEST, self.f(self.mbest()),select_acc,test_acc
