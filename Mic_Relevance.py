from minepy import MINE
import numpy as np

def cmp(x):
    return x[1]
def mic(F,C):
    mine=MINE(alpha=0.6,c=15)
    mine.compute_score(F,C)
    return mine.mic()

def Mic(F, C):

    r = 0
    _, k = F.shape
    ans=[]
    for i in range(k):
        micFC = mic(F[:, i], C)
        ans.append(micFC)
    return ans

def Mic_filter(F,C):
    _, k = F.shape
    ans=[]
    r=0
    for i in range(k):
        micFC = mic(F[:, i], C)
        if micFC<r:
            continue
        ans.append([i,micFC])
    ans.sort(key=cmp,reverse=True)
    length=int(len(ans))
    ans=np.array(ans,dtype=float)
    ans=ans[0:int(length)]
    mic_ans=ans[0:length,1]
    ans=ans[0:length,0]
    ans=np.array(ans,dtype=int)
    return ans,mic_ans