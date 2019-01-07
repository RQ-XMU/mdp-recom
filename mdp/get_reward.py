'''
直接把已有的评分当成奖励函数，未评分的电影的奖励函数当成0处理；
还有种方法是对已有的评分矩阵进行矩阵分解拟合，将拟合后的评分当成奖励函数；
这样不会存在未评分的情况。
由于是非个性化的转移概率，所以评分矩阵需要做相应的处理；
因为有1682部电影，所以每部电影对应一个奖励函数；
处理为：对于每一部电影而言，统计出所有对他评分的记录的均值作为该电影状态的奖励函数
'''

import numpy as np
import pandas as pd
from scipy import sparse

ratings=np.loadtxt(r'E:\RQ-MASTER\recommender system\mdp-recom\FPMC-master\data\ratings.txt')
rating_reward=pd.DataFrame(ratings)
def mean_nonzero(column):
    sum=0
    times=0
    for i in range(943):
        if column[i]!=0.0:
            times+=1
            sum+=column[i]
    if times!=0.0:
        return 1.0*sum/times
    else:
        return 0.0

rating_mean_list=[]
for i in range(1682):
    mean=mean_nonzero(rating_reward[i])
    rating_mean_list.append(float('%.2f' % mean))
