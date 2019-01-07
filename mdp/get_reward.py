'''
直接把已有的评分当成奖励函数，未评分的电影的奖励函数当成0处理；
还有种方法是对已有的评分矩阵进行矩阵分解拟合，将拟合后的评分当成奖励函数；
这样不会存在未评分的情况。
'''

import numpy as np
import pandas as pd
from scipy import sparse

Reward=np.loadtxt(r'E:\RQ-MASTER\recommender system\mdp-recom\FPMC-master\data\ratings.txt')
sparse_Reward=sparse.coo_matrix(Reward)
