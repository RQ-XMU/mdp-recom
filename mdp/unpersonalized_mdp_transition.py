import numpy as np
import pandas as pd
import format_transition_df
from scipy import sparse


train_df=format_transition_df(r'E:\RQ-MASTER\recommender system\mdp-recom\FPMC-master\data\train.csv')
#统计每部电影出现的次数以及相应的索引位置
movie_counts={}
movie_index={}
for movie in range(1682):
    movie_count_temp=0
    movie_index_temp=[]#记录的是在train_df中的索引
    for user in range(943):
        df_temp=train_df[train_df['UserId']==user+1]
        len_temp=len(df_temp['MovieId'])
        movie_index_list=df_temp.index#记录下在原dataframe中的索引
        #这里有个问题要注意，不能是排序后的索引，应该是未排序之前的索引，即df_temp，
        #因为排序之后索引也乱了
        for i in range(len_temp-1):#每个用户的df中的最后一条记录都不计算，不然计算转移概率
            if df_temp.iloc[i,1]==movie+1:
                movie_count_temp+=1
                movie_index_temp.append(movie_index_list[i])
    movie_counts[movie+1]=movie_count_temp
    movie_index[movie+1]=movie_index_temp


unpersonalized_mc_transition=np.zeros((1682,1682),dtype=np.float32)
for movie in range(1682):
    next_movie_list=[]
    if movie_counts[movie+1]!=0:
        for index in movie_index[movie+1]:
            next_movie_list.append(train_df.iloc[index+1,1])
        for movieid in set(next_movie_list):
            unpersonalized_mc_transition[movie,movieid-1]=1.0*next_movie_list.count(movieid)/movie_counts[movie+1]

sparse_unpersonalized_mc_transition=sparse.coo_matrix(unpersonalized_mc_transition)
sparse.save_npz(r'E:\RQ-MASTER\recommender system\mdp-recom\FPMC-master\data\unpersonalized\...',sparse_unpersonalized_mc_transition)



row=sparse_unpersonalized_mc_transition.tocoo().row
col=sparse_unpersonalized_mc_transition.tocoo().col
for action in range(1682):
    unper_mdp_transition=sparse_unpersonalized_mc_transition.todok().copy()
    for i in range(len(row)):
        if action!=col[i]:
            unper_mdp_transition[row[i],col[i]]=0.9*unper_mdp_transition[row[i],col[i]]
        else:
            unper_mdp_transition[row[i],col[i]]=1.1*unper_mdp_transition[row[i],col[i]]
    sparse.save_npz(r'E:\RQ-MASTER\recommender system\mdp-recom\FPMC-master\data\unpersonalized\unpersonalized_mdp_transition\unp_mdp_action_%d' % (action+1),unper_mdp_transition.tocoo())
