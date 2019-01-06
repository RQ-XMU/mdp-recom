import numpy as np
import pandas as pd
import format_transition_df


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
