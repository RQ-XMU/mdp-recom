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
        df_temp_sorted=df_temp.sort_index(by='TimeStamp')
        len_temp=len(df_temp_sorted['MovieId'])
        movie_index_list=df_temp_sorted.index#记录下在原dataframe中的索引
        for i in range(len_temp-1):#每个用户的df中的最后一条记录都不计算，不然计算转移概率
            if df_temp_sorted.iloc[i,1]==movie+1:
                movie_count_temp+=1
                movie_index_temp.append(movie_index_list[i])
    movie_counts[movie+1]=movie_count_temp
    movie_index[movie+1]=movie_index_temp
