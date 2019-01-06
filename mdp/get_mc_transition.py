'''
针对每一个用户统计了MC转移矩阵；在统计的时候，每个用户记录的最后一条都忽略了，没有
讲其计算在内
这样做还有一个问题：就是有时用户在同一个时间点有好多条电影评分记录，
这个没有办法分出先后，对统计转移概率矩阵有影响，目前还没有想出解决办法。
'''

import pandas as pd
import numpy as np
import scipy.sparse

def mc_transition(train_df):
    transition_dict={}#记录每个用户的转移概率
    for userid in range(943):
        print("%d user start:" % (userid+1))
        # transition_per_user=np.zeros((1682,1682),dtype=np.float32)#转移概率矩阵初始化，全为0
        transition_per_user=scipy.sparse.dok_matrix((1682,1682),dtype=np.float32)
        df_temp=train_df[train_df['UserId']==userid+1]
        df_temp_sorted=df_temp.sort_index(by='TimeStamp')
        len_temp=len(df_temp_sorted['MovieId'])
        flag_list=[]#如果MovieId在该列表里面，那么就不需在统计了
        for i in range(len_temp):
            if i==len_temp-1:
                break
            index_list=[]#记录出现的索引
            if df_temp_sorted.iloc[i,1] not in flag_list:
                flag_list.append(df_temp_sorted.iloc[i,1])
                i_times=0
                for j in range(len_temp-1):
                    if df_temp_sorted.iloc[i,1]==df_temp_sorted.iloc[j,1]:
                        i_times+=1
                        index_list.append(j)
            #某一部电影出现的次数，以及出现的下标索引
                print("%d record index_list" % i)
                print(index_list)
                movie_list=[]#记录i电影紧接着的电影id
                for k in index_list:
                    movie_list.append(df_temp_sorted.iloc[k+1,1])
                print("movie_list:")
                print(movie_list)
                for movieid in set(movie_list):
                    transition_per_user[df_temp_sorted.iloc[i,1]-1,movieid-1]=(1.0*movie_list.count(movieid)/i_times)
        print("%d user done" % (userid+1))
        transition_dict[userid+1]=transition_per_user
        scipy.sparse.save_npz(r'D:\master\transition\transition_%d.npz' % (userid+1),transition_per_user.tocoo())
    return transition_dict
