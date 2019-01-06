import pandas as pd

def format_transition_df(filename):
    f=open(filename)#这里最好加上这句代码，不然可能会遇到一些问题
    train_df=pd.read_csv(f, sep=',', names=['UserId', 'MovieId', 'Rating', 'TimeStamp'])
    return train_df
