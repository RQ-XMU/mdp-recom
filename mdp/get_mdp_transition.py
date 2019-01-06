from scipy import sparse
'''
个性化的mdp——transition，针对每个用户都有一个转移概率矩阵
但是这样做有一个问题，就是需要保存的转移概率矩阵太多，有接近170万个
即使每个转移概率矩阵文件用sparse处理之后平均为2KB，全部记录也有接近5G。
而且文件数目太多，导致删除的时候也会很慢。
'''

total_mdp_transition_dict={}
for userid in range(943):
    user_mc_transition=sparse.load_npz(r'E:\RQ-MASTER\recommender system\mdp-recom\FPMC-master\data\mc_transition\transition_%d.npz' % (userid+1))
    mdp_transition_dict={}
    row=user_mc_transition.tocoo().row
    col=user_mc_transition.tocoo().col
    for action in range(1682):
        mdp_transition=user_mc_transition.todok().copy()
        for i in range(len(row)):
            if action!=col[i]:
                mdp_transition[row[i],col[i]]=0.9*mdp_transition[row[i],col[i]]
        mdp_transition_dict[action]=mdp_transition
        sparse.save_npz(r'E:\RQ-MASTER\recommender system\mdp-recom\FPMC-master\data\mdp_transiton\mdp_transition_user%d_action%d' % (userid+1,action+1),mdp_transition.tocoo())
    total_mdp_transition_dict[userid]=mdp_transition_dict
