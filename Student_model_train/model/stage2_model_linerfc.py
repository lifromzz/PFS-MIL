from sympy.physics.quantum.circuitplot import np
import torch
from torch import nn
import torch.nn.functional as F
import random
from torch_geometric.utils import softmax
from torch_scatter import scatter


class Attn_Net_Gated(nn.Module):
    def __init__(self, D,L=128,temp=1.,dropout=0.35):
        super().__init__()
        self.a_attention=nn.Sequential(nn.Linear(D,L),nn.Sigmoid())
        self.b_attention=nn.Sequential(nn.Linear(D,L),nn.Tanh())
        self.linear = nn.Sequential(nn.Linear(L, 1))

        self.dropout=nn.Dropout(dropout,self.training)
        self.t=temp
        self.init_weight()
    def init_weight(self):
        for a_fc,b_fc,c_fc in zip(self.a_attention,self.b_attention,self.linear):
            if isinstance(a_fc,nn.Linear):
                nn.init.xavier_normal_(a_fc.weight)
            if isinstance(b_fc,nn.Linear):
                nn.init.xavier_normal_(b_fc.weight)
            if isinstance(c_fc,nn.Linear):
                nn.init.xavier_normal_(c_fc.weight)
    def forward(self, feature,batch,istrain):
        instance_norm_feature=F.normalize(feature)
        a=self.a_attention(feature)
        b=self.b_attention(feature)
        a=a*b
        score = self.linear(a) #[D,D]

        score = score / self.t
        score = softmax(score, batch)  # [10240,1] []==>[4,10240,1]

        if istrain:
            score = self.dropout(score)

        return instance_norm_feature,score


class stage2_model(nn.Module):
    def __init__(self,input_dim,embed_dim,AD,AL,AT=1,n_class=2,proxy=True):
        super(stage2_model, self).__init__()
        self.proxy=proxy
        self.nclass=n_class

        self.fc = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(inplace=True)
        )

        # 注意力机制
        self.Attention = Attn_Net_Gated(
            D=embed_dim,
            dropout=AD,
            L=AL,
            temp=AT
        )
        # 聚类中心
        self.bag_clusters=nn.Linear(embed_dim,n_class)
        self.instance_clusters=nn.Linear(embed_dim,n_class)
        # self.bag_clusters = torch.nn.Parameter(torch.randn(n_class, embed_dim), requires_grad=True)
        # self.init_weight()

    def init_weight(self):
        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1)
        nn.init.xavier_normal_(self.bag_clusters, gain=1)

        self.Attention.init_weight()
    def split_feature_by_attn_sing_bag(self,bag,attn):
        """

        :param bag:
        :param attn:
        :param istrain:
        :return: 前10%的实例和10~90%的实例融合后的包特征
        """
        top_per=0.2
        bottom_per=0.2
        num_top_nodes = int(len(bag) * top_per)  # 前20%节点数量
        num_bottom_nodes = int(len(bag) * bottom_per)  # 后20%节点数量
        attn=torch.squeeze(attn,1)
        # 找出attn_x占前20%的节点索引
        attn_clone=attn.clone().detach()
        _, top_indices = torch.topk(attn_clone, num_top_nodes)
        _, bottom_indices = torch.topk(attn_clone, num_bottom_nodes, largest=False)
        mid_indices = list(set(range(len(attn))) - set(top_indices.tolist() + bottom_indices.tolist()))
        mid_indices = torch.tensor(mid_indices)

        top_instance = bag[top_indices]
        mid_instance = bag[mid_indices]
        mid_score=attn[mid_indices]

        return top_instance,mid_instance,mid_score

    def forward(self,H,C,batch=None,istrain=True):
        H = self.fc(H)

        C = self.fc(C)

        instance_norm_feature,score=self.Attention(C,batch,istrain)

        bag_feat = scatter(score * instance_norm_feature, batch, 0)

        instance_prob=self.instance_clusters(H)
        bag_prob=self.bag_clusters(bag_feat)
        return instance_prob,bag_prob

