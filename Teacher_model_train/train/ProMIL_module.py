import torch
from torch import nn
import torch.nn.functional as F
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
    def forward(self, feature, batch,istrain=True):
        feature=F.normalize(feature)
        a=self.a_attention(feature)
        b=self.b_attention(feature)
        a=a*b
        score = self.linear(a) #[D,D]

        score = score / self.t
        score = softmax(score, batch) #[10240,1] []==>[4,10240,1]

        if istrain:
            score = self.dropout(score)
        out = scatter(score * feature, batch, 0)
        return out,score,feature

class ProMIL(nn.Module):
    def __init__(self,n_class,input_dim,embed_dim,AL=256,AD=0.35,AT=1):
        super(ProMIL, self).__init__()
        self.n_class=n_class
        self.input_dim=input_dim
        self.embed_dim=embed_dim

        self.fc=nn.Sequential(
            nn.BatchNorm1d(self.input_dim),
            nn.Linear(self.input_dim, self.embed_dim),
            nn.ReLU(inplace=True)
        )

        # 注意力机制
        self.Attention = Attn_Net_Gated(
            D=self.embed_dim,
            dropout=AD,
            L=AL,
            temp=AT
        )
        self.classifier=nn.Linear(self.embed_dim,n_class)
        #聚类中心
        # self.clusters = torch.nn.Parameter(torch.randn(n_class,self.embed_dim),requires_grad=True)
        self.init_weight()
    def init_weight(self):
        for m in self.fc.modules():
            if isinstance(m,nn.Linear):
                nn.init.xavier_normal_(m.weight,gain=1)
        #nn.init.xavier_normal_(self.classifier, gain=1)
        self.Attention.init_weight()

    def forward(self, x,batch=None,istrain=True):
        if batch is None:
            batch = torch.zeros(size=(len(x),), dtype=torch.int64).cuda()

        x=self.fc(x)

        x_norm,attention,patch_norm_feature = self.Attention(x, batch, istrain)

        prob=self.classifier(x_norm)

        return prob


def main():
    model=ProMIL(n_class=4,input_dim=3072,embed_dim=512)
    data=torch.randn(size=(1234,3072))
    batch=torch.ones(size=(1,1234))
    out=model(data,batch)

if __name__ == '__main__':
    main()