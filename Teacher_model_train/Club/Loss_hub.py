import math

import torch
from torch import nn

from torch import Tensor
class CircleLoss(nn.Module):
    def __init__(self, m=0.25, gamma=256) -> None:
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def convert_label_to_similarity(self,normed_feature: Tensor, label: Tensor):
        similarity_matrix = normed_feature @ normed_feature.transpose(1, 0) #余弦相似度矩阵
        label_matrix = label.unsqueeze(1) == label.unsqueeze(0) #标签矩阵

        positive_matrix = label_matrix.triu(diagonal=1) #对角线以下包括对角线为False
        negative_matrix = label_matrix.logical_not().triu(diagonal=1) #标签矩阵取反，并且对角线以下包括对角线为False

        similarity_matrix = similarity_matrix.view(-1) #相似度矩阵拉成64个数
        positive_matrix = positive_matrix.view(-1) #阳性矩阵拉成64个数
        negative_matrix = negative_matrix.view(-1) #阴性矩阵拉成64个数
        return similarity_matrix[positive_matrix], similarity_matrix[negative_matrix]  #返回阳性矩阵中为True的相似度值，返回阴性矩阵中为True的相似度值
    def forward(self, feat: Tensor, target: Tensor) -> Tensor:
        sp,sn=self.convert_label_to_similarity(feat,target)
        ap = torch.clamp_min(- sp.detach() + 1 + self.m, min=0.)
        an = torch.clamp_min(sn.detach() + self.m, min=0.)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = - ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))

        return loss

class ArcFace(nn.Module):
    def __init__(self,m=0.5,s=64,label_smooth=0.,easy_margin=False):
        super(ArcFace, self).__init__()
        self.m=m
        self.s=s
        self.easy_margin=easy_margin

        self.cos_m=math.cos(m)
        self.sin_m=math.sin(m)
        self.th=math.cos(math.pi-m)
        self.mm=math.sin(math.pi-m)*m

        self.ce=nn.CrossEntropyLoss(label_smoothing=label_smooth)
    def forward(self,costh,lb):
        sine = torch.sqrt((1.0 - torch.pow(costh, 2)).clamp(0, 1))
        phi = costh * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(costh > 0, phi, costh)
        else:
            phi = torch.where(costh > self.th, phi, costh - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(costh.size(), device='cuda')
        one_hot.scatter_(1, lb.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + (
                    (1.0 - one_hot) * costh)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        loss = self.ce(output, lb)
        return loss

class AMSoftmax(nn.Module):
    def __init__(self, m=0.35, s=30,smooth_label=0.1):
        super(AMSoftmax, self).__init__()
        self.m = m
        self.s = s
        self.ce = nn.CrossEntropyLoss(label_smoothing=smooth_label)

        # https://zhuanlan.zhihu.com/p/97475133

    def forward(self, costh, lb):
        # x为输入的特征,lb为真实的标签.

        # print(x_norm.shape, w_norm.shape, costh.shape)
        lb_view = lb.view(-1, 1)  # [10.]==>[10,1]
        delt_costh = torch.zeros(costh.size())
        if lb_view.is_cuda: delt_costh = delt_costh.cuda()
        delt_costh.scatter_(1, lb_view, self.m)  # [10,5]

        costh_m = costh - delt_costh  # [10,5]
        costh_m_s = self.s * costh_m
        loss = self.ce(costh_m_s, lb)
        return loss
# class amsoftmax_magface(nn.Module):
#     def __init__(self,l_m,u_m,l_a,u_a):
#         super(amsoftmax_magface, self).__init__()
#         self.l_m=l_m
#         self.u_m=u_m
#         self.l_a=l_a
#         self.u_a=u_a
#     def _margin(self,):
#     def forward(self,costh,lb):
#         lb_view = lb.view(-1, 1)  # [10.]==>[10,1]
#         delt_costh = torch.zeros(costh.size())
#         if lb_view.is_cuda: delt_costh = delt_costh.cuda()
#         delt_costh.scatter_(1, lb_view, self.m)  # [10,5]
#
#         costh_m = costh - delt_costh  # [10,5]
#         costh_m_s = self.s * costh_m
#         loss = self.ce(costh_m_s, lb)
#         return loss

class Contrast_loss(nn.Module):
    def __init__(self,lamda):
        super(Contrast_loss, self).__init__()
        self.lamda=lamda
    def forward(self,feat,label):
        B = feat.size()[0]
        # mask = (torch.eye(B) * 1.).cuda()

        label_view=label.view(1,-1)
        label_mat=torch.repeat_interleave(label_view,repeats=B,dim=0)
        label_mat=torch.stack([torch.where(tm==t,1,0) for t,tm in zip(label,label_mat)],dim=0).long().cuda()
        non_label_mat=torch.ones_like(label_mat)-label_mat
        non_label_mat=self.lamda*non_label_mat
        label_mat=-1*label_mat
        label_mat=(label_mat+non_label_mat)+torch.eye(B).cuda()
        costh_mat = torch.mm(feat, feat.T)
        loss=(costh_mat*label_mat).sum()

        # costh_mat = torch.mm(feat, feat.T)  # [b,b]
        # on_diag=(costh_mat-label_mat)**2
        # non_label_mat=self.lamuda*non_label_mat
        # off_diag=on_diag*non_label_mat
        # loss=off_diag.sum()

        return loss


if __name__ == '__main__':
    input=torch.randn(4,8).cuda()       #模拟4个样本8个列别
    label=torch.randint(0,8,(4,)).cuda()
    model=ArcFace()
    out=model(input,label)