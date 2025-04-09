import torch
import torch.nn as nn
import torch.nn.functional as F
from .sublayers import MLP_encoder,MLP_classifier


class TrustWI(nn.Module):
    def __init__(self, dataset, num_classes, hid: int = 128, dropout=0.5, use_bn = True):
        super(TrustWI, self).__init__()
        torch.manual_seed(9999)
        torch.cuda.manual_seed(9999)
        self.feat = dataset.num_node_features
        self.hid = hid
        self.num_classes = num_classes

        self.encoder = MLP_encoder(nfeat=self.feat,
                                 nhid=self.hid,
                                 ncla=self.num_classes,
                                 dropout=dropout,
                                 use_bn=use_bn)
        self.encoder_pse = MLP_encoder(nfeat=self.feat*2,
                                 nhid=self.hid*2,
                                 ncla=self.num_classes,
                                 dropout=dropout,
                                 use_bn=use_bn)

        self.classifier = MLP_classifier(nfeat=self.hid,
                                    nclass=self.num_classes,
                                    dropout=dropout)

        self.use_bn = use_bn

    def DS_Combin_two(self, alpha1, alpha2):
        """ 
        :param alpha1: Dirichlet distribution parameters of view 1
        :param alpha2: Dirichlet distribution parameters of view 2
        :return: Combined Dirichlet distribution parameters
        """
        classes = self.num_classes
        alpha = dict()
        alpha[0], alpha[1] = alpha1, alpha2
        b, S, E, u = dict(), dict(), dict(), dict()
        for v in range(2):
            # print(type(alpha[v]), alpha[v])
            S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
            E[v] = alpha[v]-1
            b[v] = E[v]/(S[v].expand(E[v].shape))
            u[v] = classes/S[v]

        # b^0 @ b^(0+1)
        bb = torch.bmm(b[0].view(-1, classes, 1), b[1].view(-1, 1, classes))
        # b^0 * u^1
        uv1_expand = u[1].expand(b[0].shape)
        bu = torch.mul(b[0], uv1_expand)
        # b^1 * u^0
        uv_expand = u[0].expand(b[0].shape)
        ub = torch.mul(b[1], uv_expand)
        # calculate C
        bb_sum = torch.sum(bb, dim=(1, 2), out=None)
        bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
        C = bb_sum - bb_diag

        # calculate b^a
        b_a = (torch.mul(b[0], b[1]) + bu + ub)/((1-C).view(-1, 1).expand(b[0].shape))
        # calculate u^a
        u_a = torch.mul(u[0], u[1])/((1-C).view(-1, 1).expand(u[0].shape))

        # calculate new S
        S_a = classes / u_a
        # calculate new e_k
        e_a = torch.mul(b_a, S_a.expand(b_a.shape))
        alpha_a = e_a + 1
        return alpha_a


    def forward(self, x, x_glo):

        evidence = F.softplus(self.encoder(x))
        evidence_glo = F.softplus(self.encoder(x_glo))

        pseudo_out = torch.cat([x, x_glo],-1)
        evidence_pse = F.softplus(self.encoder_pse(pseudo_out))    

        alpha_1,alpha_2,alpha_3 = evidence+1, evidence_glo+1, evidence_pse+1
        alpha_all = self.DS_Combin_two(self.DS_Combin_two(alpha_1,alpha_2), alpha_3)

        return alpha_1,alpha_2,alpha_3,alpha_all


