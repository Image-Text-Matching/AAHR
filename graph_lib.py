import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse

from lib.encoders import l2norm
from lib.loss import loss_select
from lib.mca import AdjacencyModel
from torch.utils.checkpoint import detach_variable

from torch.nn import init
from torch.autograd import Variable
import logging


def extra_parameters(parser):
    # loss function
    parser.add_argument('--base_loss', default='trip', type=str, help='the loss function for the initial embeddings.')
    parser.add_argument('--gnn_loss', default='trip', type=str, help='the loss function for the enhanced embeddings.')

    # warmup for training
    parser.add_argument('--warmup', default=8000, type=int,
                        help='warmup iteration for instance-level interaction network')

    # Graph modelling on fragment-level 
    parser.add_argument('--residual_weight', default=0.8, type=float,
                        help='the weight of residual operation for pooling')

    # Graph modelling on instance-level 
    parser.add_argument('--num_layers_enc', default=1, type=int, help='the num_layers of Transformer encoder')
    parser.add_argument('--nhead', default=16, type=int, help='the num_head for Transformer encoder')
    parser.add_argument('--dropout', default=0.1, type=float, help='the dropout rate for Transformer encoder')
    parser.add_argument('--graph_lr_factor', default=1., type=float,
                        help='the learning rate factor for the interaction model')

    # connection and relevance relation 
    parser.add_argument('--mask_weight', default=1.0, type=float, help='use extra weight for the attention matrix')
    parser.add_argument('--threshold', default=0.5, type=float, help='give a threshold for the mask proportion, 0-1')
    parser.add_argument('--topk', default=10, type=int, help='the topk for the region-word pair selection')
    parser.add_argument('--reg_loss_weight', default=10, type=float, help='the values for the regularization loss')
    parser.add_argument('--norm_input', default=1, type=int, help='if use L2-norm embeddings as input')

    # loss function
    parser.add_argument('--cross_loss', default=1, type=int, help='if compute the loss for cross embeddings')

    return parser


class GraphConvolutionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.2):
        super(GraphConvolutionLayer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, input, adj_matrix):
        input = self.dropout(input)
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj_matrix, support) + self.bias
        return output


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2):
        super(GCN, self).__init__()
        self.relu = nn.LeakyReLU(0.2)
        self.gc1 = GraphConvolutionLayer(input_dim, output_dim, dropout)

    def forward(self, x, adj_matrix):
        x = self.relu(self.gc1(x, adj_matrix))
        return x


class GraphLoss(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.opt = opt

        self.iter_count = 0
        self.embed_size = opt.embed_size

        # Initialize the dml objective function for embeddings learning.
        # 损失函数定义  原始特征损失以及强化特征损失
        self.base_loss = loss_select(opt, loss_type=opt.base_loss)
        self.swav_loss = loss_select(opt, loss_type='swav')
        self.gnn_loss = loss_select(opt, loss_type=opt.gnn_loss)

        # the fusion interaction mechanism
        # 融合交互机制
        encoder_layer = nn.TransformerEncoderLayer(d_model=opt.embed_size,
                                                   nhead=opt.nhead, dim_feedforward=opt.embed_size, dropout=opt.dropout)
        self.gnn = nn.TransformerEncoder(encoder_layer, num_layers=opt.num_layers_enc)

        self.gcn_img = GCN(opt.embed_size, opt.embed_size, opt.embed_size, 0.6)
        self.gcn_text = GCN(opt.embed_size, opt.embed_size, opt.embed_size, 0.6)
        self.gcn = GCN(opt.embed_size, opt.embed_size, opt.embed_size, 0.6)

        # construct the cross-embedding graph
        # 得到相似度矩阵和连接性矩阵
        self.adj_model = AdjacencyModel(hidden_size=opt.embed_size, threshold=opt.threshold, topk=opt.topk, detach=True)
        logging.basicConfig(filename=os.path.join(opt.logger_name, 'train.txt'), filemode='w',
                            format='%(asctime)s %(message)s', level=logging.INFO)
        self.logger = logging.getLogger()

    def forward(self, img_emb, cap_emb, img_ids):

        # get latent features and embeddings
        # include the pre-pooling and after-pooling features
        # 得到特征
        img_feat, img_len, clip_img_emb, img_emb, img_emb_pre_pool = img_emb
        cap_feat, cap_len, clip_cap_emb, cap_emb, cap_emb_pre_pool = cap_emb

        img_emb = l2norm(img_emb, dim=-1)
        cap_emb = l2norm(cap_emb, dim=-1)


        bs = img_emb.shape[0]
        assert img_emb.shape[0] == img_emb.shape[0]

        num_loss = 0

        # basic matching loss
        # 原始特征
        base_loss = self.base_loss(clip_img_emb, clip_cap_emb, img_ids)
        num_loss += 1

        base_swav_loss = self.swav_loss(clip_img_emb, clip_cap_emb, self.iter_count)
        num_loss += 1

        if self.iter_count <= self.opt.freeze_prototypes_niters:
            base_swav_loss = 0.
        # 预热到 8000 迭代后开启
        if self.iter_count >= self.opt.warmup:

            # get the connection relation and the relevance relation
            mask_weight = self.opt.mask_weight
            batch_c, batch_r, reg_loss = self.adj_model(img_emb, cap_emb,
                                                        img_regions=img_emb_pre_pool,
                                                        cap_words=cap_emb_pre_pool,
                                                        img_len=img_len,
                                                        cap_len=cap_len, )
            # connection relation
            # 四种关系分别按行拼接，得到两行的连接关系矩阵，其中第一行是i2i和i2t的拼接，第二行是t2i和t2t的拼接
            connect_mask = torch.cat((torch.cat((batch_c['i2i'], batch_c['i2t']), dim=1),
                                      torch.cat((batch_c['t2i'], batch_c['t2t']), dim=1)), dim=0)

            # relevance relation
            # 四种关系分别按行拼接，得到两行的连接关系矩阵，其中第一行是i2i和i2t的拼接，第二行是t2i和t2t的拼接
            relation_mask = torch.cat((torch.cat((batch_r['i2i'], batch_r['i2t']), dim=1),
                                       torch.cat((batch_r['t2i'], batch_r['t2t']), dim=1)), dim=0)
            relation_all = relation_mask.clone()
            # 注意力掩码，构建
            mask = mask_weight * relation_mask.masked_fill_(~connect_mask, float('-inf'))
            # concat mbeddings, batch as the dim=1
            # [2*batch_size, embedding_dim]
            if self.opt.norm_input:
                all_embs = torch.cat((img_emb, cap_emb), dim=0)
            # else:
            #     all_embs = torch.cat((img_emb_notnorm, cap_emb_notnorm), dim=0)

            # get the instance-level relation modeling
            # [2*batch_size, embedding_dim] 输入到 Transformer Encoder 中的维度变为 [2*batch_size, 1, embedding_dim]
            # 也就是 token 数量为 2*batch_size mask为注意力掩码，不进行注意力分数计算
            # [2*batch_size, embedding_dim] -> [2*batch_size, 1, embedding_dim] -> [2*batch_size, embedding_dim]

            # 模态内特征交互
            # 模态内连接性矩阵
            img_emb, cap_emb = torch.split(all_embs, bs, dim=0)
            relation_i2i = batch_r['i2i']
            relation_t2t = batch_r['t2t']

            img_emb_gcn = self.gcn_img(img_emb, relation_i2i)
            cap_emb_gcn = self.gcn_text(cap_emb, relation_t2t)
            all_embs_gcn_ = torch.cat((img_emb_gcn, cap_emb_gcn), dim=0)
            all_embs_gcn__ = self.gcn(all_embs_gcn_, relation_all)

            all_embs_gcn = F.normalize(all_embs_gcn__, dim=-1) + all_embs

            all_embs_gnn = self.gnn(all_embs_gcn.unsqueeze(1), mask).squeeze(1)

            img_emb_gnn, cap_emb_gnn = torch.split(all_embs_gnn, bs, dim=0)

            # L2 normalization for the relation-enhanced embeddings
            img_emb_gnn = F.normalize(img_emb_gnn)
            cap_emb_gnn = F.normalize(cap_emb_gnn)

            # compute loss
            if self.opt.cross_loss:

                gnn_loss1 = self.gnn_loss(img_emb, cap_emb_gnn, img_ids)

                gnn_loss2 = self.gnn_loss(img_emb_gnn, cap_emb, img_ids)

                num_loss += 3
                # num_loss += 2
            else:
                gnn_loss1 = 0.
                gnn_loss2 = 0.
                num_loss += 1

            gnn_loss3 = self.gnn_loss(img_emb_gnn, cap_emb_gnn, img_ids)

            gnn_loss = gnn_loss1 + gnn_loss2 + gnn_loss3

            # gnn_swav_loss = gnn_swav_loss3

            print('gnn_loss1', gnn_loss1.item())
            print('gnn_loss2', gnn_loss2.item())
            print('gnn_loss3', gnn_loss3.item())
            print('gnn_loss', gnn_loss.item())


        else:
            gnn_loss = 0.
            reg_loss = 0.
            # gnn_swav_loss = 0.
        print('base_loss', base_loss.item())
        if base_swav_loss != 0.:
            print('base_swav_loss', base_swav_loss.item())
        else:
            print('base_swav_loss', base_swav_loss)

        if base_swav_loss != 0:
            loss = (base_loss + gnn_loss) + 0.2 * base_swav_loss
        else:
            loss = (base_loss + gnn_loss)
        # 加上正则化损失 默认reg_loss_weight为10
        loss += self.opt.reg_loss_weight * reg_loss
        print('loss', loss.item())
        self.iter_count += 1

        return loss


if __name__ == '__main__':
    pass
