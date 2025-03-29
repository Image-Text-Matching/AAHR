import os
import torch
import torch.nn as nn
import numpy as np
from transformers import BertModel
import logging
import open_clip
from lib.mlp import FC_MLP
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# 'True' represents to be masked （Do not participate in the calculation of attention）
# 'False' represents not to be masked
# 用于生成填充遮罩。它接受两个参数：embs 是一个张量，表示输入的嵌入向量；lengths 是一个张量，表示每个样本的长度。
def padding_mask(embs, lengths):
    # 创建一个全为1的张量作为初始遮罩。这个遮罩的形状为[样本数, 最大长度]
    mask = torch.ones(len(lengths), embs.shape[1], device=lengths.device)
    for i in range(mask.shape[0]):
        end = int(lengths[i])
        # 有效部分置为0
        mask[i, :end] = 0.

    return mask.bool()


# 如果是线性模型，使用 Xavier 均匀初始化方法来初始化它的权重
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        # 将偏置项初始化为0
        m.bias.data.fill_(0.)


# 用于对输入张量进行 L2 归一化，有助于提高模型的稳定性和收敛速度
def l2norm(X, dim, eps=1e-8):
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


# 输入张量 x 在指定维度 dim 上进行 Max-K 池化操作，然后取平均值
def maxk_pool1d(x, dim, k):
    max_k = maxk(x, dim, k)
    return max_k.mean(dim)


# 在指定维度上找到每个样本的最大的 k 个值
def maxk(x, dim, k):
    _x, index = x.topk(k, dim=dim)
    return _x


# 对输入张量中的每个序列进行MaxK池化操作，并将池化后的结果以张量的形式返回
# uncertain length
def maxk_pool1d_var(x, dim, k, lengths):
    # k >= 1
    results = []
    # assert len(lengths) == x.size(0)
    for idx in range(x.size(0)):
        # keep use all number of features
        # 确保最大池化数量 k 不超过当前序列的长度，防止索引越界
        k = min(k, int(lengths[idx].item()))
        # 只考虑序列的有效部分
        tmp = torch.split(x[idx], split_size_or_sections=lengths[idx], dim=dim - 1)[0]

        max_k_i = maxk_pool1d(tmp, dim - 1, k)
        results.append(max_k_i)

    # construct with the batch
    results = torch.stack(results, dim=0)

    return results


# 平均池化
def avg_pool1d_var(x, dim, lengths):
    results = []
    # assert len(lengths) == x.size(0)

    for idx in range(x.size(0)):
        # keep use all number of features
        tmp = torch.split(x[idx], split_size_or_sections=lengths[idx], dim=dim - 1)[0]
        avg_i = tmp.mean(dim - 1)

        results.append(avg_i)

    # construct with the batch
    results = torch.stack(results, dim=0)

    return results


# 封装了可变 Max-K 池化操作  dim=1, k=2
class Maxk_Pooling_Variable(nn.Module):
    def __init__(self, dim=1, k=2):
        super(Maxk_Pooling_Variable, self).__init__()

        self.dim = dim
        self.k = k

    def forward(self, features, lengths):
        pool_weights = None
        pooled_features = maxk_pool1d_var(features, dim=self.dim, k=self.k, lengths=lengths)

        return pooled_features, pool_weights


# 封装了均值池化操作  dim=1
class Avg_Pooling_Variable(nn.Module):
    def __init__(self, dim=1):
        super(Avg_Pooling_Variable, self).__init__()

        self.dim = dim

    def forward(self, features, lengths):
        pool_weights = None
        pooled_features = avg_pool1d_var(features, dim=self.dim, lengths=lengths)

        return pooled_features, pool_weights


# 封装文本编码器
# embed_size：这是一个整数，表示文本编码器输出的嵌入向量的维度大小
# no_txtnorm：这是一个布尔值参数，用来控制是否对文本编码器的输出进行归一化
def get_text_encoder(opt, embed_size, no_txtnorm=False):
    text_encoder = EncoderText_BERT(opt, embed_size, no_txtnorm=no_txtnorm)

    return text_encoder


# 封装图像编码器
def get_image_encoder(opt, img_dim, embed_size, no_imgnorm=False):
    img_enc = EncoderImageAggr(opt, img_dim, embed_size, no_imgnorm)

    return img_enc


class GatedFusion(nn.Module):
    def __init__(self, embed_dim):
        super(GatedFusion, self).__init__()
        self.fc_gate = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, feat1, feat2):
        # 将两个特征向量拼接
        concat_feat = torch.cat([feat1, feat2], dim=-1)  # (batch_size, 2 * embed_dim)

        # 通过前馈神经网络计算门控值
        gate_values = torch.sigmoid(self.fc_gate(concat_feat))  # (batch_size, embed_dim)

        # 使用门控值对两个特征向量进行加权求和
        fused_feat = gate_values * feat1 + (1 - gate_values) * feat2  # (batch_size, embed_dim)

        return fused_feat


class MultiHeadAttention(nn.Module):
    def __init__(self, feature_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.feature_dim = feature_dim

        self.query = nn.ModuleList([nn.Linear(feature_dim, feature_dim) for _ in range(num_heads)])
        self.key = nn.ModuleList([nn.Linear(feature_dim, feature_dim) for _ in range(num_heads)])
        self.value = nn.ModuleList([nn.Linear(feature_dim, feature_dim) for _ in range(num_heads)])

    def forward(self, clip_emb, region_emb):
        heads = []
        for i in range(self.num_heads):
            Q = self.query[i](clip_emb)
            K = self.key[i](region_emb)
            V = self.value[i](region_emb)

            Q = F.normalize(Q, p=2, dim=-1)
            K = F.normalize(K, p=2, dim=-1)

            attention_scores = torch.bmm(K, Q.unsqueeze(-1)).squeeze(-1)
            attention_weights = F.softmax(attention_scores, dim=-1)
            weighted_regions = torch.bmm(attention_weights.unsqueeze(-1).transpose(2, 1), V).squeeze(1)

            heads.append(weighted_regions)

        return torch.mean(torch.stack(heads), dim=0)


class ClipEncoder:
    def __init__(self):
        super(ClipEncoder, self).__init__()
        self.model_path = 'CLIP/open_clip_pytorch_model.bin'
        self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-B-32',
                                                                               pretrained=self.model_path)
        self.clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')
        self.model = self.model.cuda()

    def get_clip_image_encoder(self):
        return self.model.encode_image

    def get_clip_text_encoder(self):
        return self.model.encode_text


# 图像编码器
class EncoderImageAggr(nn.Module):
    def __init__(self, opt, img_dim=2048, embed_size=1024, no_imgnorm=False):
        super(EncoderImageAggr, self).__init__()

        self.opt = opt

        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm

        # B * N * 2048 -> B * N * 1024
        # N = 36 for region features
        self.fc = FC_MLP(img_dim, embed_size // 2, embed_size, 2, bn=True)
        # Xavier均匀初始化
        self.fc.apply(init_weights)

        self.fc_clip = FC_MLP(512, embed_size, embed_size, 1, bn=False)
        # Xavier均匀初始化
        self.fc_clip.apply(init_weights)

        self.clip_encoder = ClipEncoder()
        # fragment-level relation modeling (for local features)
        # 使用 TransformerEncoder 片段层次关系建模, 图注意力

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=opt.nhead,
                                                   dim_feedforward=embed_size, dropout=opt.dropout)
        self.aggr = nn.TransformerEncoder(encoder_layer, num_layers=1, norm=None)

        # pooling function
        self.graph_pool = Avg_Pooling_Variable()
        self.gpool = Maxk_Pooling_Variable()

        self.gated_fusion = GatedFusion(embed_size)
        self.mha_pool = MultiHeadAttention(embed_size, 8)

    def forward(self, images, full_imag, image_lengths, graph=False):

        img_emb = self.fc(images)

        # initial visual embedding
        # maxpooling
        img_emb_res, _ = self.gpool(img_emb, image_lengths)

        img_emb_pre_pool = img_emb
        # 通过clip编码得到512
        with torch.no_grad():
            clip_image_encoder = self.clip_encoder.get_clip_image_encoder()
            clip_img_emb = clip_image_encoder(full_imag)
            # 512 ->1024
        clip_img_emb = self.fc_clip(clip_img_emb)

        # get padding mask
        # 得到有效部分
        src_key_padding_mask = padding_mask(img_emb, image_lengths)

        # switch the dim
        img_emb = img_emb.transpose(1, 0)
        # Transformer模型中，输入的序列需要按照时间步长（或者叫序列长度）在第一个维度上
        img_emb = self.aggr(img_emb, src_key_padding_mask=src_key_padding_mask)
        img_emb = img_emb.transpose(1, 0)
        img_emb = self.mha_pool(clip_img_emb, img_emb)
        # enhanced visual embedding
        # 均值池化
        # B * N * 1024 -> B  * 1024
        # img_emb, _ = self.graph_pool(img_emb, image_lengths)

        # the final global embedding
        # residual_weight 就是论文公式 (2) 中的 beta
        img_emb = self.opt.residual_weight * img_emb_res + (1 - self.opt.residual_weight) * img_emb

        img_emb_notnorm = img_emb
        if not self.no_imgnorm:
            # 用于对输入张量进行L2归一化
            img_emb = l2norm(img_emb, dim=-1)
            clip_img_emb = l2norm(clip_img_emb, dim=-1)

        img_emb = self.gated_fusion(img_emb, clip_img_emb)

        # img_emb = 0. * img_emb + 1.0 * clip_img_emb
        if graph:
            return images, image_lengths, img_emb, img_emb_notnorm, img_emb_pre_pool
        else:
            return img_emb


# 文本编码器
# Language Model with BERT backbone
class EncoderText_BERT(nn.Module):
    def __init__(self, opt, embed_size=1024, no_txtnorm=False):
        super(EncoderText_BERT, self).__init__()

        self.opt = opt

        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

        # self.bert = BertModel.from_pretrained(opt.bert_path)
        # 加载 bert 模型
        self.bert = BertModel.from_pretrained('./bert-base-uncased')

        # backbone features -> embbedings   768 -> 1024
        self.linear = nn.Linear(768, embed_size)

        self.fc_clip = FC_MLP(512, embed_size, embed_size, 1, bn=False)
        # Xavier均匀初始化
        self.fc_clip.apply(init_weights)

        self.clip_encoder = ClipEncoder()
        # relation modeling for local feature
        # 使用 TransformerEncoder 片段层次关系建模, 图注意力
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=opt.nhead,
                                                   dim_feedforward=embed_size, dropout=opt.dropout)
        self.aggr = nn.TransformerEncoder(encoder_layer, num_layers=1, norm=None)

        # pooling function
        self.graph_pool = Avg_Pooling_Variable()
        self.gpool = Maxk_Pooling_Variable()

        self.gated_fusion = GatedFusion(embed_size)
        self.mha_pool = MultiHeadAttention(embed_size, 8)

    def forward(self, x, clip_captions, lengths, graph=False):

        # Embed word ids to vectors
        # pad 0 for redundant tokens in previous process
        bert_attention_mask = (x != 0).float()

        # all hidden features, D=768 in bert-base model
        # attention_mask： Mask to avoid performing attention on padding token indices.
        # 避免对填充标记索引执行关注的掩码。
        # bert_output[0] is the last/final hidden states of all tokens
        # bert_output[1] is the hidden state of [CLS] + one fc layer + Tanh, can be used for classification tasks.

        # N = max_cap_lengths, D = 768

        bert_emb = self.bert(input_ids=x, attention_mask=bert_attention_mask)[0]  # B x N x D 768
        cap_len = lengths

        # B x N x embed_size 1024
        cap_emb = self.linear(bert_emb)

        # initial textual embedding
        # maxpooling
        cap_emb_res, _ = self.gpool(cap_emb, cap_len)

        cap_emb_pre_pool = cap_emb

        # fragment-level relation modeling for word features

        # get padding mask
        src_key_padding_mask = padding_mask(cap_emb, cap_len)

        # switch the dim
        cap_emb = cap_emb.transpose(1, 0)
        # 使用 TransformerEncoder 片段层次关系建模, 图注意力
        cap_emb = self.aggr(cap_emb, src_key_padding_mask=src_key_padding_mask)
        cap_emb = cap_emb.transpose(1, 0)

        # 通过clip编码得到512
        with torch.no_grad():
            clip_text_encoder = self.clip_encoder.get_clip_text_encoder()
            clip_cap_emb = clip_text_encoder(clip_captions)
        # 512 ->1024
        clip_cap_emb = self.fc_clip(clip_cap_emb)


        cap_emb = self.mha_pool(clip_cap_emb, cap_emb)

        cap_emb = self.opt.residual_weight * cap_emb_res + (1 - self.opt.residual_weight) * cap_emb

        # the final global embedding
        cap_emb_notnorm = cap_emb
        if not self.no_txtnorm:
            cap_emb = l2norm(cap_emb, dim=-1)
            clip_cap_emb = l2norm(clip_cap_emb, dim=-1)

        cap_emb = self.gated_fusion(cap_emb, clip_cap_emb)
        if graph:
            return bert_emb, cap_len, cap_emb, cap_emb_notnorm, cap_emb_pre_pool
        else:
            return cap_emb


if __name__ == '__main__':
    pass
