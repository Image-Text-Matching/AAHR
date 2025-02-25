import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

def pos_neg_mask(labels):
    # 其中第i行和第j列的元素为True表示标签labels[i]和labels[j]相等
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) 
    neg_mask = labels.unsqueeze(0) != labels.unsqueeze(1)

    return pos_mask, neg_mask


def pos_neg_mask_xy(labels_col, labels_row):

    pos_mask = (labels_row.unsqueeze(0) == labels_col.unsqueeze(1)) 
    neg_mask = (labels_row.unsqueeze(0) != labels_col.unsqueeze(1))

    return pos_mask, neg_mask


def loss_select(opt, loss_type='vse'):

    if loss_type == 'vse':
        # the default loss
        criterion = ContrastiveLoss(opt=opt, margin=opt.margin, max_violation=opt.max_violation)
    elif loss_type == 'trip':
        # Triplet loss with the distance-weight sampling
        criterion = TripletLoss(opt=opt)
    elif loss_type == 'swav':
        criterion = SwAVContrastiveLoss(opt=opt)
    else:
        raise ValueError('Invalid loss {}'.format(loss_type))
    
    return criterion

# 三元组损失函数，可以选是否启用最大负例损失函数
class ContrastiveLoss(nn.Module):

    def __init__(self, opt, margin=0.2, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.opt = opt
        self.margin = margin
        self.max_violation = max_violation
        self.mask_repeat = opt.mask_repeat

        self.false_hard = []

    def max_violation_on(self):
        self.max_violation = True
        # print('Use VSE++ objective.')

    def max_violation_off(self):
        self.max_violation = False
        # print('Use VSE0 objective.')

    def forward(self, im, s, img_ids=None):
        # 获取图像和文本的批量大小
        im_batch_size = im.size(0)
        s_batch_size = s.size(0)

        # 如果批量大小不同
        if im_batch_size != s_batch_size:
            # 填充较小的批次
            if im_batch_size < s_batch_size:
                # 填充图像特征
                im = torch.cat([im, torch.zeros(s_batch_size - im_batch_size, im.size(1)).to(im.device)], dim=0)
            else:
                # 填充文本特征
                s = torch.cat([s, torch.zeros(im_batch_size - s_batch_size, s.size(1)).to(s.device)], dim=0)

        # compute image-sentence score matrix
        # 计算相似性矩阵
        scores = get_sim(im, s)
        # 相似性得分矩阵scores的对角线元素，形成一个向量。
        diagonal = scores.diag().view(im.size(0), 1)
        # 将对角线矩阵扩展为与scores相同大小的矩阵，其中每一行都是对角线元素
        d1 = diagonal.expand_as(scores)
        # 扩展为与scores相同大小的矩阵，其中每一列都是对角线元素，这里d1和d2一样的
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval, i->t
        # 三元组损失计算
        cost_s = (self.margin + scores - d1).clamp(min=0)

        # compare every diagonal score to scores in its row
        # image retrieval t->i
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        # 清除损失矩阵中对角线上的值
        if not self.mask_repeat:
            mask = torch.eye(scores.size(0), dtype=torch.bool, device=scores.device)
        else:
            img_ids = img_ids.cuda()
            mask = (img_ids.unsqueeze(1) == img_ids.unsqueeze(0))

        cost_s = cost_s.masked_fill_(mask, 0)
        cost_im = cost_im.masked_fill_(mask, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            # 每一行最大值
            cost_s, idx_s = cost_s.max(1)
            # 每一列最大值
            cost_im, idx_im = cost_im.max(0)
            return cost_s.sum() + cost_im.sum()



        loss = cost_s.sum() + cost_im.sum()
        # 获取batch大小
        batch_size = im.size(0)
        # 计算批次平均损失
        loss = loss / batch_size

        return loss


class ContrastiveLoss_(nn.Module):
    """
    Compute contrastive loss (max-margin based)
    """

    def __init__(self, opt, margin=0.2, max_violation=False):
        super(ContrastiveLoss_, self).__init__()
        self.opt = opt
        self.margin = margin
        self.max_violation = max_violation

    def max_violation_on(self):
        self.max_violation = True
        print('Use VSE++ objective.')

    def max_violation_off(self):
        self.max_violation = False
        print('Use VSE0 objective.')

    def forward(self, im, s):

        # 获取图像和文本的批量大小
        im_batch_size = im.size(0)
        s_batch_size = s.size(0)

        # 如果批量大小不同
        if im_batch_size != s_batch_size:
            # 填充较小的批次
            if im_batch_size < s_batch_size:
                # 填充图像特征
                im = torch.cat([im, torch.zeros(s_batch_size - im_batch_size, im.size(1)).to(im.device)], dim=0)
            else:
                # 填充文本特征
                s = torch.cat([s, torch.zeros(im_batch_size - s_batch_size, s.size(1)).to(s.device)], dim=0)

        # compute image-sentence score matrix
        scores = get_sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        loss = cost_s.sum() + cost_im.sum()
        # 获取batch大小
        batch_size = im.size(0)
        # 计算批次平均损失
        loss = loss / batch_size
        return loss

def get_sim(images, captions):

    similarities = images.mm(captions.t())
    return similarities


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, img_features, txt_features, img_momentum_features, txt_momentum_features, img_queue, txt_queue):
        """
        Args:
            img_features (Tensor): Batch of image query features, shape [batch_size, dim]
            txt_features (Tensor): Batch of text query features, shape [batch_size, dim]
            img_momentum_features (Tensor): Batch of image key features, shape [batch_size, dim]
            txt_momentum_features (Tensor): Batch of text key features, shape [batch_size, dim]
            img_queue (Tensor): Queue of negative image features, shape [dim, queue_len]
            txt_queue (Tensor): Queue of negative text features, shape [dim, queue_len]
        """
        img_features = nn.functional.normalize(img_features, dim=1)
        txt_features = nn.functional.normalize(txt_features, dim=1)
        img_momentum_features = nn.functional.normalize(img_momentum_features, dim=1)
        txt_momentum_features = nn.functional.normalize(txt_momentum_features, dim=1)
        img_queue = nn.functional.normalize(img_queue, dim=0)
        txt_queue = nn.functional.normalize(txt_queue, dim=0)
        batch_size, dim = img_features.shape
        device = img_features.device

        # Compute image-to-text logits
        l_pos_img_txt = torch.einsum('nc,nc->n', [img_features, txt_momentum_features]).unsqueeze(-1)  # [N, 1]
        l_neg_img_txt = torch.einsum('nc,ck->nk', [img_features, txt_queue.detach()])  # [N, K]
        logits_img_txt = torch.cat([l_pos_img_txt, l_neg_img_txt], dim=1)  # [N, 1+K]
        logits_img_txt /= self.temperature

        # Compute text-to-image logits
        l_pos_txt_img = torch.einsum('nc,nc->n', [txt_features, img_momentum_features]).unsqueeze(-1)  # [N, 1]
        l_neg_txt_img = torch.einsum('nc,ck->nk', [txt_features, img_queue.detach()])  # [N, K]
        logits_txt_img = torch.cat([l_pos_txt_img, l_neg_txt_img], dim=1)  # [N, 1+K]
        logits_txt_img /= self.temperature

        # Ground truth labels
        labels_img_txt = torch.zeros(batch_size, dtype=torch.long, device=device)
        labels_txt_img = torch.zeros(batch_size, dtype=torch.long, device=device)

        # Compute InfoNCE losses
        loss_img_txt = self.cross_entropy(logits_img_txt, labels_img_txt)
        loss_txt_img = self.cross_entropy(logits_txt_img, labels_txt_img)

        # Total loss
        loss = loss_img_txt + loss_txt_img

        return loss


# Triplet loss + DistanceWeight Miner
# Sampling Matters in Deep Embedding Learning, ICCV, 2017
# more information refer to https://kevinmusgrave.github.io/pytorch-metric-learning/miners/#distanceweightedminer
class TripletLoss(nn.Module):

    def __init__(self, opt=None, margin=0.2, ):
        super().__init__()

        self.opt = opt
        self.margin = margin
        
        self.cut_off = 0.5
        self.d = 512

        if opt.dataset == 'coco':
            self.nonzero_loss_cutoff = 1.9         
        else:
            self.nonzero_loss_cutoff = 1.7
        
    def forward(self, im, s, img_ids):

        sim_mat = get_sim(im, s)
        img_ids = img_ids.cuda()

        if im.size(0) == s.size(0):
            pos_mask, neg_mask = pos_neg_mask(img_ids)
        else:
            pos_mask, neg_mask = pos_neg_mask_xy(torch.unique(img_ids), img_ids)

        loss_im = self.loss_forward(sim_mat, pos_mask, neg_mask)
        loss_s = self.loss_forward(sim_mat.t(), pos_mask.t(), neg_mask.t())

        loss = loss_im + loss_s

        return loss        

    def loss_forward(self, sim_mat, pos_mask, neg_mask): 

        pos_pair_idx = pos_mask.nonzero(as_tuple=False)
        anchor_idx = pos_pair_idx[:, 0]
        pos_idx = pos_pair_idx[:, 1]

        dist = (2 - 2 * sim_mat).sqrt()
        dist = dist.clamp(min=self.cut_off)

        log_weight = (2.0 - self.d) * dist.log() - ((self.d - 3.0) / 2.0) * (1.0 - 0.25 * (dist * dist)).log()
        inf_or_nan = torch.isinf(log_weight) | torch.isnan(log_weight)

        log_weight = log_weight * neg_mask  
        log_weight[inf_or_nan] = 0.      

        weight = (log_weight - log_weight.max(dim=1, keepdim=True)[0]).exp()
        weight = weight * (neg_mask * (dist < self.nonzero_loss_cutoff)).float() 
     
        weight = weight / (weight.sum(dim=1, keepdim=True) + 1e-20)
        weight = weight[anchor_idx]

        # maybe not exist
        try:
            neg_idx = torch.multinomial(weight, 1).squeeze(1)   
        except Exception:
            return torch.zeros([], requires_grad=True, device=sim_mat.device) 


        s_ap = sim_mat[anchor_idx, pos_idx]
        s_an = sim_mat[anchor_idx, neg_idx]  

        loss = F.relu(self.margin + s_an - s_ap) 
        loss = loss.sum() 

        return loss
# SwAVContrastiveLoss
class SwAVContrastiveLoss(nn.Module):
    def __init__(self, opt):
        # 继承 PyTorch 的 Module 基类
        super(SwAVContrastiveLoss, self).__init__()
        # 原型向量作为可训练参数,初始化为随机值,形状为(原型数量, 嵌入维度)
        self.prototypes = nn.Parameter(torch.randn(opt.nmb_prototypes, opt.embed_size))
        # Sinkhorn 算法中的温度系数
        self.epsilon = opt.epsilon
        # Sinkhorn 算法的迭代次数
        self.sinkhorn_iterations = opt.sinkhorn_iterations
        # 对比损失的温度系数
        self.temperature = opt.temperature
        # self.freeze_prototypes_niters = opt.freeze_prototypes_niters
        # 队列初始化为零矩阵, 形状为(opt.queue_length, opt.embed_size)
        # 队列长度需要可被批次大小整除
        self.register_buffer("queue", torch.zeros(opt.queue_length, opt.embed_size))
        self.queue_start = 0  # 队列开始的索引
        self.queue_length = opt.queue_length
        # 添加变量用于决定何时开始使用队列
        self.iteration_queue_starts = opt.iteration_queue_starts

    def enqueue_dequeue(self, embeddings):
        # 将当前批次的嵌入添加到队列
        embeddings = embeddings.clone().detach()
        batch_size = embeddings.shape[0]
        end_idx = (self.queue_start + batch_size) % self.queue_length
        if end_idx >= self.queue_start:
            self.queue[self.queue_start:end_idx] = embeddings
        else:
            split = self.queue_length - self.queue_start
            self.queue[self.queue_start:] = embeddings[:split]
            self.queue[:end_idx] = embeddings[split:]

        # 更新队列开始的索引
        self.queue_start = (self.queue_start + batch_size) % self.queue_length

    def forward(self, img_embeddings, txt_embeddings, iteration):
        # 归一化 prototypes
        self.prototypes.data.copy_(F.normalize(self.prototypes, p=2, dim=1).data)

        # if iteration >= self.iteration_queue_starts:
        #     # 队列: 将嵌入加入队列并从队列获取先前的嵌入
        #     self.enqueue_dequeue(img_embeddings)
        #     self.enqueue_dequeue(txt_embeddings)
        #
        #     # 使用更新后的self.queue计算输出
        #     img_out = torch.matmul(self.queue, self.prototypes.t())
        #     txt_out = torch.matmul(self.queue, self.prototypes.t())
        # else:
        img_out = torch.matmul(img_embeddings, self.prototypes.t())
        txt_out = torch.matmul(txt_embeddings, self.prototypes.t())

        # # 计算图像嵌入与原型向量的分数矩阵
        # img_out = torch.matmul(img_embeddings, self.prototypes.t())
        # # 计算文本嵌入与原型向量的分数矩阵
        # txt_out = torch.matmul(txt_embeddings, self.prototypes.t())

        # 使用 Sinkhorn 算法计算图像的软分配矩阵
        img_q = self.sinkhorn(img_out)
        # 使用 Sinkhorn 算法计算文本的软分配矩阵
        txt_q = self.sinkhorn(txt_out)

        # 计算图像视角的对比损失,即文本分数与图像软分配矩阵的负交叉熵
        img_loss = -torch.sum(txt_q * torch.log_softmax(img_out / self.temperature, dim=1), dim=1).mean()
        # 计算文本视角的对比损失,即图像分数与文本软分配矩阵的负交叉熵
        txt_loss = -torch.sum(img_q * torch.log_softmax(txt_out / self.temperature, dim=1), dim=1).mean()

        # 将两个视角的损失相加作为最终的对比损失
        loss = img_loss + txt_loss

        # 冻结原型向量的梯度
        # if iteration < self.freeze_prototypes_niters:
        #     for name, p in self.named_parameters():
        #         if "prototypes" in name:
        #             p.grad = None

        return loss

    @torch.no_grad()
    def sinkhorn(self, out):
        # 对分数矩阵取指数并转置,得到 Q 矩阵,形状为(原型数量, 批次大小)
        Q = torch.exp(out / self.epsilon).t()  # Q is K-by-B
        B = Q.shape[1]  # 批次大小
        K = Q.shape[0]  # 原型的数量

        # 对 Q 矩阵进行归一化,使其元素和为 1
        sum_Q = torch.sum(Q)
        Q /= sum_Q

        # 执行 Sinkhorn 迭代
        for it in range(self.sinkhorn_iterations):
            # 对每一行(原型)进行归一化,使每个原型的总权重为 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q /= sum_of_rows
            Q /= K

            # 对每一列(样本)进行归一化,使每个样本的总权重为 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        # 最后将 Q 矩阵缩放,使每一列(样本)的和为 1,从而得到软分配矩阵
        Q *= B  # the colomns must sum to 1 so that Q is an assignment
        return Q.t()

# class InfoNCELoss(nn.Module):
#     """
#     Compute InfoNCELoss loss
#     """
#
#     def __init__(self, temperature=0.01, margin=0.2):
#         super(InfoNCELoss, self).__init__()
#         self.margin = margin
#         self.temperature = temperature
#
#     def forward(self, img, txt):
#         sims = txt.mm(img.t())
#
#         ## cost of image retrieval
#         img_ret = sims - sims.diag().expand_as(sims).t() + self.margin
#         img_ret[torch.eye(sims.size(0)) > .5] = 0
#         cost_im = torch.log(torch.sum(torch.exp(img_ret / self.temperature), dim=1))
#
#         ## cost of text retrieval
#         txt_ret = sims - sims.diag().expand_as(sims) + self.margin
#         txt_ret[torch.eye(sims.size(0)) > .5] = 0
#         cost_s = torch.log(torch.sum(torch.exp(txt_ret / self.temperature), dim=0))
#
#         return cost_s.mean() + cost_im.mean()

if __name__ == '__main__':

    pass
    