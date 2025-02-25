import torch.nn as nn
import torch.nn.functional as F
import torch, copy


# cross-modal interaction
# 跨模态交互
# for instance-level relation modeling
class AdjacencyLearning(nn.Module):
    def __init__(self, hidden_size=1024, dropout_rate=0., topk=20, T=1., 
                 sigma=1., img_len_use=False, res=True, scan=True):
        super(AdjacencyLearning, self).__init__()

        self.hidden_size = hidden_size
        # 只考虑前20个词
        self.topk = topk
        self.T = T
        
        self.sigma = sigma
        self.img_len_use = img_len_use
        self.res = res

        # the same operation from SCAN
        self.scan = scan

        self.mlp_t2i =  nn.Sequential(
                                    nn.Linear(topk, topk), 
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(dropout_rate),
                                    nn.Linear(topk, 1),
                                    )
        self.mlp_i2t = copy.deepcopy(self.mlp_t2i)
        

    def forward(self, img_regions, cap_words, img_len=None, cap_len=None):
        
        i2t_sim = []
        t2i_sim = []

        i2t_sim_mean = []
        t2i_sim_mean = []

        n_image = img_regions.size(0)
        n_caption = cap_words.size(0)
        # 确保图像和文本的批次大小相同
        assert n_image == n_caption

        # L2-norm
        img_regions = F.normalize(img_regions, dim=-1)

        for i in range(n_caption):
            
            # How many words are there actually
            # words的数量
            seq_word = cap_len[i]

            # To prevent the impact of too few words, topk is not enough
            # 只考虑前 20 个词
            if seq_word < self.topk:    
                num_selected_word = self.topk 
            else:
                num_selected_word = seq_word

            # (n_images, seq_word, dim)
            # 第 0 维（即行）重复n_image次，后面两维不变
            cap_i_expand = cap_words[i,:num_selected_word, :].repeat(n_image, 1, 1)
            cap_i_expand = F.normalize(cap_i_expand, dim=-1)

            # 计算得到细粒度相似度矩阵
            # (bs, seq_word, seq_region)
            # for word-region similarity-matrix，At the end of each line is padding, 
            # the similarity is calculated as 0. If the average value is calculated, they need to be ignored.
            # (n_images, seq_word, dim) * (bs, dim, seq_region) =(bs, seq_word, seq_region)
            # 位置(a, b, c) 表示 第 a 张图片与第 i 个句子，的第 b 个词语与第 c 个区域的相似度
            cap2img_sim = torch.bmm(cap_i_expand, img_regions.transpose(1, 2))

            if self.scan:
                # from SCAN
                # 负斜率参数negative_slope = 0.1，表示当输入为负数时，斜率为0.1。
                cap2img_sim = F.leaky_relu(cap2img_sim, negative_slope=0.1)

            # print(cap2img_sim.shape)

            # t2i
            # for every word，find the region with the max similairity, (bs, seq_word)
            if self.scan:
            # word 行 l2标准化
                cap2img_sim_norm = F.normalize(cap2img_sim, dim=1)
            else:
                cap2img_sim_norm = cap2img_sim

            # consider the variable lengths for region features,
            # this will increase training time.
            # 不考虑
            if self.img_len_use:
                row_sim = []
                for i in range(n_image):
                    seq_region = img_len[i]
                    #(seq_word, seq_region_each)
                    cap2img_sim_each = cap2img_sim_norm[i, :, 0:seq_region]
                    #(seq_word, )
                    row_sim_each = cap2img_sim_each.max(dim=1)[0]   
                    row_sim.append(row_sim_each)
                
                # (bs, seq_word)
                row_sim = torch.stack(row_sim, dim=0) 
            else:
                # 沿着第2个维度（即图像区域的维度）进行了最大化操作，返回了每个单词对应的最大相似性值
                row_sim = cap2img_sim_norm.max(dim=2)[0]     

            # get the average of word-region
            # 计算这个样本的单词与所有样本图像区域相似性最大值的平均值
            # (bs, )
            row_sim_mean = row_sim[:, 0:seq_word].mean(dim=1)

            # top-K word-region scores, (bs, K)
            # 计算这个样本的单词与所有图像区域相似性最大值的前k个最大值
            row_sim_topk = row_sim.topk(k=self.topk, dim=1, largest=True)[0]
            
            # i2t 
            #  (bs, seq_region)

            if self.scan:
                # region 列 l2标准化
                cap2img_sim_norm = F.normalize(cap2img_sim, dim=2)
            else:
                cap2img_sim_norm = cap2img_sim
            # 沿着第1个维度（即word的维度）进行了最大化操作，返回了每个图像区域对应的最大相似性值
            column_sim = cap2img_sim_norm[:, 0:seq_word, :].max(dim=1)[0]
            
            # Average the matching results of each region-word to get the region-word similarity score
            # Process each image separately, this will be slow and inappropriate
            # 不考虑
            if self.img_len_use:
                column_sim_mean = []
                column_sim_topk = []
                for i in range(n_image):
                    seq_region = img_len[i]
                    column_sim_each = column_sim[i][0:seq_region]
                    mean_each = column_sim_each.mean()
                    topk_each = column_sim_each.topk(k=self.topk, dim=0, largest=True)[0]

                    column_sim_mean.append(mean_each)
                    column_sim_topk.append(topk_each)

                # (bs, )
                column_sim_mean = torch.stack(column_sim_mean, dim=0)
                column_sim_topk = torch.stack(column_sim_topk, dim=0)

            else:
                # 计算每个样本的图像区域与这个样本word相似性最大值的平均值
                # (bs, )
                column_sim_mean = column_sim.mean(dim=1)
                # top-K word-region scores, (bs, K)
                # 计算每个样本的图像区域与这个样本word相似性最大值的前k个最大值
                column_sim_topk = column_sim.topk(k=self.topk, dim=1, largest=True)[0]

            # It is a directed graph, and the edge weights of t2i and i2t are different
            # 记录(bs, bs, K), (textual, visual, K)图像到文本的细粒度相似度 topk
            i2t_sim.append(column_sim_topk)
            # 记录(bs, bs, K), (textual, visual, K)文本到图像的细粒度相似度 topk
            t2i_sim.append(row_sim_topk)

            # 记录(bs, bs), (textual, visual, K)图像到文本的细粒度相似度
            i2t_sim_mean.append(column_sim_mean)
            # 记录(bs, bs), (textual, visual)文本到图像的细粒度相似度
            t2i_sim_mean.append(row_sim_mean)
            
        # (bs, bs, K), (textual, visual, K)
        # t2i: given t, judge whether each i needs to interact with me (whether it should affect me)
        # 转化为张量
        batch_t2i = torch.stack(t2i_sim, 0)

        # (bs, bs), (textual, visual)
        # t2i connection
        batch_t2i_mean = torch.stack(t2i_sim_mean, 0)

        # i2t: given i, judge whether each t needs to interact with me (whether it should affect me)
        # (bs, bs, K), (textual, visual, K) -> (bs, bs, K), (visual, textual, K)
        batch_i2t = torch.stack(i2t_sim, 0).transpose(0, 1)

        # (bs, bs), (visual, textual)
        # i2t connection
        batch_i2t_mean = torch.stack(i2t_sim_mean, 0).transpose(0, 1)

        # the relevance relation learning,
        # the input edge weight is a vector of length K, and the output is scalar
        # (textual, visual), t2i relevance weight
        # 关联性计算
        # 将 topk 输入到mlp中得到 关联性
        # "squeeze(-1)": 张量中最后一个维度为1的维度压缩掉，从而将张量转换为标量
        batch_t2i_relation = self.mlp_t2i(batch_t2i).squeeze(-1)
        # (visual, textual), i2t relevance weight
        batch_i2t_relation = self.mlp_i2t(batch_i2t).squeeze(-1)    

        if self.res:
            # 残差连接
            batch_t2i_relation += batch_t2i_mean
            batch_i2t_relation += batch_i2t_mean

        # 返回跨模态细粒度关联性以及均值
        return batch_t2i_relation, batch_i2t_relation, batch_t2i_mean, batch_i2t_mean

    # 全局特征得到的相似性矩阵与细粒度特征得到的相似性矩阵使用KL散度对齐
    # give a constraint
    def relation_reg_loss(self, batch_t2i_relation, batch_i2t_relation, batch_img_emb, batch_cap_emb,):

        # (bs, bs), (visual, textual), as the guide, without the gradient flow
        batch_sim_matrix = torch.exp(-torch.cdist(batch_img_emb, batch_cap_emb) / self.sigma).detach()
        # 全局特征得到的相似性矩阵与细粒度特征得到的相似性矩阵使用KL散度对齐
        reg_i2t = self.kl_div(batch_sim_matrix, batch_i2t_relation)

        reg_t2i = self.kl_div(batch_sim_matrix.t(), batch_t2i_relation)

        loss = reg_i2t + reg_t2i

        return loss

    # the basic kl divergence loss
    def kl_div(self, A, B):

        log_p_A = F.log_softmax(A/self.T, dim=-1)
        p_B = F.softmax(B/self.T, dim=-1)

        kl_div  = F.kl_div(log_p_A, p_B, reduction='batchmean') * (self.T**2) 

        return kl_div

# 学习图像和文本之间的连接关系和相关性关系
# give four matrix: connection matrix and relevance matrix
# intra-modal: i2i, t2t
# inter-modal:  i2t, t2i
class AdjacencyModel(nn.Module):
    # 隐藏层大小、阈值、topk、是否分离、温度系数、sigma
    def __init__(self, hidden_size=1024, threshold=0.5, topk=10, detach=True, T=1., sigma=1., ):
        super(AdjacencyModel, self).__init__()

        self.hidden_size = hidden_size
        # 用于确定连接关系的阈值
        self.threshold = threshold
        
        self.detach = detach
        # 计算相似性关系的高斯核的标准差
        self.sigma = sigma

        # 得到跨模态细粒度关联性以及均值
        self.adj_learning = AdjacencyLearning(hidden_size=hidden_size, dropout_rate=0., topk=topk, T=T, sigma=sigma, res=True)
    
    # img_emb, cap_emb:  batch_size * feature_dim       全局特征，使用局部细粒度特征聚合得到的
    # img_regions, cap_words:  batch_size * num_seq * feature_dim   局部细粒度特征
    def forward(self, img_emb, cap_emb, img_regions, cap_words, img_len, cap_len):

        n_img = img_emb.size(0)
        n_cap = cap_emb.size(0)

        # 连接性
        # step-1 get connection relation
        # two situations：single-modal，cross-modal
        with torch.no_grad():
            # 单模态
            batch_sim_t2t = torch.matmul(cap_emb, cap_emb.t())
            batch_sim_i2i = torch.matmul(img_emb, img_emb.t())  

            # Boolean type matrix returned by logical judgment
            # 取每行的前k个最大值 然后通过 [:, -1:] 选择每行中的最后一个值 相似性矩阵中大于等于这个阈值的元素设为 1，其余为 0
            batch_t2t_connect = (batch_sim_t2t - batch_sim_t2t.topk(k=int(n_cap*self.threshold), dim=1, largest=True)[0][:, -1:]) >= 0
            batch_i2i_connect = (batch_sim_i2i - batch_sim_i2i.topk(k=int(n_img*self.threshold), dim=1, largest=True)[0][:, -1:]) >= 0

        # step-2 get relevance relation
        # two situations：single-modal，cross-modal
        # 模态内相似度使用欧式距离  cdist 即欧式距离计算公式 对应论文公式  (10)
        batch_i2i_relation = torch.exp(-torch.cdist(img_emb, img_emb) / self.sigma)
        batch_t2t_relation = torch.exp(-torch.cdist(cap_emb, cap_emb) / self.sigma)

        if self.detach:
            img_regions = img_regions.detach()
            cap_words = cap_words.detach()

        batch_t2i_relation, batch_i2t_relation, batch_t2i_mean, batch_i2t_mean = self.adj_learning(img_regions, 
                                                                                                   cap_words, 
                                                                                                   img_len, 
                                                                                                   cap_len)
        batch_t2i_relation = batch_t2i_mean
        batch_i2t_relation = batch_i2t_mean

        # get the connection relation by the given threshold
        # 取每行的前k个最大值 然后通过 [:, -1:] 选择每行中的最后一个值 相似性矩阵中大于等于这个阈值的元素设为 1，其余为 0
        batch_t2i_connect = (batch_t2i_mean - batch_t2i_mean.topk(k=int(n_img*self.threshold), dim=1, largest=True)[0][:, -1:]) >= 0
        batch_i2t_connect = (batch_i2t_mean - batch_i2t_mean.topk(k=int(n_cap*self.threshold), dim=1, largest=True)[0][:, -1:]) >= 0
        # KL散度，正则化损失
        reg_loss = self.adj_learning.relation_reg_loss(batch_t2i_relation, batch_i2t_relation, img_emb, cap_emb)

        # get all the relation matrix
        batch_connect = {
            'i2i':batch_i2i_connect,
            't2t':batch_t2t_connect,
            'i2t':batch_i2t_connect,
            't2i':batch_t2i_connect,
        }

        batch_relation = {
            'i2i':batch_i2i_relation,
            't2t':batch_t2t_relation,
            'i2t':batch_i2t_relation,
            't2i':batch_t2i_relation,
        }

        return batch_connect, batch_relation, reg_loss


if __name__ == '__main__':

    pass
    
