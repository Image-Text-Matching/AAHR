import arguments
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init
from torch.nn.utils import clip_grad_norm_

from lib.encoders import get_image_encoder, get_text_encoder
from lib.loss import *
from graph_lib import *
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.init
import torch.backends.cudnn as cudnn
import logging
import copy


logger = logging.getLogger(__name__)


class VSEModel(nn.Module):
    def __init__(self, opt, eval=False):
        super().__init__()
        model_path = 'CLIP/open_clip_pytorch_model.bin'
        self.queue_img_ids = None  # 初始化为 None
        self.opt = opt
        self.grad_clip = opt.grad_clip
        # 创建图像编码器 文本编码器
        # no_imgnorm no_txtnorm 表示特征是否正则化
        self.moco_loss = InfoNCELoss()
        self.img_enc = get_image_encoder(opt, opt.img_dim, opt.embed_size, no_imgnorm=opt.no_imgnorm)
        self.txt_enc = get_text_encoder(opt, opt.embed_size, no_txtnorm=opt.no_txtnorm)
        self.count = 0
        if opt.use_moco:
            self.K = opt.moco_M
            self.m = opt.moco_r
            self.v_encoder_k = copy.deepcopy(self.img_enc)
            self.t_encoder_k = copy.deepcopy(self.txt_enc)
            for param in self.v_encoder_k.parameters():
                param.requires_grad = False
            for param in self.t_encoder_k.parameters():
                param.requires_grad = False
            self.register_buffer("t_queue", torch.rand(opt.embed_size, self.K))
            self.t_queue = F.normalize(self.t_queue, dim=0)
            self.register_buffer("v_queue", torch.rand(opt.embed_size, self.K))
            self.v_queue = F.normalize(self.v_queue, dim=0)
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        if not eval:
            # 定义损失函数
            self.criterion = GraphLoss(opt)
            print(self.criterion)

            # Set up the lr for different parts of the VSE model
            # 权重衰减因子
            decay_factor = 1e-4
            # 交互模型的学习率因子
            criterion_lr_factor = opt.graph_lr_factor
            # 模型参数
            self.params = list(self.txt_enc.parameters()) + list(self.img_enc.parameters()) + list(self.criterion.parameters())

            all_text_params = list(self.txt_enc.parameters())
            bert_params = list(self.txt_enc.bert.parameters())

            # Tensor.data_ptr() → int, Returns the address of the first element
            bert_params_ptr = [p.data_ptr() for p in bert_params]
            text_params_no_bert = list()

            # select other parameters except BERT
            for p in all_text_params:
                if p.data_ptr() not in bert_params_ptr:
                    text_params_no_bert.append(p)
            # 定义优化器
            self.optimizer = torch.optim.AdamW([
                {'params': text_params_no_bert, 'lr': opt.learning_rate},
                {'params': bert_params, 'lr': opt.learning_rate * 0.1},# bert 部分的学习率降低10倍
                {'params': self.img_enc.parameters(), 'lr': opt.learning_rate},
                {'params': self.criterion.parameters(), 'lr': opt.learning_rate * criterion_lr_factor},
            ],
                lr=opt.learning_rate, weight_decay=decay_factor)
        # 评估模式，不计算实例交互损失
        else:
            self.criterion = ContrastiveLoss(opt)    
        
        # iteration
        self.Eiters = 0
        self.data_parallel = False

        # use the gpu
        # 转移到 GPU 上
        # 设置PyTorch使用cuDNN的自动调整功能，以提高训练速度
        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            self.criterion.cuda()
            if opt.use_moco:
                self.v_encoder_k.cuda()
                self.t_encoder_k.cuda()
                self.t_queue = self.t_queue.cuda()
                self.v_queue = self.v_queue.cuda()
                self.queue_ptr = self.queue_ptr.cuda()
            torch.backends.cudnn.benchmark = True

    # 设定在排名损失中使用max而不是sum
    def set_max_violation(self, max_violation=True):
        
        if max_violation:
            # 设定初始特征的损失
            if self.opt.base_loss == 'vse':
                self.criterion.base_loss.max_violation_on()
            # 设定强化特征的损失
            if self.opt.gnn_loss == 'vse':
                self.criterion.gnn_loss.max_violation_on()
        else:
            if self.opt.base_loss == 'vse':
                self.criterion.base_loss.max_violation_off()
            if self.opt.gnn_loss == 'vse':
                self.criterion.gnn_loss.max_violation_off()

    # 模型的状态字典是一个Python字典，它保存了模型的所有可学习参数（例如权重和偏置）以及其他状态信息
    # 返回模型的当前状态字典
    def state_dict(self):
        state_dict = [
            self.img_enc.state_dict(), 
            self.txt_enc.state_dict(), 
            ]
        return state_dict

    # 加载模型的状态字典
    def load_state_dict(self, state_dict, ):
        # strict=True, ensure keys match
        self.img_enc.load_state_dict(state_dict[0], strict=True)
        
        # Unexpected key(s) in state_dict: "bert.embeddings.position_ids". 
        # incompatible problem of transformers package version 
        self.txt_enc.load_state_dict(state_dict[1], strict=False)

    # 设置模型处于训练模式
    def train_start(self):
        self.img_enc.train()
        self.txt_enc.train()
        self.criterion.train()

    # 设置模型处于验证模式。在验证模式下，模型不会启用dropout 和batchnormalization层
    def val_start(self):
        self.img_enc.eval()
        self.txt_enc.eval()
        self.criterion.eval()

    # 并行化
    def make_data_parallel(self):
        self.img_enc = nn.DataParallel(self.img_enc)
        self.txt_enc = nn.DataParallel(self.txt_enc)
        self.data_parallel = True
        logger.info('Image/Text encoder is data paralleled (use multi GPUs).')

    @property
    def is_data_parallel(self):
        return self.data_parallel

    # Compute the image and caption embeddings
    # 用于计算图像和标题的嵌入
    def forward_emb(self, images,  full_imag, captions, clip_captions, img_ids, lengths, image_lengths=None):

        # compute images embs
        images = images.cuda()         
        image_lengths = image_lengths.cuda()
        full_imag = full_imag.cuda()
        img_emb = self.img_enc(images, full_imag, image_lengths)

        # compute caption embs
        captions = captions.cuda()
        lengths = lengths.cuda()
        clip_captions = clip_captions.cuda()
        cap_emb = self.txt_enc(captions, clip_captions, lengths)

        return img_emb, cap_emb

    # One training step given images and captions
    # 执行一个训练步骤，包含反向传播
    def train_emb(self, images, full_imag, captions, clip_captions, lengths, image_lengths=None, img_ids=None):
        # 追踪训练的迭代次数
        self.Eiters += 1
        # 更新日志记录器中的迭代次数
        self.logger.update('Iter', self.Eiters)
        # 更新日志记录器中的学习率
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute images embs
        images = images.cuda()         
        image_lengths = image_lengths.cuda()
        full_imag = full_imag.cuda()
        img_emb = self.img_enc(images, full_imag, image_lengths, graph=True)

        # compute caption embs
        captions = captions.cuda()
        lengths = lengths.cuda()
        clip_captions = clip_captions.cuda()
        cap_emb = self.txt_enc(captions, clip_captions, lengths, graph=True)

        # 清除之前的梯度
        self.optimizer.zero_grad()

        # compute loss
        loss = self.criterion(img_emb, cap_emb, img_ids=img_ids)
        img_feat, img_len, clip_img_emb, image_emb, img_emb_pre_pool = img_emb
        cap_feat, cap_len, clip_cap_emb, caption_emb, cap_emb_pre_pool = cap_emb

        if self.opt.use_moco:

            with torch.no_grad():
                self._momentum_update_key_encoder()
                v_embed_k = self.v_encoder_k(images, full_imag, image_lengths, graph=False)
                t_embed_k = self.t_encoder_k(captions, clip_captions, lengths, graph=False)
                # 合并当前批次和队列中的 img_ids
            all_img_ids = torch.cat([img_ids, self.queue_img_ids]) if self.queue_img_ids is not None else img_ids
            self.count += 1
            if self.count > self.K / self.opt.batch_size:
                loss_moco = self.moco_loss(image_emb, caption_emb, v_embed_k, t_embed_k, self.v_queue, self.t_queue)
            else:
                loss_moco = 0
            self._dequeue_and_enqueue(v_embed_k, t_embed_k, img_ids)
            loss += loss_moco
            if loss_moco != 0:
                print('moco_loss',  loss_moco.data.item())
            else:
                print('moco_loss', loss_moco)

        # 更新日志记录器中的训练损失
        self.logger.update('Loss', loss.item(), self.opt.batch_size)
        print('all_loss', loss.item())

        # compute gradient and update
        if torch.isnan(loss):
            logger.error("We have NaN numbers, ")
            return 0.

        # 反向传播
        loss.backward()

        if self.grad_clip > 0:
            # 梯度裁剪将梯度限制在一定范围内
            clip_grad_norm_(self.params, self.grad_clip)
        # 使用优化器更新模型参数
        self.optimizer.step()


    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.img_enc.parameters(), self.v_encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)
        for param_q, param_k in zip(self.txt_enc.parameters(), self.t_encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, v_keys, t_keys, img_ids):
        batch_size = v_keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.v_queue[:, ptr: ptr + batch_size] = v_keys.T
        self.t_queue[:, ptr: ptr + batch_size] = t_keys.T

        # 更新 queue_img_ids
        if self.queue_img_ids is None:
            self.queue_img_ids = img_ids
        else:
            self.queue_img_ids = torch.cat([self.queue_img_ids, img_ids], dim=0)
            if self.queue_img_ids.numel() > self.K:  # 确保队列长度不超过 K
                self.queue_img_ids = self.queue_img_ids[-self.K:]

        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr[0] = ptr

if __name__ == '__main__':

    pass
    



