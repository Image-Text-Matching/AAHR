import json

import torch
import torch.utils.data as data
import os
import numpy as np
import random
import logging
from PIL import Image
logger = logging.getLogger(__name__)


class PrecompRegionDataset(data.Dataset):
    def __init__(self, data_path, data_split, tokenizer, clip_tokenizer, preprocess, opt, train):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.clip_tokenizer = clip_tokenizer
        self.preprocess = preprocess
        self.opt = opt
        # True 或者 False 判断是否是训练集
        self.train = train

        # loc = os.path.join(data_path, '{}_precomp'.format(opt.dataset))
        if 'coco' in opt.dataset:
            data_base = os.path.join(data_path, 'coco')
        else:
            data_base = os.path.join(data_path, 'f30k')
        loc = os.path.join(data_base, 'precomp')
        loc_mapping = os.path.join(data_base, 'id_mapping.json')

        if 'coco' in opt.dataset:
            self.image_base = os.path.join(data_base, 'images')
        else:
            self.image_base = os.path.join(data_base, 'flickr30k-images')

        with open(loc_mapping, 'r') as f_mapping:
            self.id_to_path = json.load(f_mapping)

        # Raw captions
        # 保存加载的原始图像描述
        self.captions = []

        with open(os.path.join(loc, '%s_caps.txt' % data_split), 'r', encoding='utf-8') as f:
            for line in f:
                self.captions.append(line.strip())  # line.strip() 表示去除头尾的空格

        # Region features
        # 保存加载的细粒度图像特征
        self.images = np.load(os.path.join(loc, '%s_ims.npy' % data_split))

        # Get the 全局 image ids
        with open(os.path.join(loc, '{}_ids.txt'.format(data_split)), 'r') as f:
            image_ids = f.readlines()
            self.full_images = [int(x.strip()) for x in image_ids]


        # num_captions
        # 保存数据集中即图像描述的数量
        self.length = len(self.captions)
        # 保存图像的数量
        self.num_images = len(self.images)

        if self.num_images != self.length:
            # one images to five captions (train set)
            self.im_div = 5
        else:
            # one images to one captions (test set)
            self.im_div = 1

        if data_split == 'dev':
            self.length = 5000

    # 用于从数据集中获取单个样本
    def __getitem__(self, index):

        # handle the image redundancy
        # index for captions, img_index for images
        # 根据图像描述索引 index 获取对应的图像索引
        img_index = index // self.im_div
        # 根据索引 index 获取对应的图像描述
        caption = self.captions[index]
        # 使用clip编码器进行编码
        clip_caption = self.clip_tokenizer(caption)
        # 使用分词器对图像描述进行分词，将其转换为单词或子词的列表
        caption_tokens = self.tokenizer.basic_tokenizer.tokenize(caption)

        # Convert caption (string) to word ids (with Size Augmentation at training time).
        # 用于处理图像描述，并将其转换为BERT模型可接受的输入格式，分词后每个 token 都映射到词汇表中对应的索引
        target = process_caption_bert(self.tokenizer, caption_tokens, self.train)

        # 获取对应的细粒度图像特征
        image = self.images[img_index]

        # 获取全局图像特征
        image_id = self.full_images[img_index]
        image_path = os.path.join(self.image_base, self.id_to_path[str(image_id)])
        full_image = self.preprocess(Image.open(image_path)).unsqueeze(0)
        # 随机丢弃20%的图像区域特征，用于大小扩增，增加随机性，从而提高模型的鲁棒性
        if self.train and self.opt.size_augment:
            num_features = image.shape[0]
            rand_list = np.random.rand(num_features)
            image = image[np.where(rand_list > 0.20)]

        image = torch.Tensor(image)

        return image, full_image, target, clip_caption, index, img_index

    def __len__(self):
        return self.length


# 用于处理图像描述，并将其转换为BERT模型可接受的输入格式
def process_caption_bert(tokenizer, tokens, train=True):
    # 初始化两个空列表，用于存储处理后的标记（tokens）和被删除的标记的索引
    output_tokens = []
    deleted_idx = []

    for i, token in enumerate(tokens):
        # text -> token (basic_tokenizer.tokenize) -> sub_token (wordpiece_tokenizer.tokenize)
        # 使用BERT分词器将原始标记token分割成子标记（sub - tokens），比如把"apple"分成"app", "##le"
        sub_tokens = tokenizer.wordpiece_tokenizer.tokenize(token)

        prob = random.random()
        # 20 % 的概率进行数据增强操作
        # first, 20% probability use the augmenation operations
        if prob < 0.20 and train:  # mask/remove the tokens only during training
            prob /= 0.20

            # 50% change token to mask token
            # 50 % 的概率将子标记替换为[MASK]标记
            if prob < 0.5:
                for sub_token in sub_tokens:
                    output_tokens.append("[MASK]")
            # 10% randomly change token to random token from the BERT-vocab
            # 10 % 的概率将子标记随机替换为词汇表中的一个随机标记
            elif prob < 0.6:
                for sub_token in sub_tokens:
                    output_tokens.append(random.choice(list(tokenizer.vocab.keys())))
                    # -> 40% delete the token
            # 40 % 的概率删除子标记，模拟部分标记丢失或遮盖的情况
            else:
                for sub_token in sub_tokens:
                    output_tokens.append(sub_token)
                    # record the index of sub_token
                    # 记录sub_token的索引
                    deleted_idx.append(len(output_tokens) - 1)

        # 80% probability keep the token
        else:
            for sub_token in sub_tokens:
                # no masking token (will be ignored by loss function later)
                output_tokens.append(sub_token)

    # 去除被删除的子标记
    if len(deleted_idx) != 0:
        output_tokens = [output_tokens[i] for i in range(len(output_tokens)) if i not in deleted_idx]

    # and first and last notations for BERT model
    # 添加[CLS]和[SEP]标记，以符合 BERT 模型的输入要求
    output_tokens = ['[CLS]'] + output_tokens + ['[SEP]']

    # Convert token to vocabulary indices, torch.float32
    # 转换为词汇表中的索引序列。这个操作将标记列表中的每个标记都映射到词汇表中对应的索引
    target = tokenizer.convert_tokens_to_ids(output_tokens)

    # convert to the torch-tenfor
    # 索引序列转换为PyTorch张量，作为返回值
    target = torch.Tensor(target)
    return target


# 用于将一个批次（batch）的数据进行整理和处理
def collate_fn(data):
    # Sort a data list by caption length
    # 根据图像描述的长度对数据进行排序，以便在后续处理中能够更高效地处理不同长度的描述
    data.sort(key=lambda x: len(x[1]), reverse=True)

    images, full_image, captions, clip_caption, ids, img_ids = zip(*data)

    full_images = torch.cat(full_image, 0)
    clip_captions = torch.cat(clip_caption, 0)

    img_ids = torch.tensor(img_ids)
    ids = torch.tensor(ids)

    # 计算的结果就是重复出现的图像ID的数量
    # print(img_ids)
    repeat = len(img_ids) - len(torch.unique(img_ids))

    # Sort a data list by caption length
    # Merge images (convert tuple of 3D tensor to 4D tensor)
    # images = torch.stack(images, 0)
    # 将图像数据进行填充，使得所有图像的长度保持一致
    img_lengths = [len(image) for image in images]

    # dataset_size * max_lengths (maybe 36) * 2048 
    all_images = torch.zeros(len(images), max(img_lengths), images[0].size(-1))
    for i, image in enumerate(images):
        end = img_lengths[i]
        all_images[i, :end] = image[:end]

    img_lengths = torch.tensor(img_lengths)

    # 对描述数据进行填充，使得所有描述的长度保持一致，并将它们组合成一个二维张量
    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    # count the length of each captions
    lengths = [len(cap) for cap in captions]

    # pad the redundancy with zero, in order to input BERT model as a batch
    targets = torch.zeros(len(captions), max(lengths)).long()

    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    lengths = torch.tensor(lengths)

    # 批次大小，图像的数量,图像中最大的区域特征数量,每个区域特征的维度(2048)
    # 批次大小，图像的数量,所有描述中最大的长度
    # all_images: Batch_size * max_img_lengths * 2048 (the dimension of region-features)
    # targets:  Batch_size * max_cap_lengths
    return all_images, full_images, img_lengths, targets, clip_captions, lengths, ids, img_ids, repeat


# 通用的数据加载器函数，用于加载数据集
def get_loader(data_path, data_split, tokenizer,  clip_tokenizer, preprocess, opt, batch_size=100,
               shuffle=True, num_workers=2, train=True):
    drop_last = True if train else False

    dset = PrecompRegionDataset(data_path, data_split, tokenizer,  clip_tokenizer, preprocess, opt, train)

    # pin_memory = True: 指定了是否将数据加载到固定的内存区域中，对于 GPU 训练来说，可以提高数据传输效率。
    # collate_fn，用于对每个批次的数据进行预处理和组合
    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              collate_fn=collate_fn,
                                              num_workers=num_workers,
                                              drop_last=drop_last)

    return data_loader


# 用于获取训练集的数据加载器
def get_train_loader(data_path, tokenizer, clip_tokenizer, preprocess, batch_size, workers, opt):
    train_loader = get_loader(data_path, 'train', tokenizer,  clip_tokenizer, preprocess,
                              opt, batch_size, True, workers, train=True)

    return train_loader


# 用于获取测试集的数据加载器
def get_test_loader(data_path, split_name, tokenizer, clip_tokenizer, preprocess, batch_size, workers, opt):
    test_loader = get_loader(data_path, split_name, tokenizer,  clip_tokenizer, preprocess,
                             opt, batch_size, False, workers, train=False)

    return test_loader


if __name__ == '__main__':
    pass
