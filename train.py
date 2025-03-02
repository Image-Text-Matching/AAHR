import os
import time
import numpy as np
import torch
import logging
# BertTokenizer是用于对文本进行分词和编码的工具
from transformers import BertTokenizer
# 打印 tensorboard 日志
import tensorboard_logger as tb_logger
# 用于处理命令行参数或者配置文件中的参数
import arguments
# 评估模型性能
from lib import evaluation
from lib import image_caption
from lib.encoders import ClipEncoder
# 多模态检索模型的实现
from lib.vse import VSEModel
from lib.evaluation import i2t, t2i, AverageMeter, LogCollector, encode_data, compute_sim
from graph_lib import *



def main():

    # Hyper Parameters
    # 超参设定
    parser = arguments.get_argument_parser()
    parser = extra_parameters(parser)
    # 使用参数解析器解析命令行参数，并将解析结果保存在opt变量中
    opt = parser.parse_args()
    # 指定日志和检查点的存储路径一致
    opt.model_name = opt.logger_name

    # set the gpu-id for training
    # 如果未开启多GPU模式，则设置CUDA可见设备为指定的GPU ID 默认为0
    if not opt.multi_gpu:
        torch.cuda.set_device(opt.gpu_id)

    # create the folder for logger and checkpoint
    # 如果指定的模型文件夹不存在，则创建该文件夹，用于存储训练过程中的日志和检查点文件
    if not os.path.exists(opt.model_name):
        os.makedirs(opt.model_name)

    # initialize logger
    # 初始化日志记录器和TensorBoard日志记录器
    # 配置了日志记录器，指定了日志文件的路径, 日志记录方式为覆盖写入,日志记录方式为覆盖写入
    logging.basicConfig(filename=os.path.join(opt.logger_name, 'train.txt'), filemode='w', 
                        format='%(asctime)s %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)
    # 记录了命令行参数信息opt
    logger.info(opt)

    # 配置了TensorBoard日志记录器，指定了日志存储的路径为opt.logger_name，并设置了每5秒刷新一次日志
    tb_logger.configure(opt.logger_name, flush_secs=5)

    # record parameters
    # 用于将参数信息保存到 logger_name 路径下的 Parameters.txt 的文件中
    arguments.save_parameters(opt, opt.logger_name)

    # load tokenizer for TextEncoder
    # 从本地加载一个预训练的BERT模型的分词器（tokenizer）
    # tokenizer = BertTokenizer.from_pretrained(opt.bert_path) 
    tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased')
    clip = ClipEncoder()
    clip_tokenizer = clip.clip_tokenizer
    preprocess = clip.preprocess
    # get the train-set
    # 调用了image_caption模块中的get_train_loader函数，用于获取训练集的数据加载器。参数包括了数据路径opt.data_path、分词器tokenizer、
    # 批量大小opt.batch_size、工作进程数opt.workers以及其他参数opt
    train_loader = image_caption.get_train_loader(opt.data_path, tokenizer,
                                                  clip_tokenizer, preprocess, opt.batch_size, opt.workers, opt)

    # get the test-set
    split = 'dev'
    # 用于获取测试集的数据加载器
    test_loader = image_caption.get_test_loader(opt.data_path, split, tokenizer,
                                                clip_tokenizer, preprocess, opt.batch_size, opt.workers, opt)
    # 使用日志记录器记录了训练集中的图像数量
    logger.info('Number of images for train-set: {}'.format(train_loader.dataset.num_images))

    # load the multi-modal model
    # 加载了多模态模型
    model = VSEModel(opt)

    start_epoch = 0

    # use the multi gpu
    # 是否进行多 GPU 并行运算
    if (not model.is_data_parallel) and opt.multi_gpu:
        model.make_data_parallel()

    best_rsum = 0

    # start the training process
    # 开始训练
    for epoch in range(start_epoch, opt.num_epochs):

        if epoch == 0:
            logger.info('Log saving path: ' + opt.logger_name)
            logger.info('Models saving path: ' + opt.model_name)

        # 根据当前 epoch 调整学习率
        adjust_learning_rate(opt, model.optimizer, epoch)

        # set hard negative for vse loss
        # 设置VSE损失函数的 hard negative
        if (epoch >= opt.vse_mean_warmup_epochs):
            opt.max_violation = True    # 在排名损失中使用 max 而不是 sum。
            model.set_max_violation(opt.max_violation)

        # train for one epoch
        # 训练
        train(opt, train_loader, model, epoch)

        # evaluate on test set for every epoch
        # 对模型在测试集上进行评估
        rsum = validate(opt, test_loader, model)

        # remember best rsum and save checkpoint
        # 检查当前rsum是否超过历史最佳rsum，如果是，则更新最佳rsum，并记录此时的epoch 和最佳 rsum
        is_best = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)

        logger.info("Epoch: [{}], Best rsum: {:.1f}".format(epoch, best_rsum))

        # save the checkpoint
        # 'model': 保存了模型的状态字典，通过model.state_dict()获取，即模型的权重参数
        # 'opt': 保存了训练选项opt，其中包含了训练过程中使用的所有超参数和配置信息
        # 'best_rsum': 保存了历史最佳性能指标 best_rsum，即在验证集上的最高得分。
        # 'Eiters': 保存了模型的迭代次数
        state = {'model': model.state_dict(), 'opt': opt, 'epoch': epoch + 1, 'best_rsum': best_rsum, 'Eiters': model.Eiters}
        save_checkpoint(state, is_best, prefix=opt.model_name)

    logger.info('Train finish.')    

    # evaluation after training process
    # 测试
    logger.info('Evaluate the model')
    
    base = opt.logger_name
    # 配置了日志记录器，指定了测试日志文件的路径, 日志记录方式为覆盖写入,日志记录方式为覆盖写入
    logging.basicConfig(filename=os.path.join(base, 'eval.txt'), filemode='w', 
                        format='%(asctime)s %(message)s', level=logging.INFO, force=True)
    logger = logging.getLogger()

    logger.info('Evaluating {}'.format(base))
    # 获取训练过程中保存的最佳模型的路径
    model_path = os.path.join(base, 'model_best.pth')

    # Save the final results for computing ensemble results
    # 保存评估结果
    save_path = os.path.join(base, 'results_{}.npy'.format(opt.dataset)) if opt.save_results else None

    if opt.dataset == 'coco':
        # Evaluate COCO 5-fold 1K
        # Evaluate COCO 5K
        # 计算 Image to text (R@1, R@5, R@10)
        # 以及 Text to image (R@1, R@5, R@10)
        evaluation.evalrank(model_path, opt=opt, tokenizer=tokenizer, model=model, split='testall', fold5=True, save_path=save_path)
    else:
        # Evaluate Flickr30K
        evaluation.evalrank(model_path, opt=opt, tokenizer=tokenizer, model=model, split='test', fold5=False, save_path=save_path)

    logger.info('Evaluation finish')    

# 训练方法
def train(opt, train_loader, model, epoch):
   
    logger = logging.getLogger(__name__)
    # 批处理时间
    batch_time = AverageMeter()
    # 数据加载时间
    data_time = AverageMeter()
    # 用于记录训练日志
    train_logger = LogCollector()

    if epoch == 0:
        # 训练开始记录模型中可训练参数的数量
        logger.info('image encoder trainable parameters: {}M'.format(count_params(model.img_enc)))
        logger.info('txt encoder trainable parameters: {}M'.format(count_params(model.txt_enc)))
        logger.info('criterion trainable parameters: {}M'.format(count_params(model.criterion)))

    # 记录了当前时间，用于计算每个批次数据加载的时间
    end = time.time()
    # 初始化一个空列表 repeat_list，用于记录重复的图像索引（如果有的话）
    repeat_list = []
    # 计算每个epoch中批次的数量
    n_batch = len(train_loader.dataset) // opt.batch_size 

    model.train_start()    

    for i, train_data in enumerate(train_loader):

        # measure data loading time
        # 计算数据加载时间
        data_time.update(time.time() - end)

        # make sure train logger is used
        model.logger = train_logger

        # Update the model
        # images, img_lengths, captions, lengths, ids, img_ids, repeat = train_data
        images, full_imag, img_lengths, captions, clip_captions, lengths, ids, img_ids, repeat = train_data

        # 更新模型参数
        model.train_emb(images, full_imag, captions, clip_captions, lengths, image_lengths=img_lengths, img_ids=img_ids)
  
        # measure elapsed time
        # 记录批处理时间
        batch_time.update(time.time() - end)
        end = time.time()

        if model.Eiters % opt.log_step == 0:               
            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                '{e_log}\t'
                'Batch-Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t'
                    .format(
                    epoch, i+1, n_batch, batch_time=batch_time,
                    data_time=data_time, e_log=str(model.logger)))

        # Record logs in tensorboard
        # 记录到TensorBoard 中，以便于训练过程的可视化和监控
        tb_logger.log_value('epoch', epoch, step=model.Eiters)
        tb_logger.log_value('step', i, step=model.Eiters)
        tb_logger.log_value('batch_time', batch_time.val, step=model.Eiters)
        tb_logger.log_value('data_time', data_time.val, step=model.Eiters)
        model.logger.tb_log(tb_logger, step=model.Eiters)

    return repeat_list

# 测试方法
def validate(opt, val_loader, model):

    logger = logging.getLogger(__name__)
    model.val_start()
    # 关闭梯度计算
    with torch.no_grad():

        # compute the encoding for all the validation images and captions
        # 计算测试集中所有图像和标题的编码表示
        img_embs, cap_embs = encode_data(model, val_loader, opt.log_step, logging.info)

    # have repetitive image features
    # 每个图像有五个相应的标题，只保留其中一个
    img_embs = img_embs[::5]
    # 计算保留后的图像数目
    npts = img_embs.shape[0]
    # 图像特征与文本特征之间的相似度矩阵
    sims = compute_sim(img_embs, cap_embs)
    # 计算R@1、R@5、R@10，并记录
    (r1, r5, r10, medr, meanr) = i2t(npts, sims)
    logging.info("Image to text (R@1, R@5, R@10): %.1f, %.1f, %.1f" % (r1, r5, r10))

    (r1i, r5i, r10i, medri, meanr) = t2i(npts, sims)
    logging.info("Text to image (R@1, R@5, R@10): %.1f, %.1f, %.1f" % (r1i, r5i, r10i))

    # sum of recalls to be used for early stopping
    # 计算 rsum 并记录
    currscore = r1 + r5 + r10 + r1i + r5i + r10i
    logger.info('Current rsum is {}'.format(round(currscore, 1)))

    # record metrics in tensorboard
    # 记录到 TensorBoard 中
    tb_logger.log_value('r1', r1, step=model.Eiters)
    tb_logger.log_value('r5', r5, step=model.Eiters)
    tb_logger.log_value('r10', r10, step=model.Eiters)
    tb_logger.log_value('medr', medr, step=model.Eiters)
    tb_logger.log_value('meanr', meanr, step=model.Eiters)

    tb_logger.log_value('r1i', r1i, step=model.Eiters)
    tb_logger.log_value('r5i', r5i, step=model.Eiters)
    tb_logger.log_value('r10i', r10i, step=model.Eiters)
    tb_logger.log_value('medri', medri, step=model.Eiters)
    tb_logger.log_value('meanr', meanr, step=model.Eiters)
    
    tb_logger.log_value('rsum', currscore, step=model.Eiters)    

    return currscore

# 保存最好模型的检查点
def save_checkpoint(state, is_best, filename='checkpoint.pth', prefix=''):
    logger = logging.getLogger(__name__)
    # 最大尝试次数为2
    tries = 2

    # deal with unstable I/O. Usually not necessary.
    while tries:
        try:
            # don't save checkpoint
            # torch.save(state, prefix + filename)
            if is_best:
                torch.save(state, os.path.join(prefix, 'model_best.pth'))
        except IOError as e:
            error = e
            tries -= 1
        else:
            break
        logger.info('model save {} failed, remaining {} trials'.format(filename, tries))
        if not tries:
            raise error

# 用于调整学习率
def adjust_learning_rate(opt, optimizer, epoch):
    logger = logging.getLogger(__name__)
    # 衰减率默认为 0.1
    decay_rate = opt.decay_rate
    lr_schedules = opt.lr_schedules
    # lr_schedules 默认为[15, 25]，即在第15以及第25个epoch，学习率衰减
    if epoch in lr_schedules:
        logger.info('Current epoch num is {}, decrease all lr by 10'.format(epoch, ))
        for param_group in optimizer.param_groups:
            old_lr = param_group['lr']
            # 学习率降低10倍
            new_lr = old_lr * decay_rate
            param_group['lr'] = new_lr
            logger.info('new lr: {}'.format(new_lr))

# 计算模型的可训练参数数量
def count_params(model):

    # The unit is M (million)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    params = round(params/(1024**2), 2)

    return params


if __name__ == '__main__':
    
    main()
