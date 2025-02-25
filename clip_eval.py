import os
import torch
# 用于解析命令行参数
import argparse
import logging
from lib import evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='C:/datasets/data', type=str,
                        help='path to datasets')
    parser.add_argument('--dataset', default='coco', type=str, help='the dataset choice, coco or f30k')
    parser.add_argument('--save_results', type=int, default=0, help='if save the similarity matrix for ensemble')
    parser.add_argument('--gpu-id', type=int, default=0, help='the gpu-id for evaluation')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--workers', default=0, type=int,
                        help='Number of data loader workers.')
    opt = parser.parse_args()

    torch.cuda.set_device(opt.gpu_id)
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logger.info('Evaluating {}...')
    # 加载最佳检查点

    # Save the final similarity matrix
    # 保存图文相似度矩阵

    if opt.dataset == 'coco':
        # Evaluate COCO 5-fold 1K
        # Evaluate COCO 5K
        # 计算 Image to text (R@1, R@5, R@10)
        # 以及 Text to image (R@1, R@5, R@10)
        evaluation.evalrank_clip(opt, split='testall', fold5=True, data_path=opt.data_path)
    else:
        # Evaluate Flickr30K
        evaluation.evalrank_clip(opt, split='test', fold5=False, data_path=opt.data_path)


if __name__ == '__main__':
    main()
