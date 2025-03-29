import os
import fire
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO

from eccv_caption import Metrics


def eval_eccv_caption(
        coco_ann_path=r'C:/datasets/data/coco/captions_val2014.json',
        sim_matrix_path='runs/coco_test_1/results_coco.npy',
):
    # Prepare metric
    metric = Metrics()

    # Prepare the inputs
    coco = COCO(coco_ann_path)
    test_cids = metric.coco_ids

    # Load pre-computed similarity matrix
    sim_data = np.load(sim_matrix_path, allow_pickle=True).item()
    sims = sim_data['sims']

    all_iids, all_cids = [], []
    seen_iids = set()

    for cid in tqdm(test_cids):
        iid = int(coco.anns[cid]['image_id'])
        if iid not in seen_iids:
            all_iids.append(iid)
            seen_iids.add(iid)
        all_cids.append(int(cid))

    i2t = {}
    t2i = {}

    all_cids = np.array(all_cids)
    all_iids = np.array(all_iids)

    K = 50
    for idx, iid in enumerate(all_iids):
        indices = np.argsort(sims[idx, :])[::-1][:K]
        i2t[iid] = [int(cid) for cid in all_cids[indices]]

    for idx, cid in enumerate(all_cids):
        indices = np.argsort(sims[:, idx])[::-1][:K]
        t2i[cid] = [int(iid) for iid in all_iids[indices]]

    scores = metric.compute_all_metrics(
        i2t, t2i,
        target_metrics=('eccv_r1', 'eccv_map_at_r', 'eccv_rprecision',
                        'coco_1k_recalls', 'coco_5k_recalls', 'cxc_recalls'),
        Ks=(1, 5, 10),
        verbose=False
    )
    print(scores)
    # 提取你关注的指标
    eccv_map_at_r_i2t = scores['eccv_map_at_r']['i2t'] * 100
    eccv_rprecision_i2t = scores['eccv_rprecision']['i2t'] * 100
    eccv_r1_i2t = scores['eccv_r1']['i2t'] * 100

    eccv_map_at_r_t2i = scores['eccv_map_at_r']['t2i'] * 100
    eccv_rprecision_t2i = scores['eccv_rprecision']['t2i'] * 100
    eccv_r1_t2i = scores['eccv_r1']['t2i'] * 100
    eccv_caption_sum = eccv_map_at_r_i2t + eccv_rprecision_i2t + eccv_r1_i2t + \
                       eccv_map_at_r_t2i + eccv_rprecision_t2i + eccv_r1_t2i
    # 构建输出字符串
    output_str = (
        f"Image-to-Text: mAP@R {eccv_map_at_r_i2t:.1f} R-P {eccv_rprecision_i2t:.1f} R@1 {eccv_r1_i2t:.1f}\n"
        f"Text-to-Image: mAP@R {eccv_map_at_r_t2i:.1f} R-P {eccv_rprecision_t2i:.1f} R@1 {eccv_r1_t2i:.1f}\n"
        f"ECCV_caption_sum: mAP@R {eccv_caption_sum :.1f}\n"
    )

    # 打印输出
    print(output_str)

# Image-to-Text: mAP@R 34.2 R-P 44.8 R@1 81.3
# Text-to-Image: mAP@R 49.7 R-P 57.4 R@1 90.3
# ECCV_caption_sum: mAP@R 357.7

if __name__ == '__main__':
    # fire.Fire(run)
    eval_eccv_caption(sim_matrix_path='runs/coco_test_1/results_coco.npy')
