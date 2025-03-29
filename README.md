# Ambiguity-Aware and High-Order Relation Learning for Multi-Grained Image-Text Matching

![Static Badge](https://img.shields.io/badge/Pytorch-EE4C2C)
![License: MIT](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)

The codes for our paper ["Ambiguity-Aware and High-Order Relation Learning for Multi-Grained Image-Text Matching(AAHR)"](https://github.com/Image-Text-Matching/AAHR/blob/main/paper_AAHR.pdf), which is accepted by the Knowledge-Based Systems(KBS), 2025. 
We referred to the implementations of [GPO](https://github.com/woodfrog/vse_infty), [HREM](https://github.com/CrossmodalGroup/HREM), and [eccv-caption](https://github.com/naver-ai/eccv-caption) to build up our codes. We extend our gratitude for these awesome works.  

🚀 **Note**: We have uploaded the complete code.

## Introduction
Image-text matching is crucial for bridging the semantic gap between computer vision and natural language processing. However, existing methods still face challenges in handling high-order associations and semantic ambiguities among similar instances. These ambiguities arise from subtle differences between soft positive samples (semantically similar but incorrectly labeled) and soft negative samples (locally matched but globally inconsistent), creating matching uncertainties.  Furthermore, current methods fail to fully utilize the neighborhood relationships among semantically similar instances within training batches, limiting the model's ability to learn high-order shared knowledge. This paper proposes the Ambiguity-Aware and High-order Relation learning framework (AAHR) to address these issues. AAHR constructs a unified representation space through dynamic clustering prototype contrastive learning, effectively mitigating the soft positive sample problem. The framework introduces global and local feature extraction mechanisms and an adaptive aggregation network, significantly enhancing full-grained semantic understanding capabilities. Additionally, AAHR employs intra-modal and inter-modal correlation matrices to investigate neighborhood relationships among sample instances thoroughly. It incorporates GNN to enhance semantic interactions between instances. Furthermore, AAHR integrates momentum contrastive learning to expand the negative sample set. These combined strategies significantly improve the model's ability to discriminate between features. Experimental results demonstrate that AAHR outperforms existing state-of-the-art methods on Flickr30K, MSCOCO, and ECCV Caption datasets, considerably improving the accuracy and efficiency of image-text matching.
![Overview](https://github.com/Image-Text-Matching/AAHR/blob/main/Overview.png)

## Performance

Our method achieves state-of-the-art results on standard benchmarks:

![tab1](https://github.com/Image-Text-Matching/AAHR/blob/main/tab1.png)

![tab2](https://github.com/Image-Text-Matching/AAHR/blob/main/tab2.png)

We  provide the training logs and checkpoint files for two datasets:

- Training logs and checkpoints for [Flickr30K](https://drive.google.com/drive/folders/1w8wYmM_SybWI8gRH3leaCtcu_1kpN_JV?usp=drive_link)
- Training logs and checkpoints for [MSCOCO](https://drive.google.com/drive/folders/1LJEUUaJ7WQFZvZ4NmlOz_p_s9yO1eem3?usp=drive_link)

##  Citation
If you find our paper and code useful in your research, please consider giving a star ⭐ and a citation 📝:
```
@article{chen2025ambiguity,
  title={Ambiguity-Aware and High-Order Relation Learning for Multi-Grained Image-Text Matching},
  author={Chen, Junyu and Gao, Yihua and Ge, Mingyuan and Li, Mingyong},
  journal={Knowledge-Based Systems},
  pages={113355},
  year={2025},
  publisher={Elsevier}
}
```

## Preparation

### Environments

We recommended the following dependencies.

- Python 3.9
- [PyTorch](http://pytorch.org/) 1.11
- transformers  4.36.0
- open-clip-torch 2.24.0
- numpy 1.23.5
- nltk 3.7
- tensorboard-logger 0.1.0
- eccv-caption 0.1.0
- The specific required environment can be found [here](https://github.com/Image-Text-Matching/AAHR/blob/main/requirements.txt)


### Data

All data sets used in the experiment and the necessary external components are organized in the following manner:

```
data
├── coco
│   ├── precomp  # pre-computed BUTD region features for COCO, provided by SCAN
│   │      ├── train_ids.txt
│   │      ├── train_caps.txt
│   │      ├── ......
│   │── id_mapping.json
│   │── captions_val2014.json
│   ├── images   # raw coco images
│        ├── train2014
│        └── val2014
│  
├── f30k
│   ├── precomp  # pre-computed BUTD region features for Flickr30K, provided by SCAN
│   │      ├── train_ids.txt
│   │      ├── train_caps.txt
│   │      ├── ......
│   │── id_mapping.json
│   ├── flickr30k-images   # raw flickr30k images
│          ├── xxx.jpg
│          └── ...
│   
└── vocab  # vocab files provided by SCAN (only used when the text backbone is BiGRU)

AAHR
├── bert-base-uncased    # the pre-trained checkpoint files for BERT-base
│   ├── config.json
│   ├── tokenizer_config.txt
│   ├── vocab.txt
│   ├── pytorch_model.bin
│   ├── ......

└── CLIP                         # the pretrained checkpoint files for OpenCLIP
│   ├── config.json
│   ├── tokenizer_config.json
│   ├── vocab.json
│   ├── open_clip_config.json
│   ├── open_clip_pytorch_model.bin
│   ├── ......
│  
└── ....

```

#### Data Sources:

- BUTD features: [SCAN (Kaggle)](https://www.kaggle.com/datasets/kuanghueilee/scan-features) or [Baidu Yun](https://pan.baidu.com/s/1Dmnf0q9J29m4-fyL7ubqdg?pwd=AAHR) (code: AAHR)
- MSCOCO images: [Official](https://cocodataset.org/#download) or [Baidu Yun](https://pan.baidu.com/s/1NqcL4FIDs-5Did3O67apFw?pwd=AAHR ) (code: AAHR)
- Flickr30K images: [Official](https://shannon.cs.illinois.edu/DenotationGraph/) or [Baidu Yun](https://pan.baidu.com/s/1vjae2ODiLqWpNbK4AxiQ9w?pwd=AAHR) (code: AAHR)
- Pretrained models: [BERT-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) and [OpenCLIP](https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K) from HuggingFace

## Training

Train MSCOCO and Flickr30K from scratch:

```
bash  run_f30k.sh
```

```
bash  run_coco.sh
```

## Evaluation

Modify the corresponding parameters in eval.py to test the Flickr30K or MSCOCO data set:

```
python eval.py  --dataset f30k  --data_path "path/to/dataset"
```

```
python eval.py  --dataset coco --data_path "path/to/dataset"
```
To test ECCV_caption, please follow these steps:
1. First, execute eval.py to test the MSCOCO pre-trained model
- Set the parameter --save_results=1
- This will generate a results_coco.npy file
2. Then adjust the relevant parameters as needed and run the captioning command
   
The required captions_val2014.json file can be downloaded from [here](https://drive.google.com/file/d/1EPOXg3-an90J_ClqWOXF3UOvXJa1_pbh/view?usp=drive_link).

```
python eccv_caption_test.py
```

