# Learning Semantic Relationship among Instances for Image-Text Matching

![Static Badge](https://img.shields.io/badge/Pytorch-EE4C2C)
![License: MIT](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)

The codes for our paper "Ambiguity-Aware and High-Order Relation Learning for Multi-Grained Image-Text Matching(AAHR)".
We referred to the implementations of [GPO](https://github.com/woodfrog/vse_infty), [HREM](https://github.com/CrossmodalGroup/HREM), and [CHAN](https://github.com/ppanzx/CHAN) to build up our codes. We extend our gratitude to these awesome works.  

**Note**: The complete codebase will be made public upon acceptance of our paper.

## Introduction

![Overview](https://github.com/Image-Text-Matching/AAHR/AAHR/blob/main/Overview.png)

## Performance

Our method achieves state-of-the-art results on standard benchmarks:

![tab1](https://github.com/Image-Text-Matching/AAHR/AAHR/blob/main/tab1.png)

![tab2](https://github.com/Image-Text-Matching/AAHR/AAHR/blob/main/tab2.png)

We  provide the training logs and checkpoint files for two datasets:

- Training logs and checkpoints for [Flickr30K](https://drive.google.com/drive/folders/1w8wYmM_SybWI8gRH3leaCtcu_1kpN_JV?usp=drive_link)
- Training logs and checkpoints for [MSCOCO](https://drive.google.com/drive/folders/1LJEUUaJ7WQFZvZ4NmlOz_p_s9yO1eem3?usp=drive_link)

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
- The specific required environment can be found [here](https://github.com/Image-Text-Matching/AAHR/AAHR/blob/main/requirements.txt)


### Data

All data sets used in the experiment and the necessary external components are organized in the following manner:

```
data
├── coco
│   ├── precomp  # pre-computed BUTD region features for COCO, provided by SCAN
│   │      ├── train_ids.txt
│   │      ├── train_caps.txt
│   │      ├── ......
│   │
│   ├── images   # raw coco images
│        ├── train2014
│        └── val2014
│  
├── f30k
│   ├── precomp  # pre-computed BUTD region features for Flickr30K, provided by SCAN
│   │      ├── train_ids.txt
│   │      ├── train_caps.txt
│   │      ├── ......
│   │
│   ├── flickr30k-images   # raw flickr30k images
│          ├── xxx.jpg
│          └── ...
│   
└── vocab  # vocab files provided by SCAN (only used when the text backbone is BiGRU)

AAHR
├── bert-base-uncased    # the pretrained checkpoint files for BERT-base
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

##  Citation

