python train.py \
--batch_size 128 \
--data_path C:/datasets/data \
--dataset f30k \
--logger_name runs/f30k_test_5 \
--mask_weight 1.0 \
--warmup 8000 \
--num_epochs 40 \
--nmb_prototypes 384 \
--epsilon 0.1 \
--temperature 0.1 \
--moco_M 2048 \
--workers 0
python eval.py --dataset f30k --data_path C:/datasets/data
