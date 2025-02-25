python train.py \
--batch_size 256 \
--data_path C:/datasets/data \
--dataset coco \
--logger_name runs/coco_test_2 \
--mask_weight 1.5 \
--warmup 8000 \
--num_epochs 40 \
--nmb_prototypes 768 \
--epsilon 0.1 \
--temperature 0.1 \
--moco_M 2048 \
--workers 0
python eval.py --dataset coco --data_path C:/datasets/data

