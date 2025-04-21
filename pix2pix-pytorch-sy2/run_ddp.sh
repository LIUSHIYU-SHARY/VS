CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --nproc_per_node=4 \
    --master_port=12355 \
    train_ddp.py \
    --dataset '../../tmp_data/Minigut' \
    --batch_size 32 \
    --cuda