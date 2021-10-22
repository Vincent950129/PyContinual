#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 15:00:00
#SBATCH -o dil_asc_adapter_mask_mixup_1-%j.out
#SBATCH --gres gpu:1


for id in 2
do
    python run.py \
    --note random0,std$id,100,unseen,128ep\
    --ntasks 10 \
    --task celeba \
    --scenario dil_classification \
    --idrandom 0 \
    --approach mlp_hat_merge_ncl \
    --unseen \
    --resume_from_file "./models/fp32/dil_classification/celeba/mlp_tnc128_std$id" \
    --resume_from_aux_file "./models/fp32/dil_classification/celeba/mlp_aux_tnc128_std$id" \
    --eval_only \
    --eval_batch_size 128 \
    --seed $id
done

#ID 2 ---> #BATCH=32
#ID 4 ---> #BATCH=64
#ID 3 ---> #BATCH=64

#--nepochs 1
#    --train_batch_size 128 \
#-m torch.distributed.launch --nproc_per_node=2
#    --distributed \
#    --multi_gpu \
#    --ngpus 2

#    --lr 0.05 \
#    --lr_min 1e-4 \
#    --lr_factor 3 \
#    --lr_patience 5 \
#    --clipgrad 10000

#semantic cap size 1000, 500, 2048

