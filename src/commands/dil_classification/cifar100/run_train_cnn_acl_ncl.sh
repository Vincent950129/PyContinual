#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 15:00:00
#SBATCH -o til_asc_kim_derpp_4-%j.out
#SBATCH --gres gpu:1


for id in  0
do
    python run.py \
    --note random$id \
    --ntasks 10 \
    --task cifar100 \
    --scenario dil_classification \
    --idrandom $id \
    --approach cnn_acl_ncl \
    --eval_batch_size 128 \
    --train_batch_size 128 \
    --model_path "./models/fp32/dil_classification/cifar100/cnn_acl_$id" \
    --last_id \
    --save_model
done
