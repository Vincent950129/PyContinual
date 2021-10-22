#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 5:00:00
#SBATCH -o til_asc_kim_derpp_4-%j.out
#SBATCH --gres gpu:1



for id in 0 1 2 3 4
do
    python run.py \
    --bert_model 'bert-base-uncased' \
    --note random$id,online \
    --ntasks 10 \
    --task cifar10 \
    --scenario til_classification \
    --idrandom $id \
    --approach cnn_ucl_ncl \
    --image_size 32 \
    --image_channel 3 \
    --ratio 0.125 \
    --beta 0.002 \
    --lr_rho 0.01 \
    --alpha 5 \
    --optimizer SGD \
    --clipgrad 100 \
    --lr_min 2e-6 \
    --lr_factor 3 \
    --lr_patience 5\
    --nepochs=1000 \
    --ent_id \
    --model_path "./models/fp32/til_classification/cifar10/cnn_ucl_$id" \
    --resume_model \
    --eval_only \
    --resume_from_file "./models/fp32/til_classification/cifar10/cnn_ucl_$id" \
    --resume_from_task 9 \
    --eval_batch_size 1
done
#--nepochs 1
# lr_min=2e-6, lr_factor=3, lr_patience=5, clipgrad=100,