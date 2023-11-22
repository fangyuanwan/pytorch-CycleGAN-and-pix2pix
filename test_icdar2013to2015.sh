#!/bin/bash
if [ "$1" == "all" ]; then
    how_many=100000
else
    how_many=50
fi

model=ICDAR2013to2015_noIdt-each-epoch
epoch=1

CUDA_VISIBLE_DEVICES=0

python test.py --name ${model} \
    --resize_or_crop=None \
    --loadSize=32 --fineSize=32 --which_model_netD n_layers --n_layers_D 3 \
    --model cycle_gan \
    --no_flip --batchSize 100 \
    --dataroot ../../FY/cyclegan_data/ICDAR2013_2015 \
    --which_direction AtoB \
    --phase train \
    --how_many ${how_many} \
    --which_epoch ${epoch} --gpu_ids -1

python test.py --name ${model} \
    --resize_or_crop=None \
    --loadSize=32 --fineSize=32 --which_model_netD n_layers --n_layers_D 3 \
    --model cycle_gan \
    --no_flip --batchSize 100 \
    --dataroot ../../FY/cyclegan_data/ICDAR2013_2015 \
    --which_direction AtoB \
    --phase test \
    --how_many ${how_many} \
    --which_epoch ${epoch} --gpu_ids -1
