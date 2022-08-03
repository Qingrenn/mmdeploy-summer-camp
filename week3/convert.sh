#!/bin/bash

WORK_DIR=resnet18_ncnn_int8
MMCLS_DIR=/home/qingren/Project/mmclassification
MMDEPLOY_DIR=/home/qingren/Project/mmdeploy

export PATH=$PATH:/home/qingren/Project/GitHub/ncnn/build-20220711/install/bin

python ${MMDEPLOY_DIR}/tools/deploy.py \
${MMDEPLOY_DIR}/configs/mmcls/classification_ncnn-int8_static.py \
${MMCLS_DIR}/configs/resnet/resnet18_8xb16_cifar10.py \
${MMCLS_DIR}/checkpoints/resnet18_b16x8_cifar10_20210528-bd6371c8.pth \
${MMCLS_DIR}/demo/demo.JPEG \
--work-dir ${WORK_DIR} \
--device cpu \
--quant \
--dump-info

