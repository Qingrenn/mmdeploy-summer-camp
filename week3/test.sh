#!/bin/bash

MMDEPLOY_DIR=/home/qingren/Project/mmdeploy
MMCLS_DIR=/home/qingren/Project/mmclassification


python ${MMDEPLOY_DIR}/tools/test.py \
${MMDEPLOY_DIR}/configs/mmcls/classification_ncnn-int8_static.py \
${MMCLS_DIR}/configs/resnet/resnet18_8xb16_cifar10.py \
--model resnet18_ncnn_int8/end2end.param resnet18_ncnn_int8/end2end.bin \
--out out.pkl \
--metrics accuracy \
--speed-test \
--device cpu



