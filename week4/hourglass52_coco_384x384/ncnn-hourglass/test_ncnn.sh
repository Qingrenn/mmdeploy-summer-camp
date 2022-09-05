#!/bin/bash

MMDEPLOY_DIR=/home/qingren/Project/mmdeploy
MMPOSE_DIR=/home/qingren/Project/mmpose

python ${MMDEPLOY_DIR}/tools/test.py \
${MMDEPLOY_DIR}/configs/mmpose/pose-detection_ncnn_static-384x384.py \
../hourglass52_coco_384x384.py \
--model hourglass_ncnn_int8/end2end.param hourglass_ncnn_int8/end2end.bin \
--metrics mAP \
--log2file test.log \
--device cpu