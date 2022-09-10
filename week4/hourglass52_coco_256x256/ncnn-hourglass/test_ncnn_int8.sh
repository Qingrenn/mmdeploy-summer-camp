#!/bin/bash

MMDEPLOY_DIR=/home/qingren/Project/mmdeploy
MMPOSE_DIR=/home/qingren/Project/mmpose

python ${MMDEPLOY_DIR}/tools/test.py \
${MMDEPLOY_DIR}/configs/mmpose/pose-detection_ncnn-int8_static-256x256.py \
../hourglass52_coco_256x256.py \
--model hourglass_ncnn_int8/end2end_int8.param hourglass_ncnn_int8/end2end_int8.bin \
--metrics mAP \
--log2file log.txt \
--device cpu



