#!/bin/bash

MMDEPLOY_DIR=/home/qingren/Project/mmdeploy
MMPOSE_DIR=/home/qingren/Project/mmpose

python ${MMDEPLOY_DIR}/tools/test.py \
${MMDEPLOY_DIR}/configs/mmpose/pose-detection_tensorrt_static-256x256.py \
../hourglass52_coco_256x256.py \
--model hourglass_trt_fp32/end2end.engine \
--metrics mAP \
--log2file log.txt \
--device cuda



