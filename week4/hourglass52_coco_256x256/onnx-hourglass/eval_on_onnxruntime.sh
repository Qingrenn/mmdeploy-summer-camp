#!/bin/bash

MMDEPLOY_DIR=/home/qingren/Project/mmdeploy
MMPOSE_DIR=/home/qingren/Project/mmpose

python ${MMDEPLOY_DIR}/tools/test.py \
${MMDEPLOY_DIR}/configs/mmpose/pose-detection_onnxruntime_static.py \
../hourglass52_coco_256x256.py \
--model hourglass-v1/end2end.onnx \
--metrics mAP \
--device cpu