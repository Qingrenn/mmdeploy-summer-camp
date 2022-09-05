#!/bin/bash

WORK_DIR=hourglass_ncnn_int8
MMPOSE_DIR=/home/qingren/Project/mmpose
MMDEPLOY_DIR=/home/qingren/Project/mmdeploy

# onnx2ncnn
export PATH=$PATH:/home/qingren/Project/GitHub/ncnn/build-20220711/install/bin

python ${MMDEPLOY_DIR}/tools/deploy.py \
${MMDEPLOY_DIR}/configs/mmpose/pose-detection_ncnn_static-256x256.py \
../hourglass52_coco_256x256.py \
${MMPOSE_DIR}/checkpoints/hourglass52_coco_256x256-4ec713ba_20200709.pth \
${MMDEPLOY_DIR}/demo/resources/human-pose.jpg \
--work-dir ${WORK_DIR} \
--device cpu \
--quant \
--show