#!/bin/bash

MMDEPLOY_DIR=/home/qingren/Project/mmdeploy
MMPOSE_DIR=/home/qingren/Project/mmpose

python ${MMDEPLOY_DIR}/tools/torch2onnx.py \
${MMDEPLOY_DIR}/configs/mmpose/pose-detection_ncnn_static-256x256.py \
../hourglass52_coco_256x256.py \
${MMPOSE_DIR}/checkpoints/hourglass52_coco_256x256-4ec713ba_20200709.pth \
${MMPOSE_DIR}/tests/data/coco/000000000785.jpg