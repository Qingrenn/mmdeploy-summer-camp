#!/bin/bash

WORK_DIR=hourglass_torch
MMPOSE_DIR=/home/qingren/Project/mmpose

python ${MMPOSE_DIR}/tools/test.py \
../hourglass52_coco_256x256.py \
${MMPOSE_DIR}/checkpoints/hourglass52_coco_256x256-4ec713ba_20200709.pth \
--out hourglass_torch_results.json \
--work-dir ${WORK_DIR} \
--eval mAP