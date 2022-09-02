#!/bin/bash

MMPOSE_DIR=/home/qingren/Project/mmpose
CHECKPOINT_FILE=checkpoints/hourglass52_coco_256x256-4ec713ba_20200709.pth

python ${MMPOSE_DIR}/tools/test.py \
../hourglass52_coco_256x256.py \
${MMPOSE_DIR}/${CHECKPOINT_FILE} \
--out hourglass52_coco_256x256_res.json \
--eval mAP 
