# Quantitize Hourglass

## Topdown Heatmap + Hourglass on Coco

[config](https://github.com/open-mmlab/mmpose/blob/master/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hourglass52_coco_256x256.py)

[Results](https://mmpose.readthedocs.io/en/latest/papers/backbones.html#topdown-heatmap-hourglass-on-coco)

<left><img src="hourglass52_coco_256x256/mmpose-hourglass/results/benchmark.png" width="60%"></left>

## Backend: ONNX Runtime

| Arch              | Input Size | AP    | $AP^{50}$ | $AP^{75}$ | AR    | $AR^{50}$ | log |
| :---------------- | :--------- | :---- | :-------- | :-------- | :---- | :-------- | :-- |
| pose_hourglass_52 | 256x256    | 0.717 | 0.894     | 0.790     | 0.774 | 0.933     | [log](hourglass52_coco_256x256/onnx-hourglass/results/hourglass52_coco_256x256_onnx_eval.png) |

## Backend: ncnn int_8

| Arch              | Input Size | AP    | $AP^{50}$ | $AP^{75}$ | AR    | $AR^{50}$ | log |
| :---------------- | :--------- | :---- | :-------- | :-------- | :---- | :-------- | :-- |
| pose_hourglass_52 | 256x256    | 0.713 | 0.892     | 0.786     | 0.771 | 0.932     | [log](hourglass52_coco_256x256/ncnn-hourglass/results/hourglass52_coco_256x256_ncnn_int8_eval.png) |

