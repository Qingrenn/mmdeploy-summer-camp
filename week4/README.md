# Quantitize Hourglass


1. 测试转换后的模型精度时，对于 mmpose 模型，在模型配置文件中 `flip_test` 需设置为 `False`, 参见[benchmark](https://github.com/open-mmlab/mmdeploy/blob/master/docs/zh_cn/03-benchmark/benchmark.md)。
2. mmpose 模型需要额外的输入，但我们无法直接获取它。在导出模型时，可以使用 `$MMDEPLOY_DIR/demo/resources/human-pose.jpg`作为输入, 参见[supported-codebases/mmpose](https://github.com/open-mmlab/mmdeploy/blob/master/docs/zh_cn/04-supported-codebases/mmpose.md)

## [Backend: Pytorch](https://mmpose.readthedocs.io/en/latest/papers/backbones.html#topdown-heatmap-hourglass-on-coco)

<left><img src="hourglass52_coco_256x256/mmpose-hourglass/results/benchmark.png" width="60%"></left>

## Backend: ONNX Runtime

| Arch              | Input Size | AP    | $AP^{50}$ | $AP^{75}$ | AR    | $AR^{50}$ | log |
| :---------------- | :--------- | :---- | :-------- | :-------- | :---- | :-------- | :-- |
| pose_hourglass_52 | 256x256    | 0.717 | 0.894     | 0.790     | 0.774 | 0.933     | [log](hourglass52_coco_256x256/onnx-hourglass/results/hourglass52_coco_256x256_onnx_eval.png) |
| pose_hourglass_52 | 384x384    | x.xxx | x.xxx     | x.xxx     | x.xxx | x.xxx     | [log]() |

## Backend: ncnn

| Arch              | Input Size | AP    | $AP^{50}$ | $AP^{75}$ | AR    | $AR^{50}$ | log |
| :---------------- | :--------- | :---- | :-------- | :-------- | :---- | :-------- | :-- |
| pose_hourglass_52 | 256x256    | x.xxx | x.xxx     | x.xxx     | x.xxx | x.xxx     | [log]() |
| pose_hourglass_52 | 384x384    | 0.737 | 0.897     | 0.799     | 0.789 | 0.938     | [log](hourglass52_coco_384x384/ncnn-hourglass/results_ncnn/test.log) |

## Backend: ncnn int_8

| Arch              | Input Size | AP    | $AP^{50}$ | $AP^{75}$ | AR    | $AR^{50}$ | log |
| :---------------- | :--------- | :---- | :-------- | :-------- | :---- | :-------- | :-- |
| pose_hourglass_52 | 256x256    | 0.713 | 0.892     | 0.786     | 0.771 | 0.932     | [log](hourglass52_coco_256x256/ncnn-hourglass/results/hourglass52_coco_256x256_ncnn_int8_eval.png) |
| pose_hourglass_52 | 384x384    | 0.730 | 0.896     | 0.796     | 0.785 | 0.937     | [log](hourglass52_coco_384x384/ncnn-hourglass/results_ncnn_int8/test.log) |