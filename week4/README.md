# Quantize and Evaluate Hourglass

## 1. 论文阅读

[*Stacked hourglass networks for human pose estimation*](https://link.springer.com/chapter/10.1007/978-3-319-46484-8_29): [📝笔记](hourglass.md)

## 2. 部署流程

### 2.1 了解Hourglass在mmpose中的实现

根据配置文件: [config-hourglass52_coco_256x256.py](https://github.com/open-mmlab/mmpose/blob/master/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hourglass52_coco_256x256.py)，可以知道整个姿态检测网络使用`HourglassNet`作为backbone，而使用了`TopdownHeatmapMultiStageHead`作为keypoint_head（关键点的预测头）。


在[HourglassNet](https://github.com/open-mmlab/mmpose/blob/master/mmpose/models/backbones/hourglass.py)中，只涉及到了torch中的`Conv2d`和`Upsample`两种算子。其前向传播，输入为256\*256\*3的图像，输出为多个64\*64\*256的heatmap组成的list。输出的list是包含了网络中各个hourglass模块提取的特征。在配置文件中，HourglassNet仅仅只使用了一个hourglass模块，因此输出的list中仅包含一个heatmap。

在[TopdownHeatmapMultiStageHead](https://github.com/open-mmlab/mmpose/blob/master/mmpose/models/heads/topdown_heatmap_multi_stage_head.py)中，涉及到了`ConvTranspose2d`和`Conv2d`两种算子。其前向传播，输入为一个包含多个heatmap的list，然后通过多个预测头分别将其映射成64\*64\*17的预测图，并输出一个包含多个预测图的list。在配置文件中，TopdownHeatmapMultiStageHead仅仅只有一个stage，也就是只包含一个预测头；同时，反卷积数量也设置为0，因此该部分也仅仅只使用了`Conv2d`算子。

值得注意的是，对于keypoint_head，训练时调用`forward`的方法，而推理时调用其`inference_model`方法。可见[TopDown pose dtector](https://github.com/open-mmlab/mmpose/blob/master/mmpose/models/detectors/top_down.py)，其中抽象了训练和测试前向传播的流程，由backbone->neck->keypoint_head。

对于TopdownHeatmapMultiStageHead的`inference_model`方法，其在`forward`方法返回的list的基础上，取该list中的最后一个预测图作为输出。但是其在mmppse中的实现还涉及了flip_pairs操作，和将输出结果转至nd.array。上述这些在部署时是不需要的，因此在mmdeploy中对[inference_model重写](https://github.com/open-mmlab/mmdeploy/blob/master/mmdeploy/codebase/mmpose/models/heads/topdown_heatmap_multi_stage_head.py)，删除了这些操作。

通过上述分析，可以看出在模型推理过程中仅仅涉及了如下三个torch算子，而它们在ONNX中均有对应的算子。

| torch    | onnx11 |
| :------- | :----- |
| Conv2d   | [Conv](https://github.com/onnx/onnx/blob/main/docs/Operators.md#conv)   |
| upsample_bilinear2d | [Resize](https://github.com/onnx/onnx/blob/main/docs/Changelog.md#resize-11) |
| ConvTranspose2d | [ConvTranspose](https://github.com/onnx/onnx/blob/main/docs/Changelog.md#convtranspose-11) |

### 2.2 新增deploy config

deploy config由三部分组成，分别是codebase_config，onnx_config，和backend_config。参见[write_config.md](https://github.com/open-mmlab/mmdeploy/blob/master/docs/zh_cn/02-how-to-run/write_config.md)。

codebase_config：其中mmdeploy已经支持了mmpose库的PoseDetection任务，可见[codebase](https://github.com/open-mmlab/mmdeploy/blob/master/mmdeploy/codebase/mmpose/deploy/pose_detection.py)。

```Python
codebase_config = dict(type='mmpose', task='PoseDetection')
```

onnx_config：hourglass对输入图像的尺寸有要求。
```Python
onnx_config = dict(input_shape=[256, 256])
```

backend_config：使用ncnn或者ncnn-int8。
```Python
backend_config = backend_config = dict(type='ncnn', precision='FP32', use_vulkan=False)
```

[新增configs](configs)

### 2.3 部署并测试

- 启动[deploy.sh](hourglass52_coco_256x256/ncnn-hourglass/deploy.sh)。 onnx, ncnn, ncnn-int8的模型会保存至work_dir下。

- 启动[test_ncnn.sh](hourglass52_coco_256x256/ncnn-hourglass/test_ncnn.sh)/[test_ncnn_int8.sh](hourglass52_coco_256x256/ncnn-hourglass/test_ncnn_int8.sh)。在ncnn/ncnn_int8下评估模型精度。

🔧 测试结果：[result.md](result.md)。