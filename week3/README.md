# Convert and quantize Resnet to ncnn-int8

## 1. Convert and quantize Resnet

官方文档：[Convert and quantize tutorial](https://github.com/open-mmlab/mmdeploy/blob/master/docs/zh_cn/02-how-to-run/quantize_model.md)，具体的参数设置可阅读`deploy/tools/deploy.py`。

```bash
export PATH=${PATH}:${NCNN_DIR}/build/install/bin # onnx2ncnn path

python ${MMDEPLOY_DIR}/tools/deploy.py \
${MMDEPLOY_DIR}/configs/mmcls/classification_ncnn-int8_static.py \
${MMCLS_DIR}/configs/resnet/resnet18_8xb16_cifar10.py \
${MMCLS_DIR}/checkpoints/resnet18_b16x8_cifar10_20210528-bd6371c8.pth \
${MMCLS_DIR}/demo/demo.JPEG \
--work-dir ${WORK_DIR} \
--device cpu \
--quant
```

`deploy.py`的主要功能是将OpenMMlab模型转换成各种后端（ONNX, TensorRT, ncnn, PPLNN, OPenVINO）的模型文件，模型量化（fp32->int8）。

以resnet转ncnn-int8为例，deploy的完整工作流程如下：
1. 加载配置文件，即模型配置文件`model_cfg`和部署配置文件`deploy_cfg`。
2. 将torch模型转ir（end2end.onnx）
3. 将IR转至后端ncnn-fp32(end2end.param和end2end.bin)
4. 量化
    - 根据ncnn-fp32模型，利用ppq量化工具，生成量化表（end2end.table）
    - 将ncnn-fp32模型转换成ncnn-int8（end2end_int8.param和end2end_int8.bin）
5. 运行后端模型对测试图像输出推理结果(如果是跑在服务器上，没有图像输出，是会跳过这一步的)


PS：其中获得模型的ir表示后，如果在`deploy_cfg`中有配置`partition_config`，会对模型进行切分，例如[configs/mmdet/detection/two-stage_partition_ncnn_static.py](https://github.com/open-mmlab/mmdeploy/blob/master/configs/mmdet/detection/two-stage_partition_ncnn_static.py)。

## 2. Profile Resnet

官方文档：[Profile tutorial](https://github.com/open-mmlab/mmdeploy/blob/master/docs/zh_cn/02-how-to-run/profile_model.md)

```bash
python ${MMDEPLOY_DIR}/tools/test.py \
${MMDEPLOY_DIR}/configs/mmcls/classification_ncnn-int8_static.py \
${MMCLS_DIR}/configs/resnet/resnet18_8xb16_cifar10.py \
--model resnet18_ncnn_int8/end2end.param resnet18_ncnn_int8/end2end.bin \
--metrics accuracy \
--speed-test \
--device cpu
```

resnet-int8在Cifar10上的测试结果：

<left><img src="images/resnet18_ncnn_int8_cifar10.png" width="80%"></left>

resnet-fp32在Cifar10上的测试结果：
<left><img src="images/resnet18_ncnn_fp32_cifar10.png" width="80%"></left>

<table width=600 border=1>
<tr align=center> <td> Model </td> <td> dataset </td> <td> fp32 top-1 (%) </td> <td> fp32 FPS </td> <td> int8 top-1 (%) </td> <td> int8 FPS </td> </tr>
<tr align=center> <td> Resnet18 </td> <td> Cifar10 </td> <td> 94.82 </td> <td> 134.50 </td> <td> 94.80 </td> <td> 176.30 </td> </tr>
</table>

resnet18在量化后，cifar10上的精度下降0.2个点，但是FPS增加了将近31%。

官方提供的[benchmark](https://github.com/open-mmlab/mmdeploy/blob/master/docs/zh_cn/03-benchmark/quantization.md)：

<left><img src="images/resnet18_ncnn_benchmark.png" width="60%"></left>


