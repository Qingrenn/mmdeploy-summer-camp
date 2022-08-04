# Convert and quantize Resnet to ncnn-int8

## 1. Convert and quantize Resnet

[Quantize tutorial](https://github.com/open-mmlab/mmdeploy/blob/master/docs/zh_cn/02-how-to-run/quantize_model.md)

```bash
export PATH=${PATH}:${NCNN_DIR}/build/install/bin # onnx2ncnn path

python ${MMDEPLOY_DIR}/tools/deploy.py \
${MMDEPLOY_DIR}/configs/mmcls/classification_ncnn-int8_static.py \
${MMCLS_DIR}/configs/resnet/resnet18_8xb16_cifar10.py \
${MMCLS_DIR}/checkpoints/resnet18_b16x8_cifar10_20210528-bd6371c8.pth \
${MMCLS_DIR}/demo/demo.JPEG \
--work-dir ${WORK_DIR} \
--device cpu \
--quant \
--dump-info
```

## 2. Profile Resnet

[Profile tutorial](https://github.com/open-mmlab/mmdeploy/blob/master/docs/zh_cn/02-how-to-run/profile_model.md)

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

benchmark：

<left><img src="images/resnet18_ncnn_benchmark.png" width="80%"></left>


