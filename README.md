# 🐱 ncnn int8 模型量化评估

[ncnn](https://github.com/Tencent/ncnn)

[mmdeploy](https://github.com/open-mmlab/mmdeploy) 

---

## Schedule

[Week1](week1)：
- 编译ncnn
- 学习naive conv的实现

[Week2](week2)：
- 阅读量化论文
- 学习ncnn-int8的量化方案
- 学习ncnn-int8的conv实现

[Week3](week3)：
- 利用mmdeploy将resnet18转换并量化至ncnn-int8
- 理解tools/deploy.py的工作流程

[Week4](week4):
- 评估 mmpose-Hourglass (onnx, ncnn, ncnn-int8, trt)