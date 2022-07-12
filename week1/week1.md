# Week1

根据文档[how to build ncnn](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-linux)， 克隆并编译ncnn。

使用的设备是x86架构，nvidia 3060。

---

1.推理不使用VULKAN
```bash
cd ncnn
mkdir -p build-20220711
cmake -DCMAKE_BUILD_TYPE=Release
-DNCNN_BUILD_EXAMPLES=ON ..
make -j$(nproc)
# 安装： make install
```

2.下载[测试样本](https://github.com/nihui/ncnn-android-squeezenet/blob/master/screenshot.png)至`ncnn/images`，运行squeezenet Demo，验证ncnn推理是否正常:

<left><img src="https://github.com/nihui/ncnn-android-squeezenet/blob/master/screenshot.png" height=20%></left>

```bash
cd ncnn/examples
../build-20220711/examples/squeezenet ../images/screenshot.png
```

<left><img src="images/wo_vulkan.png" width=100%></left>

置信度最高的三个类别索引分别为`281,285,282`。
索引+1对应文件[synset_words.txt](https://github.com/Tencent/ncnn/blob/master/examples/synset_words.txt)中的行号。
可以得知`281: tabby, tabby cat`, `285: Egyptian cat`, `282: tiger cat`。

置信度最高的三个类别都与`cat`有关, 推理是正确的。

---

1.推理时使用VULKAN:
```bash
cd ncnn
mkdir -p build-20220701
cmake -DCMAKE_BUILD_TYPE=Release -DNCNN_VULKAN=ON -DNCNN_BUILD_EXAMPLES=ON ..
make -j$(nproc)
# 安装： make install
```

2.运行squeezenet Demo，验证ncnn推理是否正常:

```bash
cd ncnn/examples
../build-20220701/examples/squeezenet ../images/screenshot.png
```

<left><img src="images/with_vulkan.png" width=100%></left>

置信度最高的三个类别索引分别为`128,143,98`。
索引+1对应文件[synset_words.txt](https://github.com/Tencent/ncnn/blob/master/examples/synset_words.txt)中的行号。
可以得知`128: black stork, Ciconia nigra`, `143: oystercatcher, oyster catcher`, `98: red-breasted merganser, Mergus serrator`。

测试样本对应的类别应该是`cat`, 这里推理出了一些问题。

暂时不使用VULKAN进行推理，这可能存在一些问题。
