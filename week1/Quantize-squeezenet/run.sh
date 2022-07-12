# Optimize model
${NCNN_DIR}/build-20220711/tools/ncnnoptimize \
fp32/squeezenet_v1.1.param \
fp32/squeezenet_v1.1.bin \
opt/squeezenet_v1.1-opt.param \
opt/squeezenet_v1.1-opt.bin \
0

# Create calibration table file
${NCNN_DIR}/build-20220711/tools/quantize/ncnn2table \
opt/squeezenet_v1.1-opt.param \
opt/squeezenet_v1.1-opt.bin \
imagelist.txt \
squeezenet_v1.1.table \
mean=[104,117,123] \
norm=[1,1,1] \
shape=[227,227,3] \
pixel=BGR \
thread=1 \
method=kl

# Quantize model
${NCNN_DIR}/build-20220711/tools/quantize/ncnn2int8 \
opt/squeezenet_v1.1-opt.param \
opt/squeezenet_v1.1-opt.bin \
squeezenet_v1.1-int8.param \
squeezenet_v1.1-int8.bin \
squeezenet_v1.1.table
