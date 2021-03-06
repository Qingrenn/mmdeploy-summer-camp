# Week2

## 1. è®ºæéè¯»

- [*Quantization and training of neural networks for efficient integer-arithmetic-only inference*](https://openaccess.thecvf.com/content_cvpr_2018/html/Jacob_Quantization_and_Training_CVPR_2018_paper.html): [ðç¬è®°](paper1.md)

- [*EasyQuant: Post-training Quantization via Scale Optimization*](https://arxiv.org/abs/2006.16669): [ðç¬è®°](paper2.md)

---

## 2. å¯¹ç§°éååéå¯¹ç§°éå

åè[Quantizing deep convolutional networks for efficient inference: A whitepaper](https://arxiv.org/abs/1806.08342)

<left><img src="images/dc_fdc.png" width="60%"></left>

### 2.1 éå¯¹ç§°éå

å¦ä¸å¾å³è¾¹æç¤ºã

<left><img src="images/affine_quantizer.png" width="40%"></left>

å¬å¼1ï¼å¯¹çå®å¼è¿è¡å°ºåº¦ç¼©æ¾å¹¶roundåï¼åå ä¸ä¸ä¸ªint8çåç§»ézï¼å¾å°ä¸ä¸ªä¸çå®å¼å¯¹åºçæ´æ°ã

å¶ä¸­zç§°ä¸ºé¶ç¹ï¼å³éååçzå¼å¯¹åºçå®å¼ç0ã

å¬å¼2ï¼å¯¹å¬å¼1è®¡ç®å¾å°çæ´æ°è¿è¡è£åï¼æç»å¾å°çå®å¼å¯¹åºçéåå¼ã

### 2.2 å¯¹ç§°éå

å¦ä¸å¾å·¦è¾¹æç¤ºã

<left><img src="images/symmetric_quantizer.png" width="40%"></left>

å¯¹ç§°éåä¸éå¯¹ç§°éåçä¸åå¨äºå¶é¶ç¹åºå®ä¸º0ï¼ä¹å°±æ¯è¯´éååç0ä¹å¯¹åºçå®å¼ç0ã


### 2.3 å¯¹ç§°éåVSéå¯¹ç§°éå

åèï¼https://intellabs.github.io/distiller/algo_quantization.html

1.å½ä½¿ç¨éå¯¹ç§°éåæ¶ï¼éåèå´å¾å°äºååçå©ç¨ãè¿æ¯å ä¸ºæä»¬å°æµ®ç¹æ°èå´çæå°/æå¤§å¼ç²¾ç¡®å°æ å°å°éåèå´çæå°/æå¤§å¼ãèä½¿ç¨å¯¹ç§°éåï¼å¦ææµ®ç¹æ°çåå¸ååä¸ä¾§ï¼å¯è½ä¼å¯¼è´éåèå´ä¸­å¤§éçèå´è¢«ç¨äºéåæ¬æ²¡æåºç°è¿çå¼ãè¿æ¹é¢ææç«¯çä¾å­æ¯ReLUä¹åï¼æ´ä¸ªå¼ éé½æ¯æ­£çï¼ä½¿ç¨å¯¹ç§°éåæå³çæä»¬å®éä¸æ²¡æå©ç¨ç¬¦å·ä½ï¼æµªè´¹äº1ä½éåç©ºé´ã

2.å¦ä¸æ¹é¢ï¼å¨å·ç§¯åå¨è¿æ¥å±çå®ç°ä¸ï¼ç±äºæ²¡æé¶ç¹è¿ä¸ä¸ªéååæ°ï¼å¯¹ç§°éåå®éä¸å®ç°è¦ç®åå¾å¤ãå¦æä½¿ç¨éå¯¹ç§°éåï¼é¶ç¹éè¦å¨ç¡¬ä»¶/ç®æ³ä¸å¢å é»è¾ãä¾å¦ï¼å¨å¯¹éååçè¾å¥ç¹å¾å¾è¿è¡zero_paddingæ¶ï¼å¦æä½¿ç¨çæ¯éå¯¹ç§°éåï¼é£ä¹paddingçå¼åºå½æ¯zã

---

## 3. ncnnçéåæ¹æ¡

åèï¼[ç¥ä¹ï¼MegFlowåncnn int8ç®ä»](https://zhuanlan.zhihu.com/p/476605320) å[trtéåppt](https://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf)

ncnnæéç¨çæ¯ä¸ç§åè®­ç»çå¯¹ç§°éåæ¹æ¡ï¼å·ä½å®ç°è§[ncnn2table](https://github.com/Tencent/ncnn/blob/master/tools/quantize/ncnn2table.cpp)ã

### 3.1éåæºå¶

$q = clamp(round(r*S))$

å¶ä¸­$q$æ¯éååçåæ°ï¼ä¸ºint8ç±»åï¼$r$ä¸ºæéææ¿æ´»å¼ççå®å¼ï¼ä¸ºfpç±»åï¼$S$ä¸ºéååæ°-ç¼©æ¾ç³»æ°ï¼æ¯ä¸ªå·ç§¯æ ¸ææ¯å±çæ¿æ´»å¼é½å·æä¸ä¸ªä¸ä¹å¯¹åºçéååæ°ã

### 3.2ç¡®å®éååæ°

ä»¥âklâéåæ¹æ³ä¸ºä¾ï¼ä»ç»éååæ°æ¯å¦ä½ç¡®å®çã

**1.ç¡®å®æ¨¡åæéçéååæ°**

$S = 127 / absmax(W)$

å¶ä¸­$absmax(W)$æ¯å·ç§¯æ ¸æé$W$ä¸­çæå¤§ç»å¯¹å¼ã

**2.ç¡®å®åå±æ¿æ´»å¼ï¼ç¹å¾å¾ï¼çéååæ°**

<left><img src="images/img1.png" width="80%"></left>

ä¸å¾æè¿°äºç¡®å®æ¿æ´»å¼çéååæ°çè¿ç¨ã
1. åå¤ç«æ­£éï¼å°å¶éå¥æ¨¡åï¼ç¨äºè·å¾åå±æ¿æ´»å¼çåå¸ã
2. ç»è®¡æ¯ä¸å±æ¿æ´»å¼ä¸­çæå¤§ç»å¯¹å¼måå¶åå¸ãï¼ç»è®¡æ¿æ´»å¼åå¸æ¶è®¾ç½®äº2048ä¸ªç­é´éçbin, é´éå¤§å°ä¸ºst0=m/2048ï¼
3. è®¾ç½®ä¸ä¸ªéå¼$t \in [128,2048)$ã
    - è®¡ç®è£ååçåå¸`clip_distribution`: è®¾ç½®tä¸ªbinsï¼å°2048ä¸ªbinsä¸­ç´¢å¼å¤§äºtçbinä¸­çåç´ ï¼å¨é¨ç»è®¡è³ç¬¬tä¸ªbinä¸­ãå¦ä¸å¾ä¸­é¨æç¤ºã
    - è®¡ç®éååçåå¸`quantize_distribution`ï¼è®¾ç½®128ä¸ªbinsï¼æ¯ä¸ªbinså¯¹åºçé´ést1=st0*(t/128)ãä¹å°±æ¯å°2048ä¸ªbinsä¸­çåtä¸ªbinsçæ°æ®éæ°ååè³128ä¸ªbinsä¸­ã
    - å¯¹éååçåå¸è¿è¡æ©å¼ `expand_distribution`ï¼å°128ä¸ªbinsçæ°æ®æ©å¼ è³tä¸ªbinsä¸­ã
    - åº¦é`clip_distribution`å`expand_distribution`ä¸¤ä¸ªåå¸ä¹é´çKLæ£åº¦ã
4. éå¤ç¬¬ä¸æ­¥çæä½ï¼æ¾å°ä¸ä¸ªåéçéå¼tï¼ä½¿å¾KLæ£åº¦æå°ï¼å³`clip_distribution`å`expand_distribution`ææ¥è¿ã
5. å¯¹éå¼tåä¸ä¸ªæ å°ï¼å°å¶ä»binçç´¢å¼æ å°è³çå®å¼ãçå®çéå¼ä¸º$t'=(t+0.5)/2048*m$
6. ç¡®å®ç¼©æ¾å°ºåº¦$S=127/t'$

ä¸è¿°è¿ç¨çä¼ªä»£ç å¦ä¸ï¼

<left><img src="images/pseudocode.png" width="80%"></left>

å³äºå¦ä½æ±å`quantize_distribution`å`expand_distribution`ï¼å¯è§å¦ä¸ä¾å­ï¼

<left><img src="images/example.png" width="80%"></left>

ç»è¿æ­¤è¿ç¨ï¼å³å¯çæä¸ä¸ªéååæ°è¡¨ãä¾å¦week1éåsqueezenetä»»å¡ä¸­æçæç[squeezenet_v1.1.table](week1/Quantize-squeezenet/squeezenet_v1.1.table)ã






ä¸è¿°è¿ç¨ç[Pythonå®ç°](https://github.com/BUG1989/caffe-int8-convert-tools/blob/93ec69e465252e2fb15b1fc8edde4a51c9e79dbf/caffe-int8-convert-tool-dev-weight.py#L483)åC++å®ç°æ¯åä¸ä¸ªå¤§ä½¬åçã

### 3.3 å°æ¨¡åè½¬æ¢è³int8

[ncnn2int8](https://github.com/Tencent/ncnn/blob/master/tools/quantize/ncnn2int8.cpp)å®ç°äºä»¥ä¸åè½ï¼
- å°æ¨¡åæééåæint8ã
- æ ¹æ®éååæ°è¡¨ï¼å°éååæ°ç»å®è³åå±ä¸­ã
- å°åéåæä½åééåæä½èåæä¸ä¸ªæä½ãå¦ä¸å¾æç¤ºã

<left><img src="images/img2.png" width="80%"></left>

ç»è¿ä¸è¿°è¿ç¨ï¼ä¼çæéååçæ¨¡åæéãä¾å¦week1éåsqueezenetä»»å¡ä¸­æçæç`squeezenet_v1.1-int8.bin`å`squeezenet_v1.1-int8.param`ã

---

## 4. ncnn-int8 çå·ç§¯å®ç°

å·ä½å®ç°è§[æºç ](https://github.com/Tencent/ncnn/blob/master/src/layer/convolution.cpp)ã

å¯¹äºä¸ä¸ªå·ç§¯å±æ¥è¯´ï¼å¶å¯¹åºçéååæ°å¦ä¸æç¤ºï¼

```C++
// Convolution class
#if NCNN_INT8
    Mat weight_data_int8_scales;
    Mat bottom_blob_int8_scales;
    Mat top_blob_int8_scales;
#endif
```

å¶ä¸­`weight_data_int8_scales`æ¯æ¨¡åæéå¯¹åºçç¼©æ¾å å­ã

å¶ä¸­`bottom_blob_int8_scales`æ¯å½åå·ç§¯å±è¾å¥ç¹å¾å¾å¯¹åºçç¼©æ¾å å­ã

è`top_blob_int8_scales`åå½åå·ç§¯å±çä¸ä¸ä¸ªå·ç§¯ï¼å¨è¿æ¥ï¼å±è¾å¥ç¹å¾å¾å¯¹åºçç¼©æ¾å å­ã

**forward_int8çæµç¨**:

1. åºäºinput_scaleï¼å¯¹è¾å¥çç¹å¾å¾è¿è¡éåã
2. å¯¹äºä¸ä¸ªå·ç§¯æ ¸ä½ä¸æ¬¡å·ç§¯è¾åºççint32ç±»åçç»æsumï¼è¿è¡åéåï¼å³ sumfp32 = sum / (weight_scale * input_scale)ã
3. å¨sumfp32çåºç¡ä¸å ä¸biasï¼éè¿æ¿æ´»å½æ°ã
4. å¯¹è¾åºçæ¿æ´»å¼ééåï¼out = sumfp32 * scale_outãè¿ä¸æ­¥åºäºæ¨¡åçç»æï¼ä¹æå¯è½ä¸åout=sumfp32ãå¦3.3å¾æç¤ºã
5. è·å¾è¾åºç¹å¾å¾ä¸­çä¸ä¸ªåç´ çå¼outãéå¤ä¸è¿°ç2ï¼3ï¼4æ­¥ç´å°å®ææ´ä¸ªå·ç§¯æä½ã

å¶ä¸­ï¼


input_scale=`bottom_blob_int8_scales`ä¸­ä¸è¯¥å±æ¿æ´»å¼å¯¹åºçå¼

weight_scale=`weight_data_int8_scales`ä¸­ä¸è¯¥å±å·ç§¯ä¸­çæä¸ä¸ªå·ç§¯æ ¸å¯¹åºçå¼

scale_out=`top_blob_int8_scales`ä¸­ä¸è¯¥å±æ¿æ´»å¼å¯¹åºçå¼
