import numpy as np

def conv2d(input, weight, padding, stride):
    _, ih, iw = input.shape
    kn, _, kh, kw = weight.shape
    oh = (ih + 2 * padding - kh) // stride + 1
    ow = (iw + 2 * padding - kw) // stride + 1
    oc = kn
    
    _input = np.pad(input, ((0,0),(padding, padding), (padding, padding)), "constant")
    output = np.zeros((oc, oh, ow)).astype(np.float32)

    for p in range(oc):
        for i in range(oh):
            for j in range(ow):
                output[p, i, j] = np.sum(_input[:, i:i+kh, j:j+kw] * weight[p])
    return output