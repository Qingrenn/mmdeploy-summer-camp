import torch
from conv2d import conv2d

def eval(img_w, img_h, _in_channels, _out_channels, _kernel_size, _stride, _padding):
    torch_input = torch.rand(1, _in_channels, img_w, img_h)
    
    # torch conv2d
    torch_conv2d = torch.nn.Conv2d(in_channels=_in_channels, out_channels=_out_channels, 
                                    kernel_size=_kernel_size, stride=_stride, 
                                    padding=_padding, bias=False)
    torch_output = torch_conv2d(torch_input)
    torch_output = torch_output.detach().numpy()[0] # 去掉batch纬度

    # handmade conv2d
    np_input = torch_input.numpy()[0] # 去掉batch纬度
    weight = torch_conv2d.weight.detach().numpy()
    np_output = conv2d(np_input, weight, stride=_stride, padding=_padding)

    return ((np_output - torch_output) < 1e-6).any()

img_w=64
img_h=64
in_channels=3
out_channels=32
kernel_sizes=[3,5,7]
strides=[1,2,4]
paddings=[0,1,2,3]

str_prefix = "%s img_w:%d img_h:%d in_channels:%d out_channels:%d kernel_size:%d stride:%d padding:%d"
for kernel_size in kernel_sizes:
    for stride in strides:
        for padding in paddings:
            res = eval(img_w, img_h, in_channels, out_channels, kernel_size, stride, padding)
            str_postfix = (res, img_w, img_h, in_channels, out_channels, kernel_size, stride, padding)
            print(str_prefix % str_postfix)
            

