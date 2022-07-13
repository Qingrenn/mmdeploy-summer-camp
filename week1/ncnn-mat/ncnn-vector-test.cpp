#include "net.h"
#include <stdio.h>

int main() {
    printf("hello ncnn\n");
    
    // 定义长度为6的向量
    ncnn::Mat m(6);
    printf("total:%ld dims %d w:%d h:%d c:%d elemsize:%ld cstep:%ld\n", \
    m.total(), m.dims, m.w, m.h, m.c, m.elemsize, m.cstep);
    
    // total()返回 cstep * c, 对于vector而言c_step=w, c=1
    for (int i = 0; i < m.total(); i++)
        m[i] = i+1;
    
    // 打印
    const float* r = m.row(0);
    for (int i = 0; i < m.w; i++)
        printf("%.2f ", r[i]);
    printf("\n");

    // 打印
    const float* ptr = (float*) m.data;
    for (int i = 0; i < m.cstep; i++)
        printf("%.2f ", ptr[i]);
    printf("\n");

    return 0;
}