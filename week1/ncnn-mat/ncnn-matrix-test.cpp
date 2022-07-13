#include "net.h"
#include <stdio.h>

int main() {
    printf("hello ncnn\n");

    // 定义宽为2，长为3的矩阵
    ncnn::Mat m(2, 3);
    printf("total:%ld dims %d w:%d h:%d c:%d elemsize:%ld cstep:%ld\n", m.total(), m.dims, m.w, m.h, m.c, m.elemsize, m.cstep);
    
    // total() 返回 cstep * c, 对于matrix而言c_step=w*h, c=1
    for (int i = 0; i < m.total(); i++)
        m[i] = i+1;
    
    // 打印
    for (int i = 0; i < m.h; i++) {
        const float* r = m.row(i);
        for (int j = 0; j < m.w; j++) {
            printf("%.2f ", r[j]);
        }
        printf("\n");
    }

    // 打印
    const float* ptr = (float*) m.data;
    for (int i = 0; i < m.cstep; i++) {
        printf("%.2f ", ptr[i]);
    }
    printf("\n");

    return 0;
}