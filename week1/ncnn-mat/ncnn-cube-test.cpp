#include "mat.h"
#include <stdio.h>

int main() {
    printf("hello ncnn\n");
    
    // 定义宽为2，长为3，通道数为4的矩阵
    ncnn::Mat m(2, 3, 4);
    printf("total:%ld dims %d w:%d h:%d c:%d elemsize:%ld cstep:%ld\n", m.total(), m.dims, m.w, m.h, m.c, m.elemsize, m.cstep);

    // total() 返回 cstep * c, 对于cube而言cstep=alignSize((size_t)w * h * elemsize, 16), c=4
    // alignSize(a, 16) 返回大于等于a的16的最小倍数
    for (int i = 0; i < m.total(); i++)
        m[i] = i+1;
    
    // 打印
    for (int i = 0; i < m.c; i++) {
        ncnn::Mat middle = m.channel(i);
        for (int j = 0; j < m.h; j++) {
            const float* r = middle.row(j);
            for (int k = 0; k < m.w; k++) {
                printf("%.2f ", r[k]);
            }
            printf("\n");
        }
        printf("\n");
    }
    
    // 打印
    for (int i = 0; i < m.c; i++) {
        ncnn::Mat middle = m.channel(i);
        float* ptr = (float*) middle;
        for (int j = 0; j < m.cstep; j++){
            printf("%.2f ", ptr[j]);
        }
        printf("\n");
    }

    return 0;
}