#ifndef WGB2F3_H_
#define WGB2F3_H_

// API. The implementation is in winograd.c
void winconv_2x3(float *__restrict__ image, const int irows, const int icols,
                 const int C, float *__restrict__ filter, const int K,
                 const int batch, float *__restrict__ out,
                 float *__restrict__ U, float *__restrict__ V,
                 float *__restrict__ M);

#endif