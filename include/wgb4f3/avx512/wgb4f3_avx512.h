#ifndef WGB4F3_AVX512_H_
#define WGB4F3_AVX512_H_

#include "common.h"

// API. The implementation is in src/wgb4f3/avx512/wgb4f3_avx512.c
// helper functions
WinogradOptParams init_winconv_4x3_params(const int N, const int C, const int K,
                                          const int H, const int W,
                                          int precompute);

void winconv_4x3_avx512_cvt_flt(WinogradOptParams param,
                                const float *__restrict__ filter, const int C,
                                const int K, float *__restrict__ filter_cvt);

void winconv_4x3_avx512(WinogradOptParams param, float *__restrict__ image,
                        const int irows, const int icols, const int C,
                        float *__restrict__ filter, const int K,
                        const int batch, float *__restrict__ out,
                        float *__restrict__ tmpbuf);

#endif