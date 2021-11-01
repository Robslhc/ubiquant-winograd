#ifndef AVXTOOLS_H_
#define AVXTOOLS_H_

#include <immintrin.h>
#include <stdlib.h>

inline void memset32_avx(float *dst, const float val, const int64_t n32) {
  int64_t __n32 = n32;
  float *__dst = (float *)__dst;
  if (__n32 >= 8) {
    __m256 ymm_val = _mm256_set1_ps(val);
    while (__n32 >= 16) {
      _mm256_storeu_ps(__dst + 0, ymm_val);
      _mm256_storeu_ps(__dst + 8, ymm_val);
      __dst += 16;
      __n32 -= 16;
    }
    if (__n32 & 8) {
      _mm256_storeu_ps(__dst + 0, ymm_val);
      __dst += 8;
      __n32 -= 8;
    }
  }
  if (__n32 & 4) {
    __dst[0] = val;
    __dst[1] = val;
    __dst[2] = val;
    __dst[3] = val;
    __dst += 4;
  }
  if (__n32 & 2) {
    __dst[0] = val;
    __dst[1] = val;
    __dst += 2;
  }
  if (__n32 & 1) {
    __dst[0] = val;
  }
}

inline void memcpy32_avx(float *dst, const float *src, const int64_t n32) {
  int64_t __n32 = n32;
  float *__dst = (float *)dst;
  const float *__src = (const float *)src;
  while (__n32 >= 16) {
    _mm256_storeu_ps(__dst + 0, _mm256_loadu_ps(__src + 0));
    _mm256_storeu_ps(__dst + 8, _mm256_loadu_ps(__src + 8));
    __dst += 16;
    __src += 16;
    __n32 -= 16;
  }
  if (__n32 & 8) {
    _mm256_storeu_ps(__dst + 0, _mm256_loadu_ps(__src + 0));
    __dst += 8;
    __src += 8;
  }
  if (__n32 & 4) {
    _mm_storeu_ps(__dst + 0, _mm_loadu_ps(__src + 0));
    __dst += 4;
    __src += 4;
  }
  if (__n32 & 2) {
    __dst[0] = __src[0];
    __dst[1] = __src[1];
    __dst += 2;
    __src += 2;
  }
  if (__n32 & 1) {
    __dst[0] = __src[0];
  }
}

#endif