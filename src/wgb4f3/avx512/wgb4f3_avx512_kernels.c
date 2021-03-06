#include "wgb4f3/avx512/wgb4f3_avx512_kernels.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "common.h"
#include "wgb4f3/wgb4f3.h"

void winograd_b4f3_srctrans_fp32_avx512(float *tile_buffer, const int64_t ih,
                                        const int64_t iw, const int64_t src_h,
                                        const int64_t src_w,
                                        const int64_t l2_stride,
                                        float *matmul_buffer,
                                        float *src_trans) {
  const int64_t tile_h_stride = TILE_IN_W * KERNEL_ONE_REG;
  const float *tile_src = tile_buffer;
  int64_t tile_src_h_stride = tile_h_stride;

  __m512 zmm12, zmm13, zmm14;
  zmm12 = _mm512_set1_ps(2.0f);
  zmm13 = _mm512_set1_ps(4.0f);
  zmm14 = _mm512_set1_ps(5.0f);
  for (int64_t th = 0; th < TILE_IN_H; ++th) {
    const float *tile_ptr = tile_src + th * tile_src_h_stride;
    float *tmp_buffer = matmul_buffer + th * tile_h_stride;

    __m512 zmm0, zmm2, zmm4, zmm6;
    __m512 zmm8, zmm10, zmm15;

    zmm0 = _mm512_loadu_ps(tile_ptr + 0 * KERNEL_ONE_REG) * zmm13;

    zmm15 = _mm512_loadu_ps(tile_ptr + 1 * KERNEL_ONE_REG);
    zmm4 = zmm15 * zmm13;
    zmm2 = -zmm4;
    zmm8 = zmm15 * zmm12;
    zmm6 = -zmm8;
    zmm10 = zmm4;

    zmm15 = _mm512_loadu_ps(tile_ptr + 2 * KERNEL_ONE_REG);
    zmm0 -= zmm15 * zmm14;
    zmm2 -= zmm15 * zmm13;
    zmm4 -= zmm15 * zmm13;
    zmm6 -= zmm15;
    zmm8 -= zmm15;

    zmm15 = _mm512_loadu_ps(tile_ptr + 3 * KERNEL_ONE_REG);
    zmm2 += zmm15;
    zmm4 -= zmm15;
    zmm6 += zmm15 * zmm12;
    zmm8 -= zmm15 * zmm12;
    zmm10 -= zmm15 * zmm14;

    zmm15 = _mm512_loadu_ps(tile_ptr + 4 * KERNEL_ONE_REG);
    zmm0 += zmm15;
    zmm2 += zmm15;
    zmm4 += zmm15;
    zmm6 += zmm15;
    zmm8 += zmm15;

    zmm15 = _mm512_loadu_ps(tile_ptr + 5 * KERNEL_ONE_REG);
    zmm10 += zmm15;

    _mm512_storeu_ps(tmp_buffer + 0 * KERNEL_ONE_REG, zmm0);
    _mm512_storeu_ps(tmp_buffer + 1 * KERNEL_ONE_REG, zmm2);
    _mm512_storeu_ps(tmp_buffer + 2 * KERNEL_ONE_REG, zmm4);
    _mm512_storeu_ps(tmp_buffer + 3 * KERNEL_ONE_REG, zmm6);
    _mm512_storeu_ps(tmp_buffer + 4 * KERNEL_ONE_REG, zmm8);
    _mm512_storeu_ps(tmp_buffer + 5 * KERNEL_ONE_REG, zmm10);
  }

  for (int64_t tw = 0; tw < TILE_IN_W; ++tw) {
    const float *tmp_buffer = matmul_buffer + tw * KERNEL_ONE_REG;
    float *dst = src_trans + tw * l2_stride;

    __m512 zmm0, zmm2, zmm4, zmm6;
    __m512 zmm8, zmm10, zmm15;

    zmm0 = _mm512_loadu_ps(tmp_buffer + 0 * tile_h_stride) * zmm13;

    zmm15 = _mm512_loadu_ps(tmp_buffer + 1 * tile_h_stride);
    zmm4 = zmm15 * zmm13;
    zmm2 = -zmm4;
    zmm8 = zmm15 * zmm12;
    zmm6 = -zmm8;
    zmm10 = zmm4;

    zmm15 = _mm512_loadu_ps(tmp_buffer + 2 * tile_h_stride);
    zmm0 -= zmm15 * zmm14;
    zmm2 -= zmm15 * zmm13;
    zmm4 -= zmm15 * zmm13;
    zmm6 -= zmm15;
    zmm8 -= zmm15;

    zmm15 = _mm512_loadu_ps(tmp_buffer + 3 * tile_h_stride);
    zmm2 += zmm15;
    zmm4 -= zmm15;
    zmm6 += zmm15 * zmm12;
    zmm8 -= zmm15 * zmm12;
    zmm10 -= zmm15 * zmm14;

    zmm15 = _mm512_loadu_ps(tmp_buffer + 4 * tile_h_stride);
    zmm0 += zmm15;
    zmm2 += zmm15;
    zmm4 += zmm15;
    zmm6 += zmm15;
    zmm8 += zmm15;

    zmm15 = _mm512_loadu_ps(tmp_buffer + 5 * tile_h_stride);
    zmm10 += zmm15;

    _mm512_storeu_ps(dst + 0 * TILE_IN_W * l2_stride, zmm0);
    _mm512_storeu_ps(dst + 1 * TILE_IN_W * l2_stride, zmm2);
    _mm512_storeu_ps(dst + 2 * TILE_IN_W * l2_stride, zmm4);
    _mm512_storeu_ps(dst + 3 * TILE_IN_W * l2_stride, zmm6);
    _mm512_storeu_ps(dst + 4 * TILE_IN_W * l2_stride, zmm8);
    _mm512_storeu_ps(dst + 5 * TILE_IN_W * l2_stride, zmm10);
  }
}

#ifdef USE_INLINE_ASM
IMPLEMENT_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T14O32_ASM(4)
IMPLEMENT_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T14O32_ASM(5)
IMPLEMENT_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T14O32_ASM(6)
IMPLEMENT_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T14O32_ASM(7)
IMPLEMENT_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T14O32_ASM(8)
IMPLEMENT_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T14O32_ASM(9)
IMPLEMENT_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T14O32_ASM(10)
IMPLEMENT_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T14O32_ASM(11)
IMPLEMENT_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T14O32_ASM(12)
IMPLEMENT_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T14O32_ASM(13)
IMPLEMENT_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T14O32_ASM(14)
#endif

IMPLEMENT_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T14O32_FUNC(1)
IMPLEMENT_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T14O32_FUNC(2)
IMPLEMENT_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T14O32_FUNC(3)
IMPLEMENT_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T14O32_FUNC(4)
IMPLEMENT_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T14O32_FUNC(5)
IMPLEMENT_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T14O32_FUNC(6)
IMPLEMENT_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T14O32_FUNC(7)
IMPLEMENT_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T14O32_FUNC(8)
IMPLEMENT_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T14O32_FUNC(9)
IMPLEMENT_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T14O32_FUNC(10)
IMPLEMENT_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T14O32_FUNC(11)
IMPLEMENT_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T14O32_FUNC(12)
IMPLEMENT_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T14O32_FUNC(13)
IMPLEMENT_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T14O32_FUNC(14)

winograd_b4f3_gemm_kernel_fp32_avx512_func_t
    winograd_b4f3_gemm_kernel_fp32_avx512_o32_table[14] = {
        WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T14O32_FUNC(1),
        WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T14O32_FUNC(2),
        WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T14O32_FUNC(3),
        WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T14O32_FUNC(4),
        WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T14O32_FUNC(5),
        WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T14O32_FUNC(6),
        WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T14O32_FUNC(7),
        WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T14O32_FUNC(8),
        WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T14O32_FUNC(9),
        WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T14O32_FUNC(10),
        WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T14O32_FUNC(11),
        WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T14O32_FUNC(12),
        WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T14O32_FUNC(13),
        WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T14O32_FUNC(14),
};

#ifdef USE_INLINE_ASM
IMPLEMENT_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T31O16_ASM(1)
IMPLEMENT_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T31O16_ASM(2)
IMPLEMENT_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T31O16_ASM(3)
IMPLEMENT_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T31O16_ASM(4)
IMPLEMENT_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T31O16_ASM(5)
IMPLEMENT_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T31O16_ASM(6)
IMPLEMENT_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T31O16_ASM(7)
IMPLEMENT_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T31O16_ASM(8)
IMPLEMENT_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T31O16_ASM(9)
IMPLEMENT_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T31O16_ASM(10)
IMPLEMENT_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T31O16_ASM(11)
IMPLEMENT_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T31O16_ASM(12)
IMPLEMENT_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T31O16_ASM(13)
IMPLEMENT_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T31O16_ASM(14)
#endif

IMPLEMENT_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T31O16_FUNC(1)
IMPLEMENT_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T31O16_FUNC(2)
IMPLEMENT_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T31O16_FUNC(3)
IMPLEMENT_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T31O16_FUNC(4)
IMPLEMENT_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T31O16_FUNC(5)
IMPLEMENT_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T31O16_FUNC(6)
IMPLEMENT_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T31O16_FUNC(7)
IMPLEMENT_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T31O16_FUNC(8)
IMPLEMENT_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T31O16_FUNC(9)
IMPLEMENT_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T31O16_FUNC(10)
IMPLEMENT_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T31O16_FUNC(11)
IMPLEMENT_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T31O16_FUNC(12)
IMPLEMENT_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T31O16_FUNC(13)
IMPLEMENT_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T31O16_FUNC(14)

winograd_b4f3_gemm_kernel_fp32_avx512_func_t
    winograd_b4f3_gemm_kernel_fp32_avx512_o16_table[14] = {
        WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T31O16_FUNC(1),
        WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T31O16_FUNC(2),
        WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T31O16_FUNC(3),
        WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T31O16_FUNC(4),
        WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T31O16_FUNC(5),
        WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T31O16_FUNC(6),
        WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T31O16_FUNC(7),
        WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T31O16_FUNC(8),
        WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T31O16_FUNC(9),
        WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T31O16_FUNC(10),
        WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T31O16_FUNC(11),
        WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T31O16_FUNC(12),
        WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T31O16_FUNC(13),
        WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T31O16_FUNC(14),
};

void winograd_b4f3_gemm_kernel_fp32_avx512(int64_t oc_len, int64_t t_len,
                                           wgb4f3_kernel_params params) {
  if (oc_len == 2 * KERNEL_ONE_REG) {
    winograd_b4f3_gemm_kernel_fp32_avx512_o32_table[t_len - 1](
        (int64_t *)&params);
  } else if (oc_len == KERNEL_ONE_REG) {
    winograd_b4f3_gemm_kernel_fp32_avx512_o16_table[t_len - 1](
        (int64_t *)&params);
  } else {
    fprintf(stderr,
            "[ERROR %s] oc_len does not match the kernel shape. (Needs to be "
            "%d or %d, "
            "but get %d\n)",
            __FUNCTION__, KERNEL_ONE_REG, 2 * KERNEL_ONE_REG, oc_len);
    exit(-1);
  }
}

void winograd_b4f3_dsttrans_fp32_avx512(const float *dst_trans,
                                        const int64_t dst_trans_ti_stride,
                                        const int64_t dst_h_stride,
                                        float *matmul_buffer, float *dst) {
  const int64_t matmul_h_stride = TILE_OUT_W * KERNEL_ONE_REG;

  __m512 zmm13, zmm14, zmm15;
  zmm13 = _mm512_set1_ps(2.0f);
  zmm14 = _mm512_set1_ps(4.0f);
  zmm15 = _mm512_set1_ps(8.0f);

  for (int64_t th = 0; th < TILE_IN_H; ++th) {
    const float *dst_trans_ptr =
        dst_trans + th * TILE_IN_W * dst_trans_ti_stride;
    float *tmp = matmul_buffer + th * matmul_h_stride;
    __m512 zmm0, zmm2, zmm4, zmm6;
    __m512 zmm8, zmm10;

    zmm8 = _mm512_loadu_ps(dst_trans_ptr + 0 * dst_trans_ti_stride);
    zmm10 = _mm512_loadu_ps(dst_trans_ptr + 1 * dst_trans_ti_stride);
    zmm0 = zmm8 + zmm10;
    zmm2 = zmm10;
    zmm4 = zmm10;
    zmm6 = zmm10;

    zmm8 = _mm512_loadu_ps(dst_trans_ptr + 2 * dst_trans_ti_stride);
    zmm0 += zmm8;
    zmm2 -= zmm8;
    zmm4 += zmm8;
    zmm6 -= zmm8;

    zmm8 = _mm512_loadu_ps(dst_trans_ptr + 3 * dst_trans_ti_stride);
    zmm0 += zmm8;
    zmm2 += zmm8 * zmm13;
    zmm4 += zmm8 * zmm14;
    zmm6 += zmm8 * zmm15;

    zmm8 = _mm512_loadu_ps(dst_trans_ptr + 4 * dst_trans_ti_stride);
    zmm0 += zmm8;
    zmm2 -= zmm8 * zmm13;
    zmm4 += zmm8 * zmm14;
    zmm6 -= zmm8 * zmm15;

    zmm8 = _mm512_loadu_ps(dst_trans_ptr + 5 * dst_trans_ti_stride);
    zmm6 += zmm8;

    _mm512_storeu_ps(tmp + 0 * KERNEL_ONE_REG, zmm0);
    _mm512_storeu_ps(tmp + 1 * KERNEL_ONE_REG, zmm2);
    _mm512_storeu_ps(tmp + 2 * KERNEL_ONE_REG, zmm4);
    _mm512_storeu_ps(tmp + 3 * KERNEL_ONE_REG, zmm6);
  }

  for (int64_t tw = 0; tw < TILE_OUT_W; ++tw) {
    float *dst_ptr = dst + tw * KERNEL_ONE_REG;
    float *tmp = matmul_buffer + tw * KERNEL_ONE_REG;

    __m512 zmm0, zmm2, zmm4, zmm6;
    __m512 zmm8, zmm10;

    zmm8 = _mm512_loadu_ps(tmp + 0 * matmul_h_stride);
    zmm10 = _mm512_loadu_ps(tmp + 1 * matmul_h_stride);
    zmm0 = zmm8 + zmm10;
    zmm2 = zmm10;
    zmm4 = zmm10;
    zmm6 = zmm10;

    zmm8 = _mm512_loadu_ps(tmp + 2 * matmul_h_stride);
    zmm0 += zmm8;
    zmm2 -= zmm8;
    zmm4 += zmm8;
    zmm6 -= zmm8;

    zmm8 = _mm512_loadu_ps(tmp + 3 * matmul_h_stride);
    zmm0 += zmm8;
    zmm2 += zmm8 * zmm13;
    zmm4 += zmm8 * zmm14;
    zmm6 += zmm8 * zmm15;

    zmm8 = _mm512_loadu_ps(tmp + 4 * matmul_h_stride);
    zmm0 += zmm8;
    zmm2 -= zmm8 * zmm13;
    zmm4 += zmm8 * zmm14;
    zmm6 -= zmm8 * zmm15;

    zmm8 = _mm512_loadu_ps(tmp + 5 * matmul_h_stride);
    zmm6 += zmm8;

    _mm512_storeu_ps(dst_ptr + 0 * dst_h_stride, zmm0);
    _mm512_storeu_ps(dst_ptr + 1 * dst_h_stride, zmm2);
    _mm512_storeu_ps(dst_ptr + 2 * dst_h_stride, zmm4);
    _mm512_storeu_ps(dst_ptr + 3 * dst_h_stride, zmm6);
  }
}