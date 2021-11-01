#include "wgb4f3/avx512/wgb4f3_avx512_kernels.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "avxtools.h"
#include "common.h"
#include "wgb4f3/wgb4f3.h"

void winograd_b4f3_srctrans_fp32_avx512(
    const float *base_src, const int64_t ih, const int64_t iw,
    const int64_t src_h, const int64_t src_w, const int64_t l2_stride,
    float *tile_buffer, float *matmul_buffer, float *src_trans) {
  const int64_t tile_h_stride = TILE_IN_W * KERNEL_ONE_REG;
  const float *tile_src;
  int64_t tile_src_h_stride;
  if (ih + TILE_IN_H <= src_h && iw + TILE_IN_W <= src_w) {
    tile_src = base_src + ih * src_w * KERNEL_ONE_REG + iw * KERNEL_ONE_REG;
    tile_src_h_stride = src_w * KERNEL_ONE_REG;
  } else {
    tile_src = tile_buffer;
    tile_src_h_stride = tile_h_stride;
    float *tile_buffer_ptr = tile_buffer;
    WINO_DEBUG("tile_buffer_ptr = %x\n", tile_buffer_ptr);
    for (int64_t h = ih; h < ih + TILE_IN_H; ++h) {
      if (h >= src_h) {
        memset32_avx(tile_buffer_ptr, 0.0f, tile_h_stride);
      } else {
        int64_t tw_start = iw;
        int64_t tw_len = MAX(MIN(src_w, iw + TILE_IN_W) - tw_start, 0);
        int64_t tr_pad = MAX(iw + TILE_IN_W - src_w, 0);
#ifdef __DEBUG
        assert(tw_len + tr_pad == TILE_IN_W);
#endif
        int64_t w = 0;
        memcpy32_avx(tile_buffer_ptr + w * KERNEL_ONE_REG,
                     base_src + (h * src_w + tw_start) * KERNEL_ONE_REG,
                     tw_len * KERNEL_ONE_REG);
        w += tw_len;
        memset32_avx(tile_buffer_ptr + w * KERNEL_ONE_REG, 0.0f,
                     tr_pad * KERNEL_ONE_REG);
      }
      tile_buffer_ptr += tile_h_stride;
    }
  }

  __m512 zmm12, zmm13, zmm14;
  zmm12 = _mm512_set1_ps(2.0f);
  zmm13 = _mm512_set1_ps(4.0f);
  zmm14 = _mm512_set1_ps(5.0f);
  for (int64_t th = 0; th < TILE_IN_H; ++th) {
    const float *tile_ptr = tile_src + th * tile_src_h_stride;
    float *tmp_buffer = matmul_buffer + th * tile_h_stride;

    __m512 zmm0, zmm2, zmm4, zmm6;
    __m512 zmm8, zmm10, zmm15;

    zmm0 = _mm512_load_ps(tile_ptr + 0 * KERNEL_ONE_REG) * zmm13;

    zmm15 = _mm512_load_ps(tile_ptr + 1 * KERNEL_ONE_REG);
    zmm4 = zmm15 * zmm13;
    zmm2 = -zmm4;
    zmm8 = zmm15 * zmm12;
    zmm6 = -zmm8;
    zmm10 = zmm4;

    zmm15 = _mm512_load_ps(tile_ptr + 2 * KERNEL_ONE_REG);
    zmm0 -= zmm15 * zmm14;
    zmm2 -= zmm15 * zmm13;
    zmm4 -= zmm15 * zmm13;
    zmm6 -= zmm15;
    zmm8 -= zmm15;

    zmm15 = _mm512_load_ps(tile_ptr + 3 * KERNEL_ONE_REG);
    zmm2 += zmm15;
    zmm4 -= zmm15;
    zmm6 += zmm15 * zmm12;
    zmm8 -= zmm15 * zmm12;
    zmm10 -= zmm15 * zmm14;

    zmm15 = _mm512_load_ps(tile_ptr + 4 * KERNEL_ONE_REG);
    zmm0 += zmm15;
    zmm2 += zmm15;
    zmm4 += zmm15;
    zmm6 += zmm15;
    zmm8 += zmm15;

    zmm15 = _mm512_load_ps(tile_ptr + 5 * KERNEL_ONE_REG);
    zmm10 += zmm15;

    _mm512_store_ps(tmp_buffer + 0 * KERNEL_ONE_REG, zmm0);
    _mm512_store_ps(tmp_buffer + 1 * KERNEL_ONE_REG, zmm2);
    _mm512_store_ps(tmp_buffer + 2 * KERNEL_ONE_REG, zmm4);
    _mm512_store_ps(tmp_buffer + 3 * KERNEL_ONE_REG, zmm6);
    _mm512_store_ps(tmp_buffer + 4 * KERNEL_ONE_REG, zmm8);
    _mm512_store_ps(tmp_buffer + 5 * KERNEL_ONE_REG, zmm10);
  }

  for (int64_t tw = 0; tw < TILE_IN_W; ++tw) {
    const float *tmp_buffer = matmul_buffer + tw * KERNEL_ONE_REG;
    float *dst = src_trans + tw * l2_stride;

    __m512 zmm0, zmm2, zmm4, zmm6;
    __m512 zmm8, zmm10, zmm15;

    zmm0 = _mm512_load_ps(tmp_buffer + 0 * tile_h_stride) * zmm13;

    zmm15 = _mm512_load_ps(tmp_buffer + 1 * tile_h_stride);
    zmm4 = zmm15 * zmm13;
    zmm2 = -zmm4;
    zmm8 = zmm15 * zmm12;
    zmm6 = -zmm8;
    zmm10 = zmm4;

    zmm15 = _mm512_load_ps(tmp_buffer + 2 * tile_h_stride);
    zmm0 -= zmm15 * zmm14;
    zmm2 -= zmm15 * zmm13;
    zmm4 -= zmm15 * zmm13;
    zmm6 -= zmm15;
    zmm8 -= zmm15;

    zmm15 = _mm512_load_ps(tmp_buffer + 3 * tile_h_stride);
    zmm2 += zmm15;
    zmm4 -= zmm15;
    zmm6 += zmm15 * zmm12;
    zmm8 -= zmm15 * zmm12;
    zmm10 -= zmm15 * zmm14;

    zmm15 = _mm512_load_ps(tmp_buffer + 4 * tile_h_stride);
    zmm0 += zmm15;
    zmm2 += zmm15;
    zmm4 += zmm15;
    zmm6 += zmm15;
    zmm8 += zmm15;

    zmm15 = _mm512_load_ps(tmp_buffer + 5 * tile_h_stride);
    zmm10 += zmm15;

    _mm512_store_ps(dst + 0 * TILE_IN_W * l2_stride, zmm0);
    _mm512_store_ps(dst + 1 * TILE_IN_W * l2_stride, zmm2);
    _mm512_store_ps(dst + 2 * TILE_IN_W * l2_stride, zmm4);
    _mm512_store_ps(dst + 3 * TILE_IN_W * l2_stride, zmm6);
    _mm512_store_ps(dst + 4 * TILE_IN_W * l2_stride, zmm8);
    _mm512_store_ps(dst + 5 * TILE_IN_W * l2_stride, zmm10);
  }
}

void winograd_b4f3_gemm_kernel_fp32_avx512(int64_t oc_len, int64_t t_len,
                                           wgb4f3_kernel_params params) {
  if (oc_len == 2 * KERNEL_ONE_REG) {
    WINOGRAD_B4F3_T14O32_KERNEL_AVX512(params, t_len);
  } else if (oc_len == KERNEL_ONE_REG) {
    WINOGRAD_B4F3_T31O16_KERNEL_AVX512(params, t_len);
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

    zmm8 = _mm512_load_ps(dst_trans_ptr + 0 * dst_trans_ti_stride);
    zmm10 = _mm512_load_ps(dst_trans_ptr + 1 * dst_trans_ti_stride);
    zmm0 = zmm8 + zmm10;
    zmm2 = zmm10;
    zmm4 = zmm10;
    zmm6 = zmm10;

    zmm8 = _mm512_load_ps(dst_trans_ptr + 2 * dst_trans_ti_stride);
    zmm0 += zmm8;
    zmm2 -= zmm8;
    zmm4 += zmm8;
    zmm6 -= zmm8;

    zmm8 = _mm512_load_ps(dst_trans_ptr + 3 * dst_trans_ti_stride);
    zmm0 += zmm8;
    zmm2 += zmm8 * zmm13;
    zmm4 += zmm8 * zmm14;
    zmm6 += zmm8 * zmm15;

    zmm8 = _mm512_load_ps(dst_trans_ptr + 4 * dst_trans_ti_stride);
    zmm0 += zmm8;
    zmm2 -= zmm8 * zmm13;
    zmm4 += zmm8 * zmm14;
    zmm6 -= zmm8 * zmm15;

    zmm8 = _mm512_load_ps(dst_trans_ptr + 5 * dst_trans_ti_stride);
    zmm6 += zmm8;

    _mm512_store_ps(tmp + 0 * KERNEL_ONE_REG, zmm0);
    _mm512_store_ps(tmp + 1 * KERNEL_ONE_REG, zmm2);
    _mm512_store_ps(tmp + 2 * KERNEL_ONE_REG, zmm4);
    _mm512_store_ps(tmp + 3 * KERNEL_ONE_REG, zmm6);
  }

  for (int64_t tw = 0; tw < TILE_OUT_W; ++tw) {
    float *dst_ptr = dst + tw * KERNEL_ONE_REG;
    float *tmp = matmul_buffer + tw * KERNEL_ONE_REG;

    __m512 zmm0, zmm2, zmm4, zmm6;
    __m512 zmm8, zmm10;

    zmm8 = _mm512_load_ps(tmp + 0 * matmul_h_stride);
    zmm10 = _mm512_load_ps(tmp + 1 * matmul_h_stride);
    zmm0 = zmm8 + zmm10;
    zmm2 = zmm10;
    zmm4 = zmm10;
    zmm6 = zmm10;

    zmm8 = _mm512_load_ps(tmp + 2 * matmul_h_stride);
    zmm0 += zmm8;
    zmm2 -= zmm8;
    zmm4 += zmm8;
    zmm6 -= zmm8;

    zmm8 = _mm512_load_ps(tmp + 3 * matmul_h_stride);
    zmm0 += zmm8;
    zmm2 += zmm8 * zmm13;
    zmm4 += zmm8 * zmm14;
    zmm6 += zmm8 * zmm15;

    zmm8 = _mm512_load_ps(tmp + 4 * matmul_h_stride);
    zmm0 += zmm8;
    zmm2 -= zmm8 * zmm13;
    zmm4 += zmm8 * zmm14;
    zmm6 -= zmm8 * zmm15;

    zmm8 = _mm512_load_ps(tmp + 5 * matmul_h_stride);
    zmm6 += zmm8;

    _mm512_store_ps(dst_ptr + 0 * dst_h_stride, zmm0);
    _mm512_store_ps(dst_ptr + 1 * dst_h_stride, zmm2);
    _mm512_store_ps(dst_ptr + 2 * dst_h_stride, zmm4);
    _mm512_store_ps(dst_ptr + 3 * dst_h_stride, zmm6);
  }
}

void winograd_b4f3_store_dst_fp32_avx512(const float *dst_trans,
                                         const int64_t oh_len,
                                         const int64_t ow_len,
                                         const int64_t dst_h_stride,
                                         float *dst) {
  for (int64_t oh = 0; oh < oh_len; ++oh) {
    const float *dst_trans_ptr = dst_trans + oh * TILE_OUT_W * KERNEL_ONE_REG;
    float *dst_ptr = dst + oh * dst_h_stride;
    for (int64_t ow = 0; ow < ow_len; ++ow) {
      __m512 vres = _mm512_load_ps(dst_trans_ptr);
      _mm512_store_ps(dst_ptr, vres);
      dst_trans_ptr += KERNEL_ONE_REG;
      dst_ptr += KERNEL_ONE_REG;
    }
  }
}