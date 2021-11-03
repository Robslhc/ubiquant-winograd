#include "wgb4f3/avx512/wgb4f3_avx512.h"

#include <float.h>
#include <stdio.h>
#include <stdlib.h>

#include "avxtools.h"
#include "common.h"
#include "timer.h"
#include "wgb4f3/avx512/wgb4f3_avx512_kernels.h"
#include "wgb4f3/wgb4f3.h"

#define PARALLEL_SELECT_FACTOR 1024

static int64_t get_ic_l2_blk(const int64_t channels, const int64_t num_output) {
  int64_t rst = IC_L2_BLK_MAX_LARGE;
  if (channels <= num_output && channels < IC_L2_BLK_MAX_LARGE) {
    rst = IC_L2_BLK_MAX_SMALL;
  }
  if (rst > ROUND_UP(channels, KERNEL_ONE_REG)) {
    rst = ROUND_UP(channels, KERNEL_ONE_REG);
  }
  return rst;
}

static int64_t get_tiles_l2_blk(const int64_t N, const int64_t C,
                                const int64_t K, const int64_t H,
                                const int64_t W, const parallel_mode_t mode) {
  const int64_t dst_h = H - 2;
  const int64_t dst_w = W - 2;
  const int64_t num_tiles_h = DIV_UP(dst_h, TILE_OUT_H);
  const int64_t num_tiles_w = DIV_UP(dst_w, TILE_OUT_W);
  const int64_t num_tiles_b = num_tiles_h * num_tiles_w;
  const int64_t num_tiles = N * num_tiles_b;
  const int64_t num_threads = OMP_MAX_THREADS;

  int64_t tiles_l2_blk = TILE_L2_BLK_MAX_SMALL;
  if (mode == PARALLEL_OUTER) {
    float min_cost = FLT_MAX;
    for (int64_t tl2 = TILE_L2_BLK_MIN; tl2 <= TILE_L2_BLK_MAX_SMALL;
         tl2 += TILE_KERNEL_BLK) {
      const int64_t num_tasks = DIV_UP(DIV_UP(num_tiles, tl2), num_threads);
      const float factor =
          0.1 * (TILE_L2_BLK_MAX_SMALL - tl2) / TILE_L2_BLK_MAX_SMALL;
      const float cost_estimate = num_tasks * tl2 * (1 + factor);
      if (cost_estimate < min_cost) {
        min_cost = cost_estimate;
        tiles_l2_blk = tl2;
      }
    }
  } else {
    tiles_l2_blk = TILE_L2_BLK_MAX_LARGE;
  }

  tiles_l2_blk = ROUND_UP(MIN(tiles_l2_blk, num_tiles), KERNEL_M14_TILE);
  return tiles_l2_blk;
}

static int64_t get_oc_l2_blk(const int64_t channels, const int64_t num_output) {
  int64_t rst = OC_L2_BLK_MAX;
  if (rst > ROUND_UP(num_output, KERNEL_ONE_REG)) {
    rst = ROUND_UP(num_output, KERNEL_ONE_REG);
  }
  return rst;
}

WinogradOptParams init_winconv_4x3_params(const int N, const int C, const int K,
                                          const int H, const int W,
                                          int precompute) {
  WinogradOptParams param;
  param.cvt_filter_precomputed = precompute;

  const int64_t num_threads = OMP_MAX_THREADS;

  const int64_t dst_h = H - 2;
  const int64_t dst_w = W - 2;

  param.padded_ic = ROUND_UP(C, KERNEL_ONE_REG);
  param.padded_oc = ROUND_UP(K, KERNEL_ONE_REG);

  WINO_DEBUG("N = %d, C = %d, K = %d, H = %d, W = %d, padded_ic = %ld, "
             "padded_oc = %ld\n",
             N, C, K, H, W, param.padded_ic, param.padded_oc);

  param.num_tiles_h = DIV_UP(dst_h, TILE_OUT_H);
  param.num_tiles_w = DIV_UP(dst_w, TILE_OUT_W);
  param.num_tiles_b = param.num_tiles_h * param.num_tiles_w;
  param.num_tiles = N * param.num_tiles_b;

  WINO_DEBUG("num_tiles_b = %ld, num_tiles = %ld\n", param.num_tiles_b,
             param.num_tiles);

#ifdef ENABLE_PARALLEL_OUTER
  if (param.num_tiles > PARALLEL_SELECT_FACTOR * num_threads) {
    param.parallel_mode = PARALLEL_OUTER;
  } else {
#endif
    param.parallel_mode = PARALLEL_INNER;
#ifdef ENABLE_PARALLEL_OUTER
  }
#endif

  param.ic_l2_blk = get_ic_l2_blk(C, K);
  param.override_gemm = param.ic_l2_blk >= C;
  param.tiles_l2_blk = get_tiles_l2_blk(N, C, K, H, W, param.parallel_mode);

  WINO_DEBUG("override_gemm = %d, ic_l2_blk = %ld, tiles_l2_blk = %ld, "
             "oc_l2_blk = %ld\n",
             param.override_gemm, param.ic_l2_blk, param.tiles_l2_blk,
             param.oc_l2_blk);

  // image permute and cvt filter
  param.src_permute_len = ROUND_UP(N * param.padded_ic * H * W,
                                   X86_CACHELINE_BYTES / sizeof(float));
  param.dst_permute_len = ROUND_UP(N * param.padded_oc * dst_h * dst_w,
                                   X86_CACHELINE_BYTES / sizeof(float));
  if (param.cvt_filter_precomputed) {
    param.cvt_flt_len = 0;
  } else {
    param.cvt_flt_len = ROUND_UP(TILE_IN_H * TILE_IN_W * param.padded_oc * C,
                                 X86_CACHELINE_BYTES / sizeof(float));
    WINO_DEBUG("cvt flt len = %ld\n", param.cvt_flt_len);
  }

  // used in src trans
  param.blk_tile_in_len = ROUND_UP(TILE_IN_H * TILE_IN_W * KERNEL_ONE_REG,
                                   X86_CACHELINE_BYTES / sizeof(float));
  param.blk_matmul_in_len = ROUND_UP(TILE_IN_H * TILE_IN_W * KERNEL_ONE_REG,
                                     X86_CACHELINE_BYTES / sizeof(float));
  param.src_workspace_len = param.blk_tile_in_len + param.blk_matmul_in_len;

  // used in gemm
  param.src_trans_len =
      ROUND_UP(param.tiles_l2_blk * TILE_IN_H * TILE_IN_W * param.ic_l2_blk,
               X86_CACHELINE_BYTES / sizeof(float));

  // used in dst trans
  param.blk_matmul_out_len = ROUND_UP(TILE_IN_H * TILE_IN_W * KERNEL_ONE_REG,
                                      X86_CACHELINE_BYTES / sizeof(float));
  param.dst_trans_len = 2 * param.blk_matmul_out_len;
  param.workspace_len = MAX(param.src_workspace_len, param.dst_trans_len);

  if (param.parallel_mode == PARALLEL_OUTER) {
    const int64_t tiles_all_threads = num_threads * param.tiles_l2_blk;
    const int64_t oc_l2_cnt = MAX(tiles_all_threads / param.num_tiles, 1);

    param.oc_l2_blk = ROUND_UP(MAX(K / oc_l2_cnt, 1), OC_KERNEL_BLK);

    if (param.override_gemm) {
      param.gemm_out_len =
          ROUND_UP(OC_KERNEL_BLK * TILE_IN_H * TILE_IN_W * param.tiles_l2_blk,
                   X86_CACHELINE_BYTES / sizeof(float));
    } else {
      param.gemm_out_len =
          ROUND_UP(param.oc_l2_blk * TILE_IN_H * TILE_IN_W * param.tiles_l2_blk,
                   X86_CACHELINE_BYTES / sizeof(float));
    }

    param.work_buffer_size = param.src_trans_len * sizeof(float) * num_threads +
                             param.gemm_out_len * sizeof(float) * num_threads +
                             param.workspace_len * sizeof(float) * num_threads;
  } else {
    param.oc_l2_blk = get_oc_l2_blk(C, K);

    if (param.override_gemm) {
      param.gemm_out_len =
          ROUND_UP(param.oc_l2_blk * TILE_IN_H * TILE_IN_W * param.tiles_l2_blk,
                   X86_CACHELINE_BYTES / sizeof(float));
    } else {
      param.gemm_out_len =
          ROUND_UP(param.padded_oc * TILE_IN_H * TILE_IN_W * param.tiles_l2_blk,
                   X86_CACHELINE_BYTES / sizeof(float));
    }

    param.work_buffer_size = param.src_trans_len * sizeof(float) +
                             param.gemm_out_len * sizeof(float) +
                             param.workspace_len * sizeof(float) * num_threads;
  }

  param.temp_buffer_size = param.src_permute_len * sizeof(float) +
                           param.dst_permute_len * sizeof(float) +
                           param.cvt_flt_len * sizeof(float);

  return param;
}

static void transposeToNCHW16(const float *__restrict__ src, const int64_t N,
                              const int64_t C, const int64_t H, const int64_t W,
                              float *__restrict__ dst) {
  const int64_t padded_c = ROUND_UP(C, KERNEL_ONE_REG);
  const int64_t dst_batch_stride = padded_c * H * W;

  PRAGMA_OMP_PARALLEL_FOR()
  for (int64_t n = 0; n < N; ++n) {
    for (int64_t c_outer = 0; c_outer < C; c_outer += KERNEL_ONE_REG) {
      for (int64_t h = 0; h < H; ++h) {
        for (int64_t w = 0; w < W; ++w) {
          int c_inner;
          for (c_inner = 0; c_inner < MIN(C - c_outer, KERNEL_ONE_REG);
               ++c_inner) {
            dst[n * dst_batch_stride + c_outer * (H * W) +
                h * (W * KERNEL_ONE_REG) + w * KERNEL_ONE_REG + c_inner] =
                src[n * C * H * W + (c_outer + c_inner) * (H * W) + h * W + w];
          }
          if (c_inner < KERNEL_ONE_REG) {
            memset32_avx(
                &dst[n * dst_batch_stride + c_outer * (H * W) +
                     h * (w * KERNEL_ONE_REG) + w * KERNEL_ONE_REG + c_inner],
                0.0f, KERNEL_ONE_REG - c_inner);
          }
        }
      }
    }
  }
}

static void transposeFromNCHW16(const float *__restrict__ src, const int64_t N,
                                const int64_t C, const int64_t H,
                                const int64_t W, float *__restrict__ dst) {
  const int64_t padded_c = ROUND_UP(C, KERNEL_ONE_REG);
  const int64_t src_batch_stride = padded_c * H * W;

  PRAGMA_OMP_PARALLEL_FOR()
  for (int64_t n = 0; n < N; ++n) {
    for (int64_t c = 0; c < C; ++c) {
      int64_t c_outer = c / KERNEL_ONE_REG;
      int64_t c_inner = c % KERNEL_ONE_REG;
      for (int64_t h = 0; h < H; ++h) {
        for (int64_t w = 0; w < W; ++w) {
          dst[n * (C * H * W) + c * (H * W) + h * W + w] =
              src[n * src_batch_stride + c_outer * (H * W * KERNEL_ONE_REG) +
                  h * (W * KERNEL_ONE_REG) + w * KERNEL_ONE_REG + c_inner];
        }
      }
    }
  }
}

void winconv_4x3_avx512(WinogradOptParams param, float *__restrict__ image,
                        const int irows, const int icols, const int C,
                        float *__restrict__ filter, const int K,
                        const int batch, float *__restrict__ out,
                        float *__restrict__ tmpbuf) {
  const int64_t src_h = irows;
  const int64_t src_w = icols;
  const int64_t dst_h = src_h - 2;
  const int64_t dst_w = src_w - 2;

  const int64_t src_batch_stride = param.padded_ic * src_h * src_w;
  const int64_t dst_batch_stride = param.padded_oc * dst_h * dst_w;

  // src permute
  float *src = (float *)tmpbuf;
  float *dst = tmpbuf + param.src_permute_len;
  START_TIMER(param.timer, CONVERT_TO_NCHW16_TIMER);
  transposeToNCHW16(image, batch, C, irows, icols, src);
  END_TIMER(param.timer, CONVERT_TO_NCHW16_TIMER);

  // cvt filter
  float *filter_cvt;
  START_TIMER(param.timer, CONVERT_FILTER_TIMER);
  if (param.cvt_filter_precomputed) {
    filter_cvt = filter;
  } else {
    filter_cvt = dst + param.dst_permute_len;
    winconv_4x3_avx512_cvt_flt(param, filter, C, K, filter_cvt);
  }
  END_TIMER(param.timer, CONVERT_FILTER_TIMER);

  START_TIMER(param.timer, WINO_COMPUTE_TIMER);
  if (param.parallel_mode == PARALLEL_OUTER) {
    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
    for (int64_t oc_l2 = 0; oc_l2 < K; oc_l2 += param.oc_l2_blk) {
      for (int64_t tile_l2 = 0; tile_l2 < param.num_tiles;
           tile_l2 += param.tiles_l2_blk) {
        wgb4f3_kernel_params kernel_param;

        const int64_t l2_oc_compute =
            MIN(param.oc_l2_blk, param.padded_oc - oc_l2);
        const int64_t l2_tiles_compute =
            MIN(param.tiles_l2_blk, param.num_tiles - tile_l2);
        const int64_t tile_body_compute =
            ROUND(l2_tiles_compute, TILE_KERNEL_BLK);
        const int64_t tile_tail_compute = l2_tiles_compute - tile_body_compute;

        float *src_trans =
            dst + param.dst_permute_len + param.cvt_flt_len +
            (param.src_trans_len + param.gemm_out_len + param.workspace_len) *
                OMP_THREAD_ID;
        float *gemm_out_buf = src_trans + param.src_trans_len;
        float *src_work_space = gemm_out_buf + param.gemm_out_len;
        float *dst_work_space = src_work_space;
        float *tile_in_buf = src_work_space;
        float *matmul_in_buf = tile_in_buf + param.blk_tile_in_len;
        float *dst_trans_buf = dst_work_space;

        // src trans
        for (int ic_l2 = 0; ic_l2 < C; ic_l2 += param.ic_l2_blk) {
          const int64_t l2_ic_compute = MIN(param.ic_l2_blk, C - ic_l2);
          const int64_t l2_ic_compute_padded =
              ROUND_UP(l2_ic_compute, KERNEL_ONE_REG);
          const int is_first_ic = (ic_l2 == 0);
          const int is_last_ic = (ic_l2 + param.ic_l2_blk >= C);
          kernel_param.channels = l2_ic_compute;
          kernel_param.load_dst = !is_first_ic;
          kernel_param.flt_ocb_stride = l2_ic_compute * KERNEL_ONE_REG;
          kernel_param.dst_ocb_stride = l2_tiles_compute * KERNEL_ONE_REG;

          for (int64_t tile_blk = tile_l2;
               tile_blk < tile_l2 + l2_tiles_compute;
               tile_blk += TILE_KERNEL_BLK) {
            const int64_t blk_tile_compute =
                MIN(tile_l2 + l2_tiles_compute - tile_blk, TILE_KERNEL_BLK);
            for (int64_t ic_blk = ic_l2; ic_blk < ic_l2 + l2_ic_compute_padded;
                 ic_blk += KERNEL_ONE_REG) {
              for (int64_t t = 0; t < blk_tile_compute; ++t) {
                TileIndex tIndex = calculateTileIndex(param, tile_blk + t);

                const int64_t b = tIndex.b;
                const int64_t oh = tIndex.th * TILE_OUT_H;
                const int64_t ow = tIndex.tw * TILE_OUT_W;
                const int64_t ih = oh;
                const int64_t iw = ow;

                float *blk_src_trans =
                    src_trans + (tile_blk - tile_l2) * l2_ic_compute_padded +
                    (ic_blk - ic_l2) * blk_tile_compute + t * KERNEL_ONE_REG;
                const float *base_src =
                    src + b * src_batch_stride + ic_blk * src_h * src_w;

                winograd_b4f3_srctrans_fp32_avx512(
                    base_src, ih, iw, src_h, src_w,
                    l2_tiles_compute * l2_ic_compute_padded, tile_in_buf,
                    matmul_in_buf, blk_src_trans);
              }
            }
          }

          // gemm
          for (int64_t oc_blk = oc_l2; oc_blk < oc_l2 + l2_oc_compute;
               oc_blk += OC_KERNEL_BLK) {
            const int64_t blk_oc_compute =
                MIN(param.padded_oc - oc_blk, OC_KERNEL_BLK);
            const int64_t kernel_oc_len =
                ROUND_UP(blk_oc_compute, KERNEL_ONE_REG);

            for (int64_t ti = 0; ti < TILE_IN_H * TILE_IN_W; ++ti) {
              float *src_trans_in =
                  src_trans + ti * l2_tiles_compute * l2_ic_compute_padded;
              const float *cvt_flt_in =
                  filter_cvt + ic_l2 * TILE_IN_H * TILE_IN_W * param.padded_oc +
                  ti * l2_ic_compute * param.padded_oc + oc_blk * l2_ic_compute;
              float *gemm_out;
              if (param.override_gemm) {
                gemm_out =
                    gemm_out_buf + ti * blk_oc_compute * l2_tiles_compute;
              } else {
                gemm_out = gemm_out_buf +
                           ti * l2_oc_compute * l2_tiles_compute +
                           l2_tiles_compute * (oc_blk - oc_l2);
              }
              kernel_param.src = src_trans_in;
              kernel_param.flt = cvt_flt_in;
              kernel_param.dst = gemm_out;

              if (tile_body_compute) {
                kernel_param.tiles = tile_body_compute;
                kernel_param.src_tkb_stride =
                    TILE_KERNEL_BLK * l2_ic_compute_padded;
                winograd_b4f3_gemm_kernel_fp32_avx512(
                    kernel_oc_len, TILE_KERNEL_BLK, kernel_param);
                kernel_param.src += tile_body_compute * l2_ic_compute_padded;
                kernel_param.dst += tile_body_compute * KERNEL_ONE_REG;
              }
              if (tile_tail_compute) {
                kernel_param.tiles = tile_tail_compute;
                kernel_param.src_tkb_stride =
                    tile_tail_compute * l2_ic_compute_padded;
                winograd_b4f3_gemm_kernel_fp32_avx512(
                    kernel_oc_len, tile_tail_compute, kernel_param);
              }
            }

            if (is_last_ic) {
              for (int64_t ocb = oc_blk; ocb < oc_blk + blk_oc_compute;
                   ocb += KERNEL_ONE_REG) {
                for (int64_t tile_blk = tile_l2;
                     tile_blk < tile_l2 + l2_tiles_compute;
                     tile_blk += TILE_KERNEL_BLK) {
                  const int64_t blk_tile_compute = MIN(
                      tile_l2 + l2_tiles_compute - tile_blk, TILE_KERNEL_BLK);

                  for (int64_t t = 0; t < blk_tile_compute; ++t) {
                    TileIndex tIndex = calculateTileIndex(param, tile_blk + t);
                    const int64_t b = tIndex.b;
                    const int64_t oh = tIndex.th * TILE_OUT_H;
                    const int64_t ow = tIndex.tw * TILE_OUT_W;
                    const int64_t oh_len = MIN(dst_h - oh, TILE_OUT_H);
                    const int64_t ow_len = MIN(dst_w - ow, TILE_OUT_W);

                    float *blk_dst = dst + b * dst_batch_stride +
                                     ocb * (dst_h * dst_w) +
                                     (oh * dst_w + ow) * KERNEL_ONE_REG;
                    float *gemm_out;
                    int64_t gemm_out_ti_stride;
                    if (param.override_gemm) {
                      gemm_out = gemm_out_buf +
                                 l2_tiles_compute * (ocb - oc_blk) +
                                 (tile_blk - tile_l2 + t) * KERNEL_ONE_REG;
                      gemm_out_ti_stride = l2_tiles_compute * blk_oc_compute;
                    } else {
                      gemm_out = gemm_out_buf +
                                 l2_tiles_compute * (ocb - oc_l2) +
                                 (tile_blk - tile_l2 + t) * KERNEL_ONE_REG;
                      gemm_out_ti_stride = l2_tiles_compute * l2_oc_compute;
                    }

                    if (oh_len == TILE_OUT_H && ow_len == TILE_OUT_W) {
                      winograd_b4f3_dsttrans_fp32_avx512(
                          gemm_out, gemm_out_ti_stride, dst_w * KERNEL_ONE_REG,
                          dst_trans_buf, blk_dst);
                    } else {
                      float *dst_buf = dst_trans_buf + param.blk_matmul_out_len;
                      winograd_b4f3_dsttrans_fp32_avx512(
                          gemm_out, gemm_out_ti_stride,
                          TILE_OUT_W * KERNEL_ONE_REG, dst_trans_buf, dst_buf);
                      winograd_b4f3_store_dst_fp32_avx512(
                          dst_buf, oh_len, ow_len, dst_w * KERNEL_ONE_REG,
                          blk_dst);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  } else {
    PRAGMA_OMP_PARALLEL() {
      for (int64_t tile_l2 = 0; tile_l2 < param.num_tiles;
           tile_l2 += param.tiles_l2_blk) {

        wgb4f3_kernel_params kernel_param;

        // compute all the tiles in l2 cache
        const int64_t l2_tiles_compute =
            MIN(param.tiles_l2_blk, param.num_tiles - tile_l2);
        const int64_t tile_body_compute =
            ROUND(l2_tiles_compute, TILE_KERNEL_BLK);
        const int64_t tile_tail_compute = l2_tiles_compute - tile_body_compute;

        float *src_trans = dst + param.dst_permute_len + param.cvt_flt_len;
        float *gemm_out_buf = src_trans + param.src_trans_len;
        float *src_work_space = gemm_out_buf + param.gemm_out_len +
                                OMP_THREAD_ID * (param.workspace_len);
        float *dst_work_space = src_work_space;

        for (int64_t ic_l2 = 0; ic_l2 < C; ic_l2 += param.ic_l2_blk) {
          const int64_t l2_ic_compute = MIN(param.ic_l2_blk, C - ic_l2);
          const int64_t l2_ic_compute_padded =
              ROUND_UP(l2_ic_compute, KERNEL_ONE_REG);
          const int is_first_ic = (ic_l2 == 0);
          const int is_last_ic = (ic_l2 + param.ic_l2_blk >= C);
          kernel_param.channels = l2_ic_compute;
          kernel_param.load_dst = !is_first_ic;
          kernel_param.flt_ocb_stride = l2_ic_compute * KERNEL_ONE_REG;
          kernel_param.dst_ocb_stride = l2_tiles_compute * KERNEL_ONE_REG;

          // do src trans in different blocks, aka. (1 tile * 16)
          PRAGMA_OMP_FOR_COLLAPSE(2)
          for (int64_t ic_blk = ic_l2; ic_blk < ic_l2 + l2_ic_compute_padded;
               ic_blk += KERNEL_ONE_REG) {

            for (int64_t tile_blk = tile_l2;
                 tile_blk < tile_l2 + l2_tiles_compute; ++tile_blk) {
              float *tile_in_buf = src_work_space;
              float *matmul_in_buf = tile_in_buf + param.blk_tile_in_len;
              WINO_DEBUG("tile_in_buf = %x\n", tile_in_buf);

              TileIndex tIndex = calculateTileIndex(param, tile_blk);
              const int64_t b = tIndex.b;
              const int64_t oh = tIndex.th * TILE_OUT_H;
              const int64_t ow = tIndex.tw * TILE_OUT_W;
              const int64_t ih = oh;
              const int64_t iw = ow;
              const int64_t t = tile_blk % TILE_KERNEL_BLK;

              const int64_t blk_tile_compute = MIN(
                  tile_l2 + l2_tiles_compute - (tile_blk - t), TILE_KERNEL_BLK);
              // previous blocks which cover l2 ic + tiles in the same block  +
              // previous blocks with same ic
              float *blk_src_trans =
                  src_trans + (tile_blk - tile_l2 - t) * l2_ic_compute_padded +
                  (ic_blk - ic_l2) * blk_tile_compute + t * KERNEL_ONE_REG;
              const float *base_src =
                  src + b * src_batch_stride + ic_blk * src_h * src_w;
              WINO_DEBUG(
                  "Start src trans tile_blk = %ld, aka. b = %ld, t = %d, "
                  "blk_tile_compute = %d\n",
                  tile_blk, b, t, blk_tile_compute);
              WINO_DEBUG(
                  "Store blk_src_trans offset = %d * l2_ic_compute_padded + "
                  "%d * (ic_blk - ic_l2) + %d * 16\n",
                  (tile_blk - tile_l2 - t), blk_tile_compute, t);

              winograd_b4f3_srctrans_fp32_avx512(
                  base_src, ih, iw, src_h, src_w,
                  l2_tiles_compute * l2_ic_compute_padded, tile_in_buf,
                  matmul_in_buf, blk_src_trans);
            }
          }

          for (int64_t oc_l2 = 0; oc_l2 < param.padded_oc;
               oc_l2 += param.oc_l2_blk) {
            const int64_t l2_oc_compute =
                MIN(param.oc_l2_blk, param.padded_oc - oc_l2);

            PRAGMA_OMP_FOR_COLLAPSE(2)
            // batch gemm on current ic l2 blk
            for (int64_t ti = 0; ti < TILE_IN_H * TILE_IN_W; ++ti) {
              for (int64_t oc_blk = oc_l2; oc_blk < oc_l2 + l2_oc_compute;
                   oc_blk += OC_KERNEL_BLK) {
                int64_t blk_oc_compute =
                    MIN(param.padded_oc - oc_blk, OC_KERNEL_BLK);
                int64_t kernel_oc_len =
                    ROUND_UP(blk_oc_compute, KERNEL_ONE_REG);
                float *src_trans_in =
                    src_trans + ti * l2_tiles_compute * l2_ic_compute_padded;
                float *cvt_flt_in =
                    filter_cvt +
                    ic_l2 * TILE_IN_H * TILE_IN_W * param.padded_oc +
                    ti * l2_ic_compute * param.padded_oc +
                    oc_blk * l2_ic_compute;

                float *gemm_out;
                if (param.override_gemm) {
                  gemm_out = gemm_out_buf +
                             ti * l2_tiles_compute * l2_oc_compute +
                             l2_tiles_compute * (oc_blk - oc_l2);
                } else {
                  gemm_out = gemm_out_buf +
                             ti * l2_tiles_compute * param.padded_oc +
                             l2_tiles_compute * oc_blk;
                }
                kernel_param.src = src_trans_in;
                kernel_param.flt = cvt_flt_in;
                kernel_param.dst = gemm_out;
                if (tile_body_compute) {
                  // WINO_DEBUG("tl2 = %d, body compute\n", tile_l2);
                  kernel_param.tiles = tile_body_compute;
                  kernel_param.src_tkb_stride =
                      TILE_KERNEL_BLK * l2_ic_compute_padded;
                  winograd_b4f3_gemm_kernel_fp32_avx512(
                      kernel_oc_len, TILE_KERNEL_BLK, kernel_param);
                  kernel_param.src += tile_body_compute * l2_ic_compute_padded;
                  kernel_param.dst += tile_body_compute * KERNEL_ONE_REG;
                }
                if (tile_tail_compute) {
                  // WINO_DEBUG("tl2 = %d, tail compute\n", tile_l2);
                  kernel_param.tiles = tile_tail_compute;
                  kernel_param.src_tkb_stride =
                      tile_tail_compute * l2_ic_compute_padded;
                  winograd_b4f3_gemm_kernel_fp32_avx512(
                      kernel_oc_len, tile_tail_compute, kernel_param);
                }
              }
            }

            // dst trans
            if (is_last_ic) {
              PRAGMA_OMP_FOR_COLLAPSE(2)
              for (int64_t oc_blk = oc_l2; oc_blk < oc_l2 + l2_oc_compute;
                   oc_blk += KERNEL_ONE_REG) {
                for (int64_t tile_blk = tile_l2;
                     tile_blk < tile_l2 + l2_tiles_compute; ++tile_blk) {
                  float *dst_trans_buf = dst_work_space;

                  TileIndex tIndex = calculateTileIndex(param, tile_blk);
                  const int64_t b = tIndex.b;
                  const int64_t oh = tIndex.th * TILE_OUT_H;
                  const int64_t ow = tIndex.tw * TILE_OUT_W;
                  const int64_t oh_len = MIN(dst_h - oh, TILE_OUT_H);
                  const int64_t ow_len = MIN(dst_w - ow, TILE_OUT_W);

                  float *blk_dst = dst + b * dst_batch_stride +
                                   oc_blk * (dst_h * dst_w) +
                                   (oh * dst_w + ow) * KERNEL_ONE_REG;
                  float *gemm_out;
                  int64_t gemm_out_ti_stride;
                  if (param.override_gemm) {
                    gemm_out = gemm_out_buf +
                               l2_tiles_compute * (oc_blk - oc_l2) +
                               (tile_blk - tile_l2) * KERNEL_ONE_REG;
                    gemm_out_ti_stride = l2_tiles_compute * l2_oc_compute;
                  } else {
                    gemm_out = gemm_out_buf + l2_tiles_compute * oc_blk +
                               (tile_blk - tile_l2) * KERNEL_ONE_REG;
                    gemm_out_ti_stride = l2_tiles_compute * param.padded_oc;
                  }

                  if (oh_len == TILE_OUT_H && ow_len == TILE_OUT_W) {
                    winograd_b4f3_dsttrans_fp32_avx512(
                        gemm_out, gemm_out_ti_stride, dst_w * KERNEL_ONE_REG,
                        dst_trans_buf, blk_dst);
                  } else {
                    float *dst_buf = dst_trans_buf + param.blk_matmul_out_len;
                    winograd_b4f3_dsttrans_fp32_avx512(
                        gemm_out, gemm_out_ti_stride,
                        TILE_OUT_W * KERNEL_ONE_REG, dst_trans_buf, dst_buf);
                    winograd_b4f3_store_dst_fp32_avx512(dst_buf, oh_len, ow_len,
                                                        dst_w * KERNEL_ONE_REG,
                                                        blk_dst);
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  END_TIMER(param.timer, WINO_COMPUTE_TIMER);

  START_TIMER(param.timer, CONVERT_FROM_NCHW16_TIMER);
  // dst permute back
  transposeFromNCHW16(dst, batch, K, dst_h, dst_w, out);
  END_TIMER(param.timer, CONVERT_FROM_NCHW16_TIMER);
}

void winconv_4x3_avx512_cvt_flt(WinogradOptParams param,
                                const float *__restrict__ filter, const int C,
                                const int K, float *__restrict__ filter_cvt) {
  const float mat_G[TILE_IN_H][FLT_H] = {
      {1.f / 4, 0.f, 0.f},
      {-1.f / 6, -1.f / 6, -1.f / 6},
      {-1.f / 6, 1.f / 6, -1.f / 6},
      {1.f / 24, 1.f / 12, 1.f / 6},
      {1.f / 24, -1.f / 12, 1.f / 6},
      {0.f, 0.f, 1.f},
  };

  // convert to [IC_L2_BLK, th, tw, O/16, ic_l2, 16]
  for (int64_t ic_l2 = 0; ic_l2 < param.padded_ic; ic_l2 += param.ic_l2_blk) {
    for (int64_t oc_blk = 0; oc_blk < param.padded_oc;
         oc_blk += KERNEL_ONE_REG) {
      const int64_t l2_ic_compute = MIN(C - ic_l2, param.ic_l2_blk);
      const int64_t blk_oc_compute = MIN(K - oc_blk, KERNEL_ONE_REG);
      float mat_T[TILE_IN_H][FLT_W];
      for (int64_t ic = ic_l2; ic < ic_l2 + l2_ic_compute; ++ic) {
        const float *filter_in =
            filter + oc_blk * C * FLT_H * FLT_W + ic * FLT_H * FLT_W;
        float *flt_cvt_out =
            filter_cvt + ic_l2 * TILE_IN_H * TILE_IN_W * param.padded_oc +
            oc_blk * l2_ic_compute + (ic - ic_l2) * KERNEL_ONE_REG;

        for (int64_t oc = 0; oc < blk_oc_compute; ++oc) {
          // G * filter
          for (int64_t i = 0; i < TILE_IN_H; ++i) {
            for (int64_t j = 0; j < FLT_W; ++j) {
              float sum = 0.0f;
              for (int64_t k = 0; k < FLT_H; ++k) {
                sum += mat_G[i][k] *
                       filter_in[oc * C * FLT_H * FLT_W + k * FLT_W + j];
              }
              mat_T[i][j] = sum;
            }
          }
          // (G * filter) * GT
          for (int64_t i = 0; i < TILE_IN_H; ++i) {
            for (int64_t j = 0; j < TILE_IN_W; ++j) {
              float sum = 0.0f;
              for (int64_t k = 0; k < FLT_W; ++k) {
                sum += mat_T[i][k] * mat_G[j][k];
              }
              flt_cvt_out[(i * TILE_IN_W + j) * param.padded_oc *
                              l2_ic_compute +
                          oc] = sum;
            }
          }
        }

        if (blk_oc_compute < KERNEL_ONE_REG) {
          for (int64_t i = 0; i < TILE_IN_H; ++i) {
            for (int64_t j = 0; j < TILE_IN_W; ++j) {
              for (int64_t oc = blk_oc_compute; oc < KERNEL_ONE_REG; ++oc) {
                flt_cvt_out[(i * TILE_IN_W + j) * param.padded_oc *
                                l2_ic_compute +
                            oc] = 0.0f;
              }
            }
          }
        }
      }
    }
  }
}
