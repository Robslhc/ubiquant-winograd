#ifndef SRC_WGB4F3_AVX512_WGB4F3_AVX512_KERNEL_H_
#define SRC_WGB4F3_AVX512_WGB4F3_AVX512_KERNEL_H_

#include <immintrin.h>
#include <xmmintrin.h>

#include "common.h"

// gemm: TILE->M, IC->K, OC->N
#define KERNEL_ONE_REG 16
#define KERNEL_M14_TILE 14
#define KERNEL_M14_OC_REG 2
#define TILE_KERNEL_BLK KERNEL_M14_TILE
#define OC_KERNEL_BLK (KERNEL_M14_OC_REG * KERNEL_ONE_REG)

// L2 BLK
#define IC_L2_BLK_MAX_LARGE (16 * KERNEL_ONE_REG)
#define IC_L2_BLK_MAX_SMALL (8 * KERNEL_ONE_REG)
#define TILE_L2_BLK_MIN (1 * KERNEL_M14_TILE)
#define TILE_L2_BLK_MAX_SMALL (2 * KERNEL_M14_TILE)
#define TILE_L2_BLK_MAX_LARGE (8 * KERNEL_M14_TILE)
#define OC_L2_BLK_MAX (16 * OC_KERNEL_BLK)

// KERNEL PARAMETERS
typedef struct 
{
    const float *src;
    float *dst;
    const float *flt;
    int64_t tiles;
    int64_t channels;
    int64_t src_tkb_stride;
    int64_t dst_ocb_stride;
    int64_t flt_ocb_stride;
    int load_dst;
} wgb4f3_kernel_params;


#define WINOGRAD_B4F3_T14O32_KERNEL_AVX512_IC_COMPUTE(T_LEN, IC_SRC, IC_FLT)   \
  do {                                                                         \
    zmm28 = _mm512_loadu_ps(IC_FLT + 0 * flt_ocb_stride);                      \
    zmm29 = _mm512_loadu_ps(IC_FLT + 1 * flt_ocb_stride);                      \
    if (T_LEN > 12) {                                                          \
      _mm_prefetch((const char *)(IC_FLT + 0 * flt_ocb_stride +                \
                                  KERNEL_ONE_REG * KERNEL_ONE_REG),            \
                   _MM_HINT_T0);                                               \
      _mm_prefetch((const char *)(IC_FLT + 1 * flt_ocb_stride +                \
                                  KERNEL_ONE_REG * KERNEL_ONE_REG),            \
                   _MM_HINT_T0);                                               \
    }                                                                          \
    if (T_LEN > 0) {                                                           \
      zmm30 = _mm512_set1_ps(IC_SRC[0 + 0 * KERNEL_ONE_REG]);                  \
      zmm0 = _mm512_fmadd_ps(zmm28, zmm30, zmm0);                              \
      zmm14 = _mm512_fmadd_ps(zmm29, zmm30, zmm14);                            \
    }                                                                          \
    if (T_LEN > 1) {                                                           \
      zmm31 = _mm512_set1_ps(IC_SRC[0 + 1 * KERNEL_ONE_REG]);                  \
      zmm1 = _mm512_fmadd_ps(zmm28, zmm31, zmm1);                              \
      zmm15 = _mm512_fmadd_ps(zmm29, zmm31, zmm15);                            \
    }                                                                          \
    if (T_LEN > 2) {                                                           \
      zmm30 = _mm512_set1_ps(IC_SRC[0 + 2 * KERNEL_ONE_REG]);                  \
      zmm2 = _mm512_fmadd_ps(zmm28, zmm30, zmm2);                              \
      zmm16 = _mm512_fmadd_ps(zmm29, zmm30, zmm16);                            \
    }                                                                          \
    if (T_LEN > 3) {                                                           \
      zmm31 = _mm512_set1_ps(IC_SRC[0 + 3 * KERNEL_ONE_REG]);                  \
      zmm3 = _mm512_fmadd_ps(zmm28, zmm31, zmm3);                              \
      zmm17 = _mm512_fmadd_ps(zmm29, zmm31, zmm17);                            \
    }                                                                          \
    if (T_LEN > 4) {                                                           \
      zmm30 = _mm512_set1_ps(IC_SRC[0 + 4 * KERNEL_ONE_REG]);                  \
      zmm4 = _mm512_fmadd_ps(zmm28, zmm30, zmm4);                              \
      zmm18 = _mm512_fmadd_ps(zmm29, zmm30, zmm18);                            \
    }                                                                          \
    if (T_LEN > 5) {                                                           \
      zmm31 = _mm512_set1_ps(IC_SRC[0 + 5 * KERNEL_ONE_REG]);                  \
      zmm5 = _mm512_fmadd_ps(zmm28, zmm31, zmm5);                              \
      zmm19 = _mm512_fmadd_ps(zmm29, zmm31, zmm19);                            \
    }                                                                          \
    if (T_LEN > 6) {                                                           \
      zmm30 = _mm512_set1_ps(IC_SRC[0 + 6 * KERNEL_ONE_REG]);                  \
      zmm6 = _mm512_fmadd_ps(zmm28, zmm30, zmm6);                              \
      zmm20 = _mm512_fmadd_ps(zmm29, zmm30, zmm20);                            \
    }                                                                          \
    if (T_LEN > 7) {                                                           \
      zmm31 = _mm512_set1_ps(IC_SRC[0 + 7 * KERNEL_ONE_REG]);                  \
      zmm7 = _mm512_fmadd_ps(zmm28, zmm31, zmm7);                              \
      zmm21 = _mm512_fmadd_ps(zmm29, zmm31, zmm21);                            \
    }                                                                          \
    if (T_LEN > 8) {                                                           \
      zmm30 = _mm512_set1_ps(IC_SRC[0 + 8 * KERNEL_ONE_REG]);                  \
      zmm8 = _mm512_fmadd_ps(zmm28, zmm30, zmm8);                              \
      zmm22 = _mm512_fmadd_ps(zmm29, zmm30, zmm22);                            \
    }                                                                          \
    if (T_LEN > 9) {                                                           \
      zmm31 = _mm512_set1_ps(IC_SRC[0 + 9 * KERNEL_ONE_REG]);                  \
      zmm9 = _mm512_fmadd_ps(zmm28, zmm31, zmm9);                              \
      zmm23 = _mm512_fmadd_ps(zmm29, zmm31, zmm23);                            \
    }                                                                          \
    if (T_LEN > 10) {                                                          \
      zmm30 = _mm512_set1_ps(IC_SRC[0 + 10 * KERNEL_ONE_REG]);                 \
      zmm10 = _mm512_fmadd_ps(zmm28, zmm30, zmm10);                            \
      zmm24 = _mm512_fmadd_ps(zmm29, zmm30, zmm24);                            \
    }                                                                          \
    if (T_LEN > 11) {                                                          \
      zmm31 = _mm512_set1_ps(IC_SRC[0 + 11 * KERNEL_ONE_REG]);                 \
      zmm11 = _mm512_fmadd_ps(zmm28, zmm31, zmm11);                            \
      zmm25 = _mm512_fmadd_ps(zmm29, zmm31, zmm25);                            \
    }                                                                          \
    if (T_LEN > 12) {                                                          \
      zmm30 = _mm512_set1_ps(IC_SRC[0 + 12 * KERNEL_ONE_REG]);                 \
      zmm12 = _mm512_fmadd_ps(zmm28, zmm30, zmm12);                            \
      zmm26 = _mm512_fmadd_ps(zmm29, zmm30, zmm26);                            \
    }                                                                          \
    if (T_LEN > 13) {                                                          \
      zmm31 = _mm512_set1_ps(IC_SRC[0 + 13 * KERNEL_ONE_REG]);                 \
      zmm13 = _mm512_fmadd_ps(zmm28, zmm31, zmm13);                            \
      zmm27 = _mm512_fmadd_ps(zmm29, zmm31, zmm27);                            \
    }                                                                          \
  } while (0)

#define WINOGRAD_B4F3_T14O32_KERNEL_AVX512_ITER_T(KERNEL_PARAMS, T_LEN, T_SRC, \
                                                  T_DST)                       \
  if (load_dst) {                                                              \
    float *l_dst = T_DST;                                                      \
    if (T_LEN > 0)                                                             \
      zmm0 = _mm512_loadu_ps(l_dst + 0 * KERNEL_ONE_REG);                      \
    if (T_LEN > 1)                                                             \
      zmm1 = _mm512_loadu_ps(l_dst + 1 * KERNEL_ONE_REG);                      \
    if (T_LEN > 2)                                                             \
      zmm2 = _mm512_loadu_ps(l_dst + 2 * KERNEL_ONE_REG);                      \
    if (T_LEN > 3)                                                             \
      zmm3 = _mm512_loadu_ps(l_dst + 3 * KERNEL_ONE_REG);                      \
    if (T_LEN > 4)                                                             \
      zmm4 = _mm512_loadu_ps(l_dst + 4 * KERNEL_ONE_REG);                      \
    if (T_LEN > 5)                                                             \
      zmm5 = _mm512_loadu_ps(l_dst + 5 * KERNEL_ONE_REG);                      \
    if (T_LEN > 6)                                                             \
      zmm6 = _mm512_loadu_ps(l_dst + 6 * KERNEL_ONE_REG);                      \
    if (T_LEN > 7)                                                             \
      zmm7 = _mm512_loadu_ps(l_dst + 7 * KERNEL_ONE_REG);                      \
    if (T_LEN > 8)                                                             \
      zmm8 = _mm512_loadu_ps(l_dst + 8 * KERNEL_ONE_REG);                      \
    if (T_LEN > 9)                                                             \
      zmm9 = _mm512_loadu_ps(l_dst + 9 * KERNEL_ONE_REG);                      \
    if (T_LEN > 10)                                                            \
      zmm10 = _mm512_loadu_ps(l_dst + 10 * KERNEL_ONE_REG);                    \
    if (T_LEN > 11)                                                            \
      zmm11 = _mm512_loadu_ps(l_dst + 11 * KERNEL_ONE_REG);                    \
    if (T_LEN > 12)                                                            \
      zmm12 = _mm512_loadu_ps(l_dst + 12 * KERNEL_ONE_REG);                    \
    if (T_LEN > 13)                                                            \
      zmm13 = _mm512_loadu_ps(l_dst + 13 * KERNEL_ONE_REG);                    \
    l_dst += dst_ocb_stride;                                                   \
    if (T_LEN > 0)                                                             \
      zmm14 = _mm512_loadu_ps(l_dst + 0 * KERNEL_ONE_REG);                     \
    if (T_LEN > 1)                                                             \
      zmm15 = _mm512_loadu_ps(l_dst + 1 * KERNEL_ONE_REG);                     \
    if (T_LEN > 2)                                                             \
      zmm16 = _mm512_loadu_ps(l_dst + 2 * KERNEL_ONE_REG);                     \
    if (T_LEN > 3)                                                             \
      zmm17 = _mm512_loadu_ps(l_dst + 3 * KERNEL_ONE_REG);                     \
    if (T_LEN > 4)                                                             \
      zmm18 = _mm512_loadu_ps(l_dst + 4 * KERNEL_ONE_REG);                     \
    if (T_LEN > 5)                                                             \
      zmm19 = _mm512_loadu_ps(l_dst + 5 * KERNEL_ONE_REG);                     \
    if (T_LEN > 6)                                                             \
      zmm20 = _mm512_loadu_ps(l_dst + 6 * KERNEL_ONE_REG);                     \
    if (T_LEN > 7)                                                             \
      zmm21 = _mm512_loadu_ps(l_dst + 7 * KERNEL_ONE_REG);                     \
    if (T_LEN > 8)                                                             \
      zmm22 = _mm512_loadu_ps(l_dst + 8 * KERNEL_ONE_REG);                     \
    if (T_LEN > 9)                                                             \
      zmm23 = _mm512_loadu_ps(l_dst + 9 * KERNEL_ONE_REG);                     \
    if (T_LEN > 10)                                                            \
      zmm24 = _mm512_loadu_ps(l_dst + 10 * KERNEL_ONE_REG);                    \
    if (T_LEN > 11)                                                            \
      zmm25 = _mm512_loadu_ps(l_dst + 11 * KERNEL_ONE_REG);                    \
    if (T_LEN > 12)                                                            \
      zmm26 = _mm512_loadu_ps(l_dst + 12 * KERNEL_ONE_REG);                    \
    if (T_LEN > 13)                                                            \
      zmm27 = _mm512_loadu_ps(l_dst + 13 * KERNEL_ONE_REG);                    \
  } else {                                                                     \
    if (T_LEN > 0)                                                             \
      zmm0 = _mm512_setzero_ps();                                              \
    if (T_LEN > 1)                                                             \
      zmm1 = _mm512_setzero_ps();                                              \
    if (T_LEN > 2)                                                             \
      zmm2 = _mm512_setzero_ps();                                              \
    if (T_LEN > 3)                                                             \
      zmm3 = _mm512_setzero_ps();                                              \
    if (T_LEN > 4)                                                             \
      zmm4 = _mm512_setzero_ps();                                              \
    if (T_LEN > 5)                                                             \
      zmm5 = _mm512_setzero_ps();                                              \
    if (T_LEN > 6)                                                             \
      zmm6 = _mm512_setzero_ps();                                              \
    if (T_LEN > 7)                                                             \
      zmm7 = _mm512_setzero_ps();                                              \
    if (T_LEN > 8)                                                             \
      zmm8 = _mm512_setzero_ps();                                              \
    if (T_LEN > 9)                                                             \
      zmm9 = _mm512_setzero_ps();                                              \
    if (T_LEN > 10)                                                            \
      zmm10 = _mm512_setzero_ps();                                             \
    if (T_LEN > 11)                                                            \
      zmm11 = _mm512_setzero_ps();                                             \
    if (T_LEN > 12)                                                            \
      zmm12 = _mm512_setzero_ps();                                             \
    if (T_LEN > 13)                                                            \
      zmm13 = _mm512_setzero_ps();                                             \
    if (T_LEN > 0)                                                             \
      zmm14 = _mm512_setzero_ps();                                             \
    if (T_LEN > 1)                                                             \
      zmm15 = _mm512_setzero_ps();                                             \
    if (T_LEN > 2)                                                             \
      zmm16 = _mm512_setzero_ps();                                             \
    if (T_LEN > 3)                                                             \
      zmm17 = _mm512_setzero_ps();                                             \
    if (T_LEN > 4)                                                             \
      zmm18 = _mm512_setzero_ps();                                             \
    if (T_LEN > 5)                                                             \
      zmm19 = _mm512_setzero_ps();                                             \
    if (T_LEN > 6)                                                             \
      zmm20 = _mm512_setzero_ps();                                             \
    if (T_LEN > 7)                                                             \
      zmm21 = _mm512_setzero_ps();                                             \
    if (T_LEN > 8)                                                             \
      zmm22 = _mm512_setzero_ps();                                             \
    if (T_LEN > 9)                                                             \
      zmm23 = _mm512_setzero_ps();                                             \
    if (T_LEN > 10)                                                            \
      zmm24 = _mm512_setzero_ps();                                             \
    if (T_LEN > 11)                                                            \
      zmm25 = _mm512_setzero_ps();                                             \
    if (T_LEN > 12)                                                            \
      zmm26 = _mm512_setzero_ps();                                             \
    if (T_LEN > 13)                                                            \
      zmm27 = _mm512_setzero_ps();                                             \
  }                                                                            \
  const float *icb_src = T_SRC;                                                \
  const float *icb_flt = KERNEL_PARAMS.flt;        \
  int64_t icb = KERNEL_PARAMS.channels;                                       \
  while (icb >= KERNEL_ONE_REG) {                                              \
    icb -= KERNEL_ONE_REG;                                                     \
    const float *ic_src = icb_src;                                             \
    const float *ic_flt = icb_flt;                                             \
    for (int64_t ic = 0; ic < KERNEL_ONE_REG; ++ic) {                          \
      WINOGRAD_B4F3_T14O32_KERNEL_AVX512_IC_COMPUTE(T_LEN, ic_src, ic_flt);    \
      ic_src += 1;                                                             \
      ic_flt += KERNEL_ONE_REG;                                                \
    }                                                                          \
    icb_flt += KERNEL_ONE_REG * KERNEL_ONE_REG;                                \
    icb_src += T_LEN * KERNEL_ONE_REG;                                         \
  }                                                                            \
  if (icb > 0) {                                                               \
    const float *ic_src = icb_src;                                             \
    const float *ic_flt = icb_flt;                                             \
    for (int64_t ic = 0; ic < icb; ++ic) {                                     \
      WINOGRAD_B4F3_T14O32_KERNEL_AVX512_IC_COMPUTE(T_LEN, ic_src, ic_flt);    \
      ic_src += 1;                                                             \
      ic_flt += KERNEL_ONE_REG;                                                \
    }                                                                          \
  }                                                                            \
  float *st_dst = T_DST;                                                       \
  if (T_LEN > 0)                                                               \
    _mm512_storeu_ps(st_dst + 0 * KERNEL_ONE_REG, zmm0);                       \
  if (T_LEN > 1)                                                               \
    _mm512_storeu_ps(st_dst + 1 * KERNEL_ONE_REG, zmm1);                       \
  if (T_LEN > 2)                                                               \
    _mm512_storeu_ps(st_dst + 2 * KERNEL_ONE_REG, zmm2);                       \
  if (T_LEN > 3)                                                               \
    _mm512_storeu_ps(st_dst + 3 * KERNEL_ONE_REG, zmm3);                       \
  if (T_LEN > 4)                                                               \
    _mm512_storeu_ps(st_dst + 4 * KERNEL_ONE_REG, zmm4);                       \
  if (T_LEN > 5)                                                               \
    _mm512_storeu_ps(st_dst + 5 * KERNEL_ONE_REG, zmm5);                       \
  if (T_LEN > 6)                                                               \
    _mm512_storeu_ps(st_dst + 6 * KERNEL_ONE_REG, zmm6);                       \
  if (T_LEN > 7)                                                               \
    _mm512_storeu_ps(st_dst + 7 * KERNEL_ONE_REG, zmm7);                       \
  if (T_LEN > 8)                                                               \
    _mm512_storeu_ps(st_dst + 8 * KERNEL_ONE_REG, zmm8);                       \
  if (T_LEN > 9)                                                               \
    _mm512_storeu_ps(st_dst + 9 * KERNEL_ONE_REG, zmm9);                       \
  if (T_LEN > 10)                                                              \
    _mm512_storeu_ps(st_dst + 10 * KERNEL_ONE_REG, zmm10);                     \
  if (T_LEN > 11)                                                              \
    _mm512_storeu_ps(st_dst + 11 * KERNEL_ONE_REG, zmm11);                     \
  if (T_LEN > 12)                                                              \
    _mm512_storeu_ps(st_dst + 12 * KERNEL_ONE_REG, zmm12);                     \
  if (T_LEN > 13)                                                              \
    _mm512_storeu_ps(st_dst + 13 * KERNEL_ONE_REG, zmm13);                     \
  st_dst += dst_ocb_stride;                                                    \
  if (T_LEN > 0)                                                               \
    _mm512_storeu_ps(st_dst + 0 * KERNEL_ONE_REG, zmm14);                      \
  if (T_LEN > 1)                                                               \
    _mm512_storeu_ps(st_dst + 1 * KERNEL_ONE_REG, zmm15);                      \
  if (T_LEN > 2)                                                               \
    _mm512_storeu_ps(st_dst + 2 * KERNEL_ONE_REG, zmm16);                      \
  if (T_LEN > 3)                                                               \
    _mm512_storeu_ps(st_dst + 3 * KERNEL_ONE_REG, zmm17);                      \
  if (T_LEN > 4)                                                               \
    _mm512_storeu_ps(st_dst + 4 * KERNEL_ONE_REG, zmm18);                      \
  if (T_LEN > 5)                                                               \
    _mm512_storeu_ps(st_dst + 5 * KERNEL_ONE_REG, zmm19);                      \
  if (T_LEN > 6)                                                               \
    _mm512_storeu_ps(st_dst + 6 * KERNEL_ONE_REG, zmm20);                      \
  if (T_LEN > 7)                                                               \
    _mm512_storeu_ps(st_dst + 7 * KERNEL_ONE_REG, zmm21);                      \
  if (T_LEN > 8)                                                               \
    _mm512_storeu_ps(st_dst + 8 * KERNEL_ONE_REG, zmm22);                      \
  if (T_LEN > 9)                                                               \
    _mm512_storeu_ps(st_dst + 9 * KERNEL_ONE_REG, zmm23);                      \
  if (T_LEN > 10)                                                              \
    _mm512_storeu_ps(st_dst + 10 * KERNEL_ONE_REG, zmm24);                     \
  if (T_LEN > 11)                                                              \
    _mm512_storeu_ps(st_dst + 11 * KERNEL_ONE_REG, zmm25);                     \
  if (T_LEN > 12)                                                              \
    _mm512_storeu_ps(st_dst + 12 * KERNEL_ONE_REG, zmm26);                     \
  if (T_LEN > 13)                                                              \
  _mm512_storeu_ps(st_dst + 13 * KERNEL_ONE_REG, zmm27)

#define WINOGRAD_B4F3_T14O32_KERNEL_AVX512(KERNEL_PARAMS, T_LEN)               \
  __m512 zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6, zmm7;                       \
  __m512 zmm8, zmm9, zmm10, zmm11, zmm12, zmm13, zmm14, zmm15;                 \
  __m512 zmm16, zmm17, zmm18, zmm19, zmm20, zmm21, zmm22, zmm23;               \
  __m512 zmm24, zmm25, zmm26, zmm27, zmm28, zmm29, zmm30, zmm31;               \
  const int64_t src_tkb_stride = KERNEL_PARAMS.src_tkb_stride;                \
  const int64_t flt_ocb_stride = KERNEL_PARAMS.flt_ocb_stride;                \
  const int64_t dst_ocb_stride = KERNEL_PARAMS.dst_ocb_stride;                \
  const int load_dst = KERNEL_PARAMS.load_dst;                            \
  const float *t_src = KERNEL_PARAMS.src;          \
  float *t_dst = KERNEL_PARAMS.dst;                      \
  int64_t t = KERNEL_PARAMS.tiles;                                            \
  WINO_DEBUG("init_t = %ld\n", t); \
  do {                                                                         \
    WINOGRAD_B4F3_T14O32_KERNEL_AVX512_ITER_T(KERNEL_PARAMS, T_LEN, t_src,     \
                                              t_dst);                          \
    t_src += src_tkb_stride;                                                   \
    t_dst += T_LEN * KERNEL_ONE_REG;                                           \
    t -= T_LEN;                                                                \
    WINO_DEBUG("t = %ld\n", t); \
  } while (t > 0)

#define WINOGRAD_B4F3_T31O16_KERNEL_AVX512_IC_COMPUTE(T_LEN, IC_SRC, IC_FLT)   \
  do {                                                                         \
    zmm31 = _mm512_loadu_ps(IC_FLT);                                           \
    if (T_LEN > 0)                                                             \
      zmm0 = _mm512_fmadd_ps(_mm512_set1_ps(IC_SRC[0 + 0 * KERNEL_ONE_REG]),   \
                             zmm31, zmm0);                                     \
    if (T_LEN > 1)                                                             \
      zmm1 = _mm512_fmadd_ps(_mm512_set1_ps(IC_SRC[0 + 1 * KERNEL_ONE_REG]),   \
                             zmm31, zmm1);                                     \
    if (T_LEN > 2)                                                             \
      zmm2 = _mm512_fmadd_ps(_mm512_set1_ps(IC_SRC[0 + 2 * KERNEL_ONE_REG]),   \
                             zmm31, zmm2);                                     \
    if (T_LEN > 3)                                                             \
      zmm3 = _mm512_fmadd_ps(_mm512_set1_ps(IC_SRC[0 + 3 * KERNEL_ONE_REG]),   \
                             zmm31, zmm3);                                     \
    if (T_LEN > 4)                                                             \
      zmm4 = _mm512_fmadd_ps(_mm512_set1_ps(IC_SRC[0 + 4 * KERNEL_ONE_REG]),   \
                             zmm31, zmm4);                                     \
    if (T_LEN > 5)                                                             \
      zmm5 = _mm512_fmadd_ps(_mm512_set1_ps(IC_SRC[0 + 5 * KERNEL_ONE_REG]),   \
                             zmm31, zmm5);                                     \
    if (T_LEN > 6)                                                             \
      zmm6 = _mm512_fmadd_ps(_mm512_set1_ps(IC_SRC[0 + 6 * KERNEL_ONE_REG]),   \
                             zmm31, zmm6);                                     \
    if (T_LEN > 7)                                                             \
      zmm7 = _mm512_fmadd_ps(_mm512_set1_ps(IC_SRC[0 + 7 * KERNEL_ONE_REG]),   \
                             zmm31, zmm7);                                     \
    if (T_LEN > 8)                                                             \
      zmm8 = _mm512_fmadd_ps(_mm512_set1_ps(IC_SRC[0 + 8 * KERNEL_ONE_REG]),   \
                             zmm31, zmm8);                                     \
    if (T_LEN > 9)                                                             \
      zmm9 = _mm512_fmadd_ps(_mm512_set1_ps(IC_SRC[0 + 9 * KERNEL_ONE_REG]),   \
                             zmm31, zmm9);                                     \
    if (T_LEN > 10)                                                            \
      zmm10 = _mm512_fmadd_ps(_mm512_set1_ps(IC_SRC[0 + 10 * KERNEL_ONE_REG]), \
                              zmm31, zmm10);                                   \
    if (T_LEN > 11)                                                            \
      zmm11 = _mm512_fmadd_ps(_mm512_set1_ps(IC_SRC[0 + 11 * KERNEL_ONE_REG]), \
                              zmm31, zmm11);                                   \
    if (T_LEN > 12)                                                            \
      zmm12 = _mm512_fmadd_ps(_mm512_set1_ps(IC_SRC[0 + 12 * KERNEL_ONE_REG]), \
                              zmm31, zmm12);                                   \
    if (T_LEN > 13)                                                            \
      zmm13 = _mm512_fmadd_ps(_mm512_set1_ps(IC_SRC[0 + 13 * KERNEL_ONE_REG]), \
                              zmm31, zmm13);                                   \
    if (T_LEN > 14)                                                            \
      zmm14 = _mm512_fmadd_ps(_mm512_set1_ps(IC_SRC[0 + 14 * KERNEL_ONE_REG]), \
                              zmm31, zmm14);                                   \
    if (T_LEN > 15)                                                            \
      zmm15 = _mm512_fmadd_ps(_mm512_set1_ps(IC_SRC[0 + 15 * KERNEL_ONE_REG]), \
                              zmm31, zmm15);                                   \
    if (T_LEN > 16)                                                            \
      zmm16 = _mm512_fmadd_ps(_mm512_set1_ps(IC_SRC[0 + 16 * KERNEL_ONE_REG]), \
                              zmm31, zmm16);                                   \
    if (T_LEN > 17)                                                            \
      zmm17 = _mm512_fmadd_ps(_mm512_set1_ps(IC_SRC[0 + 17 * KERNEL_ONE_REG]), \
                              zmm31, zmm17);                                   \
    if (T_LEN > 18)                                                            \
      zmm18 = _mm512_fmadd_ps(_mm512_set1_ps(IC_SRC[0 + 18 * KERNEL_ONE_REG]), \
                              zmm31, zmm18);                                   \
    if (T_LEN > 19)                                                            \
      zmm19 = _mm512_fmadd_ps(_mm512_set1_ps(IC_SRC[0 + 19 * KERNEL_ONE_REG]), \
                              zmm31, zmm19);                                   \
    if (T_LEN > 20)                                                            \
      zmm20 = _mm512_fmadd_ps(_mm512_set1_ps(IC_SRC[0 + 20 * KERNEL_ONE_REG]), \
                              zmm31, zmm20);                                   \
    if (T_LEN > 21)                                                            \
      zmm21 = _mm512_fmadd_ps(_mm512_set1_ps(IC_SRC[0 + 21 * KERNEL_ONE_REG]), \
                              zmm31, zmm21);                                   \
    if (T_LEN > 22)                                                            \
      zmm22 = _mm512_fmadd_ps(_mm512_set1_ps(IC_SRC[0 + 22 * KERNEL_ONE_REG]), \
                              zmm31, zmm22);                                   \
    if (T_LEN > 23)                                                            \
      zmm23 = _mm512_fmadd_ps(_mm512_set1_ps(IC_SRC[0 + 23 * KERNEL_ONE_REG]), \
                              zmm31, zmm23);                                   \
    if (T_LEN > 24)                                                            \
      zmm24 = _mm512_fmadd_ps(_mm512_set1_ps(IC_SRC[0 + 24 * KERNEL_ONE_REG]), \
                              zmm31, zmm24);                                   \
    if (T_LEN > 25)                                                            \
      zmm25 = _mm512_fmadd_ps(_mm512_set1_ps(IC_SRC[0 + 25 * KERNEL_ONE_REG]), \
                              zmm31, zmm25);                                   \
    if (T_LEN > 26)                                                            \
      zmm26 = _mm512_fmadd_ps(_mm512_set1_ps(IC_SRC[0 + 26 * KERNEL_ONE_REG]), \
                              zmm31, zmm26);                                   \
    if (T_LEN > 27)                                                            \
      zmm27 = _mm512_fmadd_ps(_mm512_set1_ps(IC_SRC[0 + 27 * KERNEL_ONE_REG]), \
                              zmm31, zmm27);                                   \
    if (T_LEN > 28)                                                            \
      zmm28 = _mm512_fmadd_ps(_mm512_set1_ps(IC_SRC[0 + 28 * KERNEL_ONE_REG]), \
                              zmm31, zmm28);                                   \
    if (T_LEN > 29)                                                            \
      zmm29 = _mm512_fmadd_ps(_mm512_set1_ps(IC_SRC[0 + 29 * KERNEL_ONE_REG]), \
                              zmm31, zmm29);                                   \
    if (T_LEN > 30)                                                            \
      zmm30 = _mm512_fmadd_ps(_mm512_set1_ps(IC_SRC[0 + 30 * KERNEL_ONE_REG]), \
                              zmm31, zmm30);                                   \
  } while (0)

#define WINOGRAD_B4F3_T31O16_KERNEL_AVX512_ITER_T(KERNEL_PARAMS, T_LEN, T_SRC, \
                                                  T_DST)                       \
  if (load_dst) {                                                              \
    const float *l_dst = T_DST;                                                \
    if (T_LEN > 0)                                                             \
      zmm0 = _mm512_loadu_ps(l_dst + 0 * KERNEL_ONE_REG);                      \
    if (T_LEN > 1)                                                             \
      zmm1 = _mm512_loadu_ps(l_dst + 1 * KERNEL_ONE_REG);                      \
    if (T_LEN > 2)                                                             \
      zmm2 = _mm512_loadu_ps(l_dst + 2 * KERNEL_ONE_REG);                      \
    if (T_LEN > 3)                                                             \
      zmm3 = _mm512_loadu_ps(l_dst + 3 * KERNEL_ONE_REG);                      \
    if (T_LEN > 4)                                                             \
      zmm4 = _mm512_loadu_ps(l_dst + 4 * KERNEL_ONE_REG);                      \
    if (T_LEN > 5)                                                             \
      zmm5 = _mm512_loadu_ps(l_dst + 5 * KERNEL_ONE_REG);                      \
    if (T_LEN > 6)                                                             \
      zmm6 = _mm512_loadu_ps(l_dst + 6 * KERNEL_ONE_REG);                      \
    if (T_LEN > 7)                                                             \
      zmm7 = _mm512_loadu_ps(l_dst + 7 * KERNEL_ONE_REG);                      \
    if (T_LEN > 8)                                                             \
      zmm8 = _mm512_loadu_ps(l_dst + 8 * KERNEL_ONE_REG);                      \
    if (T_LEN > 9)                                                             \
      zmm9 = _mm512_loadu_ps(l_dst + 9 * KERNEL_ONE_REG);                      \
    if (T_LEN > 10)                                                            \
      zmm10 = _mm512_loadu_ps(l_dst + 10 * KERNEL_ONE_REG);                    \
    if (T_LEN > 11)                                                            \
      zmm11 = _mm512_loadu_ps(l_dst + 11 * KERNEL_ONE_REG);                    \
    if (T_LEN > 12)                                                            \
      zmm12 = _mm512_loadu_ps(l_dst + 12 * KERNEL_ONE_REG);                    \
    if (T_LEN > 13)                                                            \
      zmm13 = _mm512_loadu_ps(l_dst + 13 * KERNEL_ONE_REG);                    \
    if (T_LEN > 14)                                                            \
      zmm14 = _mm512_loadu_ps(l_dst + 14 * KERNEL_ONE_REG);                    \
    if (T_LEN > 15)                                                            \
      zmm15 = _mm512_loadu_ps(l_dst + 15 * KERNEL_ONE_REG);                    \
    if (T_LEN > 16)                                                            \
      zmm16 = _mm512_loadu_ps(l_dst + 16 * KERNEL_ONE_REG);                    \
    if (T_LEN > 17)                                                            \
      zmm17 = _mm512_loadu_ps(l_dst + 17 * KERNEL_ONE_REG);                    \
    if (T_LEN > 18)                                                            \
      zmm18 = _mm512_loadu_ps(l_dst + 18 * KERNEL_ONE_REG);                    \
    if (T_LEN > 19)                                                            \
      zmm19 = _mm512_loadu_ps(l_dst + 19 * KERNEL_ONE_REG);                    \
    if (T_LEN > 20)                                                            \
      zmm20 = _mm512_loadu_ps(l_dst + 20 * KERNEL_ONE_REG);                    \
    if (T_LEN > 21)                                                            \
      zmm21 = _mm512_loadu_ps(l_dst + 21 * KERNEL_ONE_REG);                    \
    if (T_LEN > 22)                                                            \
      zmm22 = _mm512_loadu_ps(l_dst + 22 * KERNEL_ONE_REG);                    \
    if (T_LEN > 23)                                                            \
      zmm23 = _mm512_loadu_ps(l_dst + 23 * KERNEL_ONE_REG);                    \
    if (T_LEN > 24)                                                            \
      zmm24 = _mm512_loadu_ps(l_dst + 24 * KERNEL_ONE_REG);                    \
    if (T_LEN > 25)                                                            \
      zmm25 = _mm512_loadu_ps(l_dst + 25 * KERNEL_ONE_REG);                    \
    if (T_LEN > 26)                                                            \
      zmm26 = _mm512_loadu_ps(l_dst + 26 * KERNEL_ONE_REG);                    \
    if (T_LEN > 27)                                                            \
      zmm27 = _mm512_loadu_ps(l_dst + 27 * KERNEL_ONE_REG);                    \
    if (T_LEN > 28)                                                            \
      zmm28 = _mm512_loadu_ps(l_dst + 28 * KERNEL_ONE_REG);                    \
    if (T_LEN > 29)                                                            \
      zmm29 = _mm512_loadu_ps(l_dst + 29 * KERNEL_ONE_REG);                    \
    if (T_LEN > 30)                                                            \
      zmm30 = _mm512_loadu_ps(l_dst + 30 * KERNEL_ONE_REG);                    \
  } else {                                                                     \
    if (T_LEN > 0)                                                             \
      zmm0 = _mm512_setzero_ps();                                              \
    if (T_LEN > 1)                                                             \
      zmm1 = _mm512_setzero_ps();                                              \
    if (T_LEN > 2)                                                             \
      zmm2 = _mm512_setzero_ps();                                              \
    if (T_LEN > 3)                                                             \
      zmm3 = _mm512_setzero_ps();                                              \
    if (T_LEN > 4)                                                             \
      zmm4 = _mm512_setzero_ps();                                              \
    if (T_LEN > 5)                                                             \
      zmm5 = _mm512_setzero_ps();                                              \
    if (T_LEN > 6)                                                             \
      zmm6 = _mm512_setzero_ps();                                              \
    if (T_LEN > 7)                                                             \
      zmm7 = _mm512_setzero_ps();                                              \
    if (T_LEN > 8)                                                             \
      zmm8 = _mm512_setzero_ps();                                              \
    if (T_LEN > 9)                                                             \
      zmm9 = _mm512_setzero_ps();                                              \
    if (T_LEN > 10)                                                            \
      zmm10 = _mm512_setzero_ps();                                             \
    if (T_LEN > 11)                                                            \
      zmm11 = _mm512_setzero_ps();                                             \
    if (T_LEN > 12)                                                            \
      zmm12 = _mm512_setzero_ps();                                             \
    if (T_LEN > 13)                                                            \
      zmm13 = _mm512_setzero_ps();                                             \
    if (T_LEN > 14)                                                            \
      zmm14 = _mm512_setzero_ps();                                             \
    if (T_LEN > 15)                                                            \
      zmm15 = _mm512_setzero_ps();                                             \
    if (T_LEN > 16)                                                            \
      zmm16 = _mm512_setzero_ps();                                             \
    if (T_LEN > 17)                                                            \
      zmm17 = _mm512_setzero_ps();                                             \
    if (T_LEN > 18)                                                            \
      zmm18 = _mm512_setzero_ps();                                             \
    if (T_LEN > 19)                                                            \
      zmm19 = _mm512_setzero_ps();                                             \
    if (T_LEN > 20)                                                            \
      zmm20 = _mm512_setzero_ps();                                             \
    if (T_LEN > 21)                                                            \
      zmm21 = _mm512_setzero_ps();                                             \
    if (T_LEN > 22)                                                            \
      zmm22 = _mm512_setzero_ps();                                             \
    if (T_LEN > 23)                                                            \
      zmm23 = _mm512_setzero_ps();                                             \
    if (T_LEN > 24)                                                            \
      zmm24 = _mm512_setzero_ps();                                             \
    if (T_LEN > 25)                                                            \
      zmm25 = _mm512_setzero_ps();                                             \
    if (T_LEN > 26)                                                            \
      zmm26 = _mm512_setzero_ps();                                             \
    if (T_LEN > 27)                                                            \
      zmm27 = _mm512_setzero_ps();                                             \
    if (T_LEN > 28)                                                            \
      zmm28 = _mm512_setzero_ps();                                             \
    if (T_LEN > 29)                                                            \
      zmm29 = _mm512_setzero_ps();                                             \
    if (T_LEN > 30)                                                            \
      zmm30 = _mm512_setzero_ps();                                             \
  }                                                                            \
  const float *icb_src = T_SRC;                                                \
  const float *icb_flt = KERNEL_PARAMS.flt;        \
  int64_t icb = KERNEL_PARAMS.channels;                                       \
  while (icb >= KERNEL_ONE_REG) {                                              \
    icb -= KERNEL_ONE_REG;                                                     \
    const float *ic_src = icb_src;                                             \
    const float *ic_flt = icb_flt;                                             \
    for (int64_t ic = 0; ic < KERNEL_ONE_REG; ++ic) {                          \
      WINOGRAD_B4F3_T31O16_KERNEL_AVX512_IC_COMPUTE(T_LEN, ic_src, ic_flt);    \
      ic_src += 1;                                                             \
      ic_flt += KERNEL_ONE_REG;                                                \
    }                                                                          \
    icb_flt += KERNEL_ONE_REG * KERNEL_ONE_REG;                                \
    icb_src += T_LEN * KERNEL_ONE_REG;                                         \
  }                                                                            \
  if (icb > 0) {                                                               \
    const float *ic_src = icb_src;                                             \
    const float *ic_flt = icb_flt;                                             \
    for (int64_t ic = 0; ic < icb; ++ic) {                                     \
      WINOGRAD_B4F3_T31O16_KERNEL_AVX512_IC_COMPUTE(T_LEN, ic_src, ic_flt);    \
      ic_src += 1;                                                             \
      ic_flt += KERNEL_ONE_REG;                                                \
    }                                                                          \
  }                                                                            \
  float *st_dst = T_DST;                                                       \
  if (T_LEN > 0)                                                               \
    _mm512_storeu_ps(st_dst + 0 * KERNEL_ONE_REG, zmm0);                       \
  if (T_LEN > 1)                                                               \
    _mm512_storeu_ps(st_dst + 1 * KERNEL_ONE_REG, zmm1);                       \
  if (T_LEN > 2)                                                               \
    _mm512_storeu_ps(st_dst + 2 * KERNEL_ONE_REG, zmm2);                       \
  if (T_LEN > 3)                                                               \
    _mm512_storeu_ps(st_dst + 3 * KERNEL_ONE_REG, zmm3);                       \
  if (T_LEN > 4)                                                               \
    _mm512_storeu_ps(st_dst + 4 * KERNEL_ONE_REG, zmm4);                       \
  if (T_LEN > 5)                                                               \
    _mm512_storeu_ps(st_dst + 5 * KERNEL_ONE_REG, zmm5);                       \
  if (T_LEN > 6)                                                               \
    _mm512_storeu_ps(st_dst + 6 * KERNEL_ONE_REG, zmm6);                       \
  if (T_LEN > 7)                                                               \
    _mm512_storeu_ps(st_dst + 7 * KERNEL_ONE_REG, zmm7);                       \
  if (T_LEN > 8)                                                               \
    _mm512_storeu_ps(st_dst + 8 * KERNEL_ONE_REG, zmm8);                       \
  if (T_LEN > 9)                                                               \
    _mm512_storeu_ps(st_dst + 9 * KERNEL_ONE_REG, zmm9);                       \
  if (T_LEN > 10)                                                              \
    _mm512_storeu_ps(st_dst + 10 * KERNEL_ONE_REG, zmm10);                     \
  if (T_LEN > 11)                                                              \
    _mm512_storeu_ps(st_dst + 11 * KERNEL_ONE_REG, zmm11);                     \
  if (T_LEN > 12)                                                              \
    _mm512_storeu_ps(st_dst + 12 * KERNEL_ONE_REG, zmm12);                     \
  if (T_LEN > 13)                                                              \
    _mm512_storeu_ps(st_dst + 13 * KERNEL_ONE_REG, zmm13);                     \
  if (T_LEN > 14)                                                              \
    _mm512_storeu_ps(st_dst + 14 * KERNEL_ONE_REG, zmm14);                     \
  if (T_LEN > 15)                                                              \
    _mm512_storeu_ps(st_dst + 15 * KERNEL_ONE_REG, zmm15);                     \
  if (T_LEN > 16)                                                              \
    _mm512_storeu_ps(st_dst + 16 * KERNEL_ONE_REG, zmm16);                     \
  if (T_LEN > 17)                                                              \
    _mm512_storeu_ps(st_dst + 17 * KERNEL_ONE_REG, zmm17);                     \
  if (T_LEN > 18)                                                              \
    _mm512_storeu_ps(st_dst + 18 * KERNEL_ONE_REG, zmm18);                     \
  if (T_LEN > 19)                                                              \
    _mm512_storeu_ps(st_dst + 19 * KERNEL_ONE_REG, zmm19);                     \
  if (T_LEN > 20)                                                              \
    _mm512_storeu_ps(st_dst + 20 * KERNEL_ONE_REG, zmm20);                     \
  if (T_LEN > 21)                                                              \
    _mm512_storeu_ps(st_dst + 21 * KERNEL_ONE_REG, zmm21);                     \
  if (T_LEN > 22)                                                              \
    _mm512_storeu_ps(st_dst + 22 * KERNEL_ONE_REG, zmm22);                     \
  if (T_LEN > 23)                                                              \
    _mm512_storeu_ps(st_dst + 23 * KERNEL_ONE_REG, zmm23);                     \
  if (T_LEN > 24)                                                              \
    _mm512_storeu_ps(st_dst + 24 * KERNEL_ONE_REG, zmm24);                     \
  if (T_LEN > 25)                                                              \
    _mm512_storeu_ps(st_dst + 25 * KERNEL_ONE_REG, zmm25);                     \
  if (T_LEN > 26)                                                              \
    _mm512_storeu_ps(st_dst + 26 * KERNEL_ONE_REG, zmm26);                     \
  if (T_LEN > 27)                                                              \
    _mm512_storeu_ps(st_dst + 27 * KERNEL_ONE_REG, zmm27);                     \
  if (T_LEN > 28)                                                              \
    _mm512_storeu_ps(st_dst + 28 * KERNEL_ONE_REG, zmm28);                     \
  if (T_LEN > 29)                                                              \
    _mm512_storeu_ps(st_dst + 29 * KERNEL_ONE_REG, zmm29);                     \
  if (T_LEN > 30)                                                              \
  _mm512_storeu_ps(st_dst + 30 * KERNEL_ONE_REG, zmm30)

#define WINOGRAD_B4F3_T31O16_KERNEL_AVX512(KERNEL_PARAMS, T_LEN)               \
  __m512 zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6, zmm7;                       \
  __m512 zmm8, zmm9, zmm10, zmm11, zmm12, zmm13, zmm14, zmm15;                 \
  __m512 zmm16, zmm17, zmm18, zmm19, zmm20, zmm21, zmm22, zmm23;               \
  __m512 zmm24, zmm25, zmm26, zmm27, zmm28, zmm29, zmm30, zmm31;               \
  const int64_t src_tkb_stride = KERNEL_PARAMS.src_tkb_stride;                \
  const int load_dst = KERNEL_PARAMS.load_dst;                            \
  const float *t_src = KERNEL_PARAMS.src;          \
  float *t_dst = KERNEL_PARAMS.dst;                      \
  int64_t t = KERNEL_PARAMS.tiles;                                            \
  WINO_DEBUG("init_t = %ld\n", t); \
  do {                                                                         \
    WINOGRAD_B4F3_T31O16_KERNEL_AVX512_ITER_T(KERNEL_PARAMS, T_LEN, t_src,     \
                                              t_dst);                          \
    t_src += src_tkb_stride;                                                   \
    t_dst += T_LEN * KERNEL_ONE_REG;                                           \
    t -= T_LEN;                                                                \
    WINO_DEBUG("t = %ld\n", t); \
  } while (t > 0)

void winograd_b4f3_srctrans_fp32_avx512(const float *base_src, const int64_t ih,
                                        const int64_t iw, const int64_t src_h,
                                        const int64_t src_w,
                                        const int64_t src_trans_ti_stride,
                                        float *tile_buffer,
                                        float *matmul_buffer, float *src_trans);

void winograd_b4f3_gemm_kernel_fp32_avx512(int64_t oc_len, int64_t t_len,
                                           wgb4f3_kernel_params params);

void winograd_b4f3_dsttrans_fp32_avx512(const float *dst_trans,
                                        const int64_t dst_trans_ti_stride,
                                        const int64_t dst_h_stride,
                                        float *matmul_buffer, float *dst);

void winograd_b4f3_store_dst_fp32_avx512(const float *dst_trans,
                                         const int64_t oh_len,
                                         const int64_t ow_len,
                                         const int64_t dst_h_stride,
                                         float *dst);

#endif