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
#ifndef TILE_L2_BLK_MAX_SMALL
#define TILE_L2_BLK_MAX_SMALL (2 * KERNEL_M14_TILE)
#endif
#define TILE_L2_BLK_MAX_LARGE (8 * KERNEL_M14_TILE)
#define OC_L2_BLK_MAX (16 * OC_KERNEL_BLK)

// KERNEL PARAMETERS
typedef struct {
  const float *src;
  float *dst;
  const float *flt;
  int64_t tiles;
  int64_t channels;
  int64_t src_tkb_stride;
  int64_t dst_ocb_stride;
  int64_t flt_ocb_stride;
  int64_t load_dst;
} wgb4f3_kernel_params;

typedef void (*winograd_b4f3_gemm_kernel_fp32_avx512_func_t)(const int64_t *);

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
  const float *icb_flt = (const float *)KERNEL_PARAMS[2];                      \
  int64_t icb = KERNEL_PARAMS[4];                                              \
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

#define WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T14O32_FUNC(TLEN)                    \
  winograd_b4f3_gemm_kernel_fp32_avx512_o32_t##TLEN

#ifdef USE_INLINE_ASM
#define WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T14O32_ASM(TLEN)                     \
  winograd_b4f3_gemm_kernel_fp32_avx512_asm_o32_t##TLEN

#define DECLARE_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T14O32_ASM(TLEN)             \
  void winograd_b4f3_gemm_kernel_fp32_avx512_asm_o32_t##TLEN(                  \
      const int64_t *param)

DECLARE_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T14O32_ASM(4);
DECLARE_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T14O32_ASM(5);
DECLARE_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T14O32_ASM(6);
DECLARE_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T14O32_ASM(7);
DECLARE_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T14O32_ASM(8);
DECLARE_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T14O32_ASM(9);
DECLARE_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T14O32_ASM(10);
DECLARE_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T14O32_ASM(11);
DECLARE_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T14O32_ASM(12);
DECLARE_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T14O32_ASM(13);
DECLARE_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T14O32_ASM(14);

#define IMPLEMENT_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T14O32_ASM(TLEN)           \
  void winograd_b4f3_gemm_kernel_fp32_avx512_asm_o32_t##TLEN(                  \
      const int64_t *param) {                                                  \
    __asm__ __volatile__(                                                      \
        ".equ P_BYTES, 8\n"                                                    \
        ".equ D_BYTES, 4\n"                                                    \
        ".equ CH_DT_BLK, 16\n"                                                 \
        ".equ SRC_IDX,      (0 * P_BYTES)\n"                                   \
        ".equ DST_IDX,      (1 * P_BYTES)\n"                                   \
        ".equ FLT_IDX,      (2 * P_BYTES)\n"                                   \
        ".equ TILES_IDX,    (3 * P_BYTES)\n"                                   \
        ".equ CHANNELS_IDX, (4 * P_BYTES)\n"                                   \
        ".equ SRC_TKB_STRIDE_IDX, (5 * P_BYTES)\n"                             \
        ".equ DST_OCB_STRIDE_IDX, (6 * P_BYTES)\n"                             \
        ".equ FLT_OCB_STRIDE_IDX, (7 * P_BYTES)\n"                             \
        ".equ LOAD_DST_IDX,       (8 * P_BYTES)\n"                             \
        ".equ T_LEN, %c[T_LEN]\n"                                              \
        "mov SRC_TKB_STRIDE_IDX(%[param]), %%r8\n"                             \
        "mov FLT_OCB_STRIDE_IDX(%[param]), %%r9\n"                             \
        "mov DST_OCB_STRIDE_IDX(%[param]), %%r11\n"                            \
        "mov LOAD_DST_IDX(%[param]), %%r12\n"                                  \
        "mov DST_IDX(%[param]), %%r13\n"                                       \
        "mov SRC_IDX(%[param]), %%r14\n"                                       \
        "mov TILES_IDX(%[param]), %%r15\n"                                     \
        "1:\n" /* label_init_session */                                        \
        "test %%r12, %%r12\n"                                                  \
        "jnz 2f\n" /* label_load_dst */                                        \
        ".if T_LEN > 0\n vpxord %%zmm0, %%zmm0, %%zmm0\n .endif\n"             \
        ".if T_LEN > 1\n vpxord %%zmm1, %%zmm1, %%zmm1\n .endif\n"             \
        ".if T_LEN > 2\n vpxord %%zmm2, %%zmm2, %%zmm2\n .endif\n"             \
        ".if T_LEN > 3\n vpxord %%zmm3, %%zmm3, %%zmm3\n .endif\n"             \
        ".if T_LEN > 4\n vpxord %%zmm4, %%zmm4, %%zmm4\n .endif\n"             \
        ".if T_LEN > 5\n vpxord %%zmm5, %%zmm5, %%zmm5\n .endif\n"             \
        ".if T_LEN > 6\n vpxord %%zmm6, %%zmm6, %%zmm6\n .endif\n"             \
        ".if T_LEN > 7\n vpxord %%zmm7, %%zmm7, %%zmm7\n .endif\n"             \
        ".if T_LEN > 8\n vpxord %%zmm8, %%zmm8, %%zmm8\n .endif\n"             \
        ".if T_LEN > 9\n vpxord %%zmm9, %%zmm9, %%zmm9\n .endif\n"             \
        ".if T_LEN > 10\n vpxord %%zmm10, %%zmm10, %%zmm10\n .endif\n"         \
        ".if T_LEN > 11\n vpxord %%zmm11, %%zmm11, %%zmm11\n .endif\n"         \
        ".if T_LEN > 12\n vpxord %%zmm12, %%zmm12, %%zmm12\n .endif\n"         \
        ".if T_LEN > 13\n vpxord %%zmm13, %%zmm13, %%zmm13\n .endif\n"         \
        ".if T_LEN > 0\n vpxord %%zmm14, %%zmm14, %%zmm14\n .endif\n"          \
        ".if T_LEN > 1\n vpxord %%zmm15, %%zmm15, %%zmm15\n .endif\n"          \
        ".if T_LEN > 2\n vpxord %%zmm16, %%zmm16, %%zmm16\n .endif\n"          \
        ".if T_LEN > 3\n vpxord %%zmm17, %%zmm17, %%zmm17\n .endif\n"          \
        ".if T_LEN > 4\n vpxord %%zmm18, %%zmm18, %%zmm18\n .endif\n"          \
        ".if T_LEN > 5\n vpxord %%zmm19, %%zmm19, %%zmm19\n .endif\n"          \
        ".if T_LEN > 6\n vpxord %%zmm20, %%zmm20, %%zmm20\n .endif\n"          \
        ".if T_LEN > 7\n vpxord %%zmm21, %%zmm21, %%zmm21\n .endif\n"          \
        ".if T_LEN > 8\n vpxord %%zmm22, %%zmm22, %%zmm22\n .endif\n"          \
        ".if T_LEN > 9\n vpxord %%zmm23, %%zmm23, %%zmm23\n .endif\n"          \
        ".if T_LEN > 10\n vpxord %%zmm24, %%zmm24, %%zmm24\n .endif\n"         \
        ".if T_LEN > 11\n vpxord %%zmm25, %%zmm25, %%zmm25\n .endif\n"         \
        ".if T_LEN > 12\n vpxord %%zmm26, %%zmm26, %%zmm26\n .endif\n"         \
        ".if T_LEN > 13\n vpxord %%zmm27, %%zmm27, %%zmm27\n .endif\n"         \
        "jmp 3f\n" /* label_compute_session */                                 \
        "2:\n"     /* label_load_dst */                                        \
        "lea (%%r13, %%r11, D_BYTES), %%r10\n"                                 \
        ".if T_LEN > 0\n vmovups (0 * CH_DT_BLK * D_BYTES)(%%r13), %%zmm0\n "  \
        ".endif\n"                                                             \
        ".if T_LEN > 1\n vmovups (1 * CH_DT_BLK * D_BYTES)(%%r13), %%zmm1\n "  \
        ".endif\n"                                                             \
        ".if T_LEN > 2\n vmovups (2 * CH_DT_BLK * D_BYTES)(%%r13), %%zmm2\n "  \
        ".endif\n"                                                             \
        ".if T_LEN > 3\n vmovups (3 * CH_DT_BLK * D_BYTES)(%%r13), %%zmm3\n "  \
        ".endif\n"                                                             \
        ".if T_LEN > 4\n vmovups (4 * CH_DT_BLK * D_BYTES)(%%r13), %%zmm4\n "  \
        ".endif\n"                                                             \
        ".if T_LEN > 5\n vmovups (5 * CH_DT_BLK * D_BYTES)(%%r13), %%zmm5\n "  \
        ".endif\n"                                                             \
        ".if T_LEN > 6\n vmovups (6 * CH_DT_BLK * D_BYTES)(%%r13), %%zmm6\n "  \
        ".endif\n"                                                             \
        ".if T_LEN > 7\n vmovups (7 * CH_DT_BLK * D_BYTES)(%%r13), %%zmm7\n "  \
        ".endif\n"                                                             \
        ".if T_LEN > 8\n vmovups (8 * CH_DT_BLK * D_BYTES)(%%r13), %%zmm8\n "  \
        ".endif\n"                                                             \
        ".if T_LEN > 9\n vmovups (9 * CH_DT_BLK * D_BYTES)(%%r13), %%zmm9\n "  \
        ".endif\n"                                                             \
        ".if T_LEN > 10\n vmovups (10 * CH_DT_BLK * D_BYTES)(%%r13), "         \
        "%%zmm10\n .endif\n"                                                   \
        ".if T_LEN > 11\n vmovups (11 * CH_DT_BLK * D_BYTES)(%%r13), "         \
        "%%zmm11\n .endif\n"                                                   \
        ".if T_LEN > 12\n vmovups (12 * CH_DT_BLK * D_BYTES)(%%r13), "         \
        "%%zmm12\n .endif\n"                                                   \
        ".if T_LEN > 13\n vmovups (13 * CH_DT_BLK * D_BYTES)(%%r13), "         \
        "%%zmm13\n .endif\n"                                                   \
        ".if T_LEN > 0\n vmovups (0 * CH_DT_BLK * D_BYTES)(%%r10), %%zmm14\n " \
        ".endif\n"                                                             \
        ".if T_LEN > 1\n vmovups (1 * CH_DT_BLK * D_BYTES)(%%r10), %%zmm15\n " \
        ".endif\n"                                                             \
        ".if T_LEN > 2\n vmovups (2 * CH_DT_BLK * D_BYTES)(%%r10), %%zmm16\n " \
        ".endif\n"                                                             \
        ".if T_LEN > 3\n vmovups (3 * CH_DT_BLK * D_BYTES)(%%r10), %%zmm17\n " \
        ".endif\n"                                                             \
        ".if T_LEN > 4\n vmovups (4 * CH_DT_BLK * D_BYTES)(%%r10), %%zmm18\n " \
        ".endif\n"                                                             \
        ".if T_LEN > 5\n vmovups (5 * CH_DT_BLK * D_BYTES)(%%r10), %%zmm19\n " \
        ".endif\n"                                                             \
        ".if T_LEN > 6\n vmovups (6 * CH_DT_BLK * D_BYTES)(%%r10), %%zmm20\n " \
        ".endif\n"                                                             \
        ".if T_LEN > 7\n vmovups (7 * CH_DT_BLK * D_BYTES)(%%r10), %%zmm21\n " \
        ".endif\n"                                                             \
        ".if T_LEN > 8\n vmovups (8 * CH_DT_BLK * D_BYTES)(%%r10), %%zmm22\n " \
        ".endif\n"                                                             \
        ".if T_LEN > 9\n vmovups (9 * CH_DT_BLK * D_BYTES)(%%r10), %%zmm23\n " \
        ".endif\n"                                                             \
        ".if T_LEN > 10\n vmovups (10 * CH_DT_BLK * D_BYTES)(%%r10), "         \
        "%%zmm24\n .endif\n"                                                   \
        ".if T_LEN > 11\n vmovups (11 * CH_DT_BLK * D_BYTES)(%%r10), "         \
        "%%zmm25\n .endif\n"                                                   \
        ".if T_LEN > 12\n vmovups (12 * CH_DT_BLK * D_BYTES)(%%r10), "         \
        "%%zmm26\n .endif\n"                                                   \
        ".if T_LEN > 13\n vmovups (13 * CH_DT_BLK * D_BYTES)(%%r10), "         \
        "%%zmm27\n .endif\n"                                                   \
        "3:\n" /* label_compute_session */                                     \
        "mov %%r14, %%rax\n"                                                   \
        "mov FLT_IDX(%[param]), %%rbx\n"                                       \
        "mov CHANNELS_IDX(%[param]), %%r10\n"                                  \
        "lea (%%rbx, %%r9, D_BYTES), %%rcx\n"                                  \
        "cmp $CH_DT_BLK, %%r10\n"                                              \
        "jl 5f\n" /* label_ic_remain */                                        \
        "4:\n"    /* label_ic_body */                                          \
        ".align 16\n"                                                          \
        ".if T_LEN < 6\n"                                                      \
        ".irp IC,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15\n"                      \
        "vmovups (\\IC * CH_DT_BLK * D_BYTES)(%%rbx), %%zmm28\n"               \
        "vmovups (\\IC * CH_DT_BLK * D_BYTES)(%%rcx), %%zmm29\n"               \
        "prefetcht0 ((\\IC * CH_DT_BLK + CH_DT_BLK * CH_DT_BLK) * "            \
        "D_BYTES)(%%rbx)\n"                                                    \
        "prefetcht0 ((\\IC * CH_DT_BLK + CH_DT_BLK * CH_DT_BLK) * "            \
        "D_BYTES)(%%rcx)\n"                                                    \
        ".if T_LEN > 0\n"                                                      \
        "vbroadcastss ((\\IC + 0 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm30\n"    \
        "vfmadd231ps %%zmm28, %%zmm30, %%zmm0\n"                               \
        "vfmadd231ps %%zmm29, %%zmm30, %%zmm14\n"                              \
        ".endif\n"                                                             \
        ".if T_LEN > 1\n"                                                      \
        "vbroadcastss ((\\IC + 1 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm31\n"    \
        "vfmadd231ps %%zmm28, %%zmm31, %%zmm1\n"                               \
        "vfmadd231ps %%zmm29, %%zmm31, %%zmm15\n"                              \
        ".endif\n"                                                             \
        ".if T_LEN > 2\n"                                                      \
        "vbroadcastss ((\\IC + 2 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm30\n"    \
        "vfmadd231ps %%zmm28, %%zmm30, %%zmm2\n"                               \
        "vfmadd231ps %%zmm29, %%zmm30, %%zmm16\n"                              \
        ".endif\n"                                                             \
        ".if T_LEN > 3\n"                                                      \
        "vbroadcastss ((\\IC + 3 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm31\n"    \
        "vfmadd231ps %%zmm28, %%zmm31, %%zmm3\n"                               \
        "vfmadd231ps %%zmm29, %%zmm31, %%zmm17\n"                              \
        ".endif\n"                                                             \
        ".if T_LEN > 4\n"                                                      \
        "vbroadcastss ((\\IC + 4 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm30\n"    \
        "vfmadd231ps %%zmm28, %%zmm30, %%zmm4\n"                               \
        "vfmadd231ps %%zmm29, %%zmm30, %%zmm18\n"                              \
        ".endif\n"                                                             \
        ".if T_LEN > 5\n"                                                      \
        "vbroadcastss ((\\IC + 5 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm31\n"    \
        "vfmadd231ps %%zmm28, %%zmm31, %%zmm5\n"                               \
        "vfmadd231ps %%zmm29, %%zmm31, %%zmm19\n"                              \
        ".endif\n"                                                             \
        ".if T_LEN > 6\n"                                                      \
        "vbroadcastss ((\\IC + 6 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm30\n"    \
        "vfmadd231ps %%zmm28, %%zmm30, %%zmm6\n"                               \
        "vfmadd231ps %%zmm29, %%zmm30, %%zmm20\n"                              \
        ".endif\n"                                                             \
        ".if T_LEN > 7\n"                                                      \
        "vbroadcastss ((\\IC + 7 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm31\n"    \
        "vfmadd231ps %%zmm28, %%zmm31, %%zmm7\n"                               \
        "vfmadd231ps %%zmm29, %%zmm31, %%zmm21\n"                              \
        ".endif\n"                                                             \
        ".if T_LEN > 8\n"                                                      \
        "vbroadcastss ((\\IC + 8 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm30\n"    \
        "vfmadd231ps %%zmm28, %%zmm30, %%zmm8\n"                               \
        "vfmadd231ps %%zmm29, %%zmm30, %%zmm22\n"                              \
        ".endif\n"                                                             \
        ".if T_LEN > 9\n"                                                      \
        "vbroadcastss ((\\IC + 9 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm31\n"    \
        "vfmadd231ps %%zmm28, %%zmm31, %%zmm9\n"                               \
        "vfmadd231ps %%zmm29, %%zmm31, %%zmm23\n"                              \
        ".endif\n"                                                             \
        ".if T_LEN > 10\n"                                                     \
        "vbroadcastss ((\\IC + 10 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm30\n"   \
        "vfmadd231ps %%zmm28, %%zmm30, %%zmm10\n"                              \
        "vfmadd231ps %%zmm29, %%zmm30, %%zmm24\n"                              \
        ".endif\n"                                                             \
        ".if T_LEN > 11\n"                                                     \
        "vbroadcastss ((\\IC + 11 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm31\n"   \
        "vfmadd231ps %%zmm28, %%zmm31, %%zmm11\n"                              \
        "vfmadd231ps %%zmm29, %%zmm31, %%zmm25\n"                              \
        ".endif\n"                                                             \
        ".if T_LEN > 12\n"                                                     \
        "vbroadcastss ((\\IC + 12 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm30\n"   \
        "vfmadd231ps %%zmm28, %%zmm30, %%zmm12\n"                              \
        "vfmadd231ps %%zmm29, %%zmm30, %%zmm26\n"                              \
        ".endif\n"                                                             \
        ".if T_LEN > 13\n"                                                     \
        "vbroadcastss ((\\IC + 13 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm31\n"   \
        "vfmadd231ps %%zmm28, %%zmm31, %%zmm13\n"                              \
        "vfmadd231ps %%zmm29, %%zmm31, %%zmm27\n"                              \
        ".endif\n"                                                             \
        ".endr\n"                                                              \
        "lea (CH_DT_BLK * CH_DT_BLK * D_BYTES)(%%rbx), %%rbx\n"                \
        "lea (CH_DT_BLK * CH_DT_BLK * D_BYTES)(%%rcx), %%rcx\n"                \
        "lea (T_LEN * CH_DT_BLK * D_BYTES)(%%rax), %%rax\n"                    \
        ".else\n" /* .if T_LEN < 6 */                                          \
        "mov $CH_DT_BLK, %%rsi\n"                                              \
        "9:\n" /* label_ic */                                                  \
        "vmovups (0 * CH_DT_BLK * D_BYTES)(%%rbx), %%zmm28\n"                  \
        "vmovups (0 * CH_DT_BLK * D_BYTES)(%%rcx), %%zmm29\n"                  \
        "prefetcht0 ((0 * CH_DT_BLK + CH_DT_BLK * CH_DT_BLK) * "               \
        "D_BYTES)(%%rbx)\n"                                                    \
        "prefetcht0 ((0 * CH_DT_BLK + CH_DT_BLK * CH_DT_BLK) * "               \
        "D_BYTES)(%%rcx)\n"                                                    \
        "lea (CH_DT_BLK * D_BYTES)(%%rbx), %%rbx\n"                            \
        "lea (CH_DT_BLK * D_BYTES)(%%rcx), %%rcx\n"                            \
        ".if T_LEN > 0\n"                                                      \
        "vbroadcastss ((0 + 0 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm30\n"       \
        "vfmadd231ps %%zmm28, %%zmm30, %%zmm0\n"                               \
        "vfmadd231ps %%zmm29, %%zmm30, %%zmm14\n"                              \
        ".endif\n"                                                             \
        ".if T_LEN > 1\n"                                                      \
        "vbroadcastss ((0 + 1 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm31\n"       \
        "vfmadd231ps %%zmm28, %%zmm31, %%zmm1\n"                               \
        "vfmadd231ps %%zmm29, %%zmm31, %%zmm15\n"                              \
        ".endif\n"                                                             \
        ".if T_LEN > 2\n"                                                      \
        "vbroadcastss ((0 + 2 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm30\n"       \
        "vfmadd231ps %%zmm28, %%zmm30, %%zmm2\n"                               \
        "vfmadd231ps %%zmm29, %%zmm30, %%zmm16\n"                              \
        ".endif\n"                                                             \
        ".if T_LEN > 3\n"                                                      \
        "vbroadcastss ((0 + 3 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm31\n"       \
        "vfmadd231ps %%zmm28, %%zmm31, %%zmm3\n"                               \
        "vfmadd231ps %%zmm29, %%zmm31, %%zmm17\n"                              \
        ".endif\n"                                                             \
        ".if T_LEN > 4\n"                                                      \
        "vbroadcastss ((0 + 4 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm30\n"       \
        "vfmadd231ps %%zmm28, %%zmm30, %%zmm4\n"                               \
        "vfmadd231ps %%zmm29, %%zmm30, %%zmm18\n"                              \
        ".endif\n"                                                             \
        ".if T_LEN > 5\n"                                                      \
        "vbroadcastss ((0 + 5 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm31\n"       \
        "vfmadd231ps %%zmm28, %%zmm31, %%zmm5\n"                               \
        "vfmadd231ps %%zmm29, %%zmm31, %%zmm19\n"                              \
        ".endif\n"                                                             \
        ".if T_LEN > 6\n"                                                      \
        "vbroadcastss ((0 + 6 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm30\n"       \
        "vfmadd231ps %%zmm28, %%zmm30, %%zmm6\n"                               \
        "vfmadd231ps %%zmm29, %%zmm30, %%zmm20\n"                              \
        ".endif\n"                                                             \
        ".if T_LEN > 7\n"                                                      \
        "vbroadcastss ((0 + 7 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm31\n"       \
        "vfmadd231ps %%zmm28, %%zmm31, %%zmm7\n"                               \
        "vfmadd231ps %%zmm29, %%zmm31, %%zmm21\n"                              \
        ".endif\n"                                                             \
        ".if T_LEN > 8\n"                                                      \
        "vbroadcastss ((0 + 8 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm30\n"       \
        "vfmadd231ps %%zmm28, %%zmm30, %%zmm8\n"                               \
        "vfmadd231ps %%zmm29, %%zmm30, %%zmm22\n"                              \
        ".endif\n"                                                             \
        ".if T_LEN > 9\n"                                                      \
        "vbroadcastss ((0 + 9 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm31\n"       \
        "vfmadd231ps %%zmm28, %%zmm31, %%zmm9\n"                               \
        "vfmadd231ps %%zmm29, %%zmm31, %%zmm23\n"                              \
        ".endif\n"                                                             \
        ".if T_LEN > 10\n"                                                     \
        "vbroadcastss ((0 + 10 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm30\n"      \
        "vfmadd231ps %%zmm28, %%zmm30, %%zmm10\n"                              \
        "vfmadd231ps %%zmm29, %%zmm30, %%zmm24\n"                              \
        ".endif\n"                                                             \
        ".if T_LEN > 11\n"                                                     \
        "vbroadcastss ((0 + 11 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm31\n"      \
        "vfmadd231ps %%zmm28, %%zmm31, %%zmm11\n"                              \
        "vfmadd231ps %%zmm29, %%zmm31, %%zmm25\n"                              \
        ".endif\n"                                                             \
        ".if T_LEN > 12\n"                                                     \
        "vbroadcastss ((0 + 12 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm30\n"      \
        "vfmadd231ps %%zmm28, %%zmm30, %%zmm12\n"                              \
        "vfmadd231ps %%zmm29, %%zmm30, %%zmm26\n"                              \
        ".endif\n"                                                             \
        ".if T_LEN > 13\n"                                                     \
        "vbroadcastss ((0 + 13 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm31\n"      \
        "vfmadd231ps %%zmm28, %%zmm31, %%zmm13\n"                              \
        "vfmadd231ps %%zmm29, %%zmm31, %%zmm27\n"                              \
        ".endif\n"                                                             \
        "lea D_BYTES(%%rax), %%rax\n"                                          \
        "sub $1, %%rsi\n"                                                      \
        "cmp $0, %%rsi\n"                                                      \
        "jne 9b\n" /* label_ic */                                              \
        "lea ((T_LEN - 1) * CH_DT_BLK * D_BYTES)(%%rax), %%rax\n"              \
        ".endif\n"                                                             \
        "sub $CH_DT_BLK, %%r10\n"                                              \
        "cmp $CH_DT_BLK, %%r10\n"                                              \
        "jge 4b\n" /* label_ic_body */                                         \
        "cmp $0, %%r10\n"                                                      \
        "je 6f\n" /* label_finalize_session */                                 \
        "5:\n"    /* label_ic_remain */                                        \
        "vmovups (0 * CH_DT_BLK * D_BYTES)(%%rbx), %%zmm28\n"                  \
        "vmovups (0 * CH_DT_BLK * D_BYTES)(%%rcx), %%zmm29\n"                  \
        ".if T_LEN > 0\n"                                                      \
        "vbroadcastss ((0 + 0 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm30\n"       \
        "vfmadd231ps %%zmm28, %%zmm30, %%zmm0\n"                               \
        "vfmadd231ps %%zmm29, %%zmm30, %%zmm14\n"                              \
        ".endif\n"                                                             \
        ".if T_LEN > 1\n"                                                      \
        "vbroadcastss ((0 + 1 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm31\n"       \
        "vfmadd231ps %%zmm28, %%zmm31, %%zmm1\n"                               \
        "vfmadd231ps %%zmm29, %%zmm31, %%zmm15\n"                              \
        ".endif\n"                                                             \
        ".if T_LEN > 2\n"                                                      \
        "vbroadcastss ((0 + 2 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm30\n"       \
        "vfmadd231ps %%zmm28, %%zmm30, %%zmm2\n"                               \
        "vfmadd231ps %%zmm29, %%zmm30, %%zmm16\n"                              \
        ".endif\n"                                                             \
        ".if T_LEN > 3\n"                                                      \
        "vbroadcastss ((0 + 3 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm31\n"       \
        "vfmadd231ps %%zmm28, %%zmm31, %%zmm3\n"                               \
        "vfmadd231ps %%zmm29, %%zmm31, %%zmm17\n"                              \
        ".endif\n"                                                             \
        ".if T_LEN > 4\n"                                                      \
        "vbroadcastss ((0 + 4 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm30\n"       \
        "vfmadd231ps %%zmm28, %%zmm30, %%zmm4\n"                               \
        "vfmadd231ps %%zmm29, %%zmm30, %%zmm18\n"                              \
        ".endif\n"                                                             \
        ".if T_LEN > 5\n"                                                      \
        "vbroadcastss ((0 + 5 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm31\n"       \
        "vfmadd231ps %%zmm28, %%zmm31, %%zmm5\n"                               \
        "vfmadd231ps %%zmm29, %%zmm31, %%zmm19\n"                              \
        ".endif\n"                                                             \
        ".if T_LEN > 6\n"                                                      \
        "vbroadcastss ((0 + 6 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm30\n"       \
        "vfmadd231ps %%zmm28, %%zmm30, %%zmm6\n"                               \
        "vfmadd231ps %%zmm29, %%zmm30, %%zmm20\n"                              \
        ".endif\n"                                                             \
        ".if T_LEN > 7\n"                                                      \
        "vbroadcastss ((0 + 7 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm31\n"       \
        "vfmadd231ps %%zmm28, %%zmm31, %%zmm7\n"                               \
        "vfmadd231ps %%zmm29, %%zmm31, %%zmm21\n"                              \
        ".endif\n"                                                             \
        ".if T_LEN > 8\n"                                                      \
        "vbroadcastss ((0 + 8 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm30\n"       \
        "vfmadd231ps %%zmm28, %%zmm30, %%zmm8\n"                               \
        "vfmadd231ps %%zmm29, %%zmm30, %%zmm22\n"                              \
        ".endif\n"                                                             \
        ".if T_LEN > 9\n"                                                      \
        "vbroadcastss ((0 + 9 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm31\n"       \
        "vfmadd231ps %%zmm28, %%zmm31, %%zmm9\n"                               \
        "vfmadd231ps %%zmm29, %%zmm31, %%zmm23\n"                              \
        ".endif\n"                                                             \
        ".if T_LEN > 10\n"                                                     \
        "vbroadcastss ((0 + 10 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm30\n"      \
        "vfmadd231ps %%zmm28, %%zmm30, %%zmm10\n"                              \
        "vfmadd231ps %%zmm29, %%zmm30, %%zmm24\n"                              \
        ".endif\n"                                                             \
        ".if T_LEN > 11\n"                                                     \
        "vbroadcastss ((0 + 11 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm31\n"      \
        "vfmadd231ps %%zmm28, %%zmm31, %%zmm11\n"                              \
        "vfmadd231ps %%zmm29, %%zmm31, %%zmm25\n"                              \
        ".endif\n"                                                             \
        ".if T_LEN > 12\n"                                                     \
        "vbroadcastss ((0 + 12 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm30\n"      \
        "vfmadd231ps %%zmm28, %%zmm30, %%zmm12\n"                              \
        "vfmadd231ps %%zmm29, %%zmm30, %%zmm26\n"                              \
        ".endif\n"                                                             \
        ".if T_LEN > 13\n"                                                     \
        "vbroadcastss ((0 + 13 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm31\n"      \
        "vfmadd231ps %%zmm28, %%zmm31, %%zmm13\n"                              \
        "vfmadd231ps %%zmm29, %%zmm31, %%zmm27\n"                              \
        ".endif\n"                                                             \
        "lea (CH_DT_BLK * D_BYTES)(%%rbx), %%rbx\n"                            \
        "lea (CH_DT_BLK * D_BYTES)(%%rcx), %%rcx\n"                            \
        "lea D_BYTES(%%rax), %%rax\n"                                          \
        "sub $1, %%r10\n"                                                      \
        "cmp $0, %%r10\n"                                                      \
        "jne 5b\n" /* label_ic_remain */                                       \
        "6:\n"     /* label_finalize_session */                                \
        "lea (%%r13, %%r11, D_BYTES), %%r10\n"                                 \
        ".if T_LEN > 0\n vmovups %%zmm0, (0 * CH_DT_BLK * D_BYTES)(%%r13)\n "  \
        ".endif\n"                                                             \
        ".if T_LEN > 1\n vmovups %%zmm1, (1 * CH_DT_BLK * D_BYTES)(%%r13)\n "  \
        ".endif\n"                                                             \
        ".if T_LEN > 2\n vmovups %%zmm2, (2 * CH_DT_BLK * D_BYTES)(%%r13)\n "  \
        ".endif\n"                                                             \
        ".if T_LEN > 3\n vmovups %%zmm3, (3 * CH_DT_BLK * D_BYTES)(%%r13)\n "  \
        ".endif\n"                                                             \
        ".if T_LEN > 4\n vmovups %%zmm4, (4 * CH_DT_BLK * D_BYTES)(%%r13)\n "  \
        ".endif\n"                                                             \
        ".if T_LEN > 5\n vmovups %%zmm5, (5 * CH_DT_BLK * D_BYTES)(%%r13)\n "  \
        ".endif\n"                                                             \
        ".if T_LEN > 6\n vmovups %%zmm6, (6 * CH_DT_BLK * D_BYTES)(%%r13)\n "  \
        ".endif\n"                                                             \
        ".if T_LEN > 7\n vmovups %%zmm7, (7 * CH_DT_BLK * D_BYTES)(%%r13)\n "  \
        ".endif\n"                                                             \
        ".if T_LEN > 8\n vmovups %%zmm8, (8 * CH_DT_BLK * D_BYTES)(%%r13)\n "  \
        ".endif\n"                                                             \
        ".if T_LEN > 9\n vmovups %%zmm9, (9 * CH_DT_BLK * D_BYTES)(%%r13)\n "  \
        ".endif\n"                                                             \
        ".if T_LEN > 10\n vmovups %%zmm10, (10 * CH_DT_BLK * "                 \
        "D_BYTES)(%%r13)\n .endif\n"                                           \
        ".if T_LEN > 11\n vmovups %%zmm11, (11 * CH_DT_BLK * "                 \
        "D_BYTES)(%%r13)\n .endif\n"                                           \
        ".if T_LEN > 12\n vmovups %%zmm12, (12 * CH_DT_BLK * "                 \
        "D_BYTES)(%%r13)\n .endif\n"                                           \
        ".if T_LEN > 13\n vmovups %%zmm13, (13 * CH_DT_BLK * "                 \
        "D_BYTES)(%%r13)\n .endif\n"                                           \
        ".if T_LEN > 0\n vmovups %%zmm14, (0 * CH_DT_BLK * D_BYTES)(%%r10)\n " \
        ".endif\n"                                                             \
        ".if T_LEN > 1\n vmovups %%zmm15, (1 * CH_DT_BLK * D_BYTES)(%%r10)\n " \
        ".endif\n"                                                             \
        ".if T_LEN > 2\n vmovups %%zmm16, (2 * CH_DT_BLK * D_BYTES)(%%r10)\n " \
        ".endif\n"                                                             \
        ".if T_LEN > 3\n vmovups %%zmm17, (3 * CH_DT_BLK * D_BYTES)(%%r10)\n " \
        ".endif\n"                                                             \
        ".if T_LEN > 4\n vmovups %%zmm18, (4 * CH_DT_BLK * D_BYTES)(%%r10)\n " \
        ".endif\n"                                                             \
        ".if T_LEN > 5\n vmovups %%zmm19, (5 * CH_DT_BLK * D_BYTES)(%%r10)\n " \
        ".endif\n"                                                             \
        ".if T_LEN > 6\n vmovups %%zmm20, (6 * CH_DT_BLK * D_BYTES)(%%r10)\n " \
        ".endif\n"                                                             \
        ".if T_LEN > 7\n vmovups %%zmm21, (7 * CH_DT_BLK * D_BYTES)(%%r10)\n " \
        ".endif\n"                                                             \
        ".if T_LEN > 8\n vmovups %%zmm22, (8 * CH_DT_BLK * D_BYTES)(%%r10)\n " \
        ".endif\n"                                                             \
        ".if T_LEN > 9\n vmovups %%zmm23, (9 * CH_DT_BLK * D_BYTES)(%%r10)\n " \
        ".endif\n"                                                             \
        ".if T_LEN > 10\n vmovups %%zmm24, (10 * CH_DT_BLK * "                 \
        "D_BYTES)(%%r10)\n .endif\n"                                           \
        ".if T_LEN > 11\n vmovups %%zmm25, (11 * CH_DT_BLK * "                 \
        "D_BYTES)(%%r10)\n .endif\n"                                           \
        ".if T_LEN > 12\n vmovups %%zmm26, (12 * CH_DT_BLK * "                 \
        "D_BYTES)(%%r10)\n .endif\n"                                           \
        ".if T_LEN > 13\n vmovups %%zmm27, (13 * CH_DT_BLK * "                 \
        "D_BYTES)(%%r10)\n .endif\n"                                           \
        "sub $T_LEN, %%r15\n"                                                  \
        "cmp $0, %%r15\n"                                                      \
        "lea (T_LEN * CH_DT_BLK * D_BYTES)(%%r13), %%r13\n"                    \
        "lea (%%r14, %%r8, D_BYTES), %%r14\n"                                  \
        "jne 1b\n" /* label_init_session */                                    \
        :                                                                      \
        : [param] "r"(param), [T_LEN] "i"(TLEN)                                \
        : "cc", "rax", "rbx", "rcx", "rdx", "rsi", "r8", "r9", "r10", "r11",   \
          "r12", "r13", "r14", "r15", "zmm0", "zmm1", "zmm2", "zmm3", "zmm4",  \
          "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12",   \
          "zmm13", "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19",       \
          "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26",       \
          "zmm27", "zmm28", "zmm29", "zmm30", "zmm31", "memory");              \
  }

#define IMPLEMENT_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T14O32_FUNC(TLEN)          \
  void winograd_b4f3_gemm_kernel_fp32_avx512_o32_t##TLEN(                      \
      const int64_t *params) {                                                 \
    if (TLEN > 3) {                                                            \
      WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T14O32_ASM(TLEN)(params);              \
      return;                                                                  \
    }                                                                          \
    __m512 zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6, zmm7;                     \
    __m512 zmm8, zmm9, zmm10, zmm11, zmm12, zmm13, zmm14, zmm15;               \
    __m512 zmm16, zmm17, zmm18, zmm19, zmm20, zmm21, zmm22, zmm23;             \
    __m512 zmm24, zmm25, zmm26, zmm27, zmm28, zmm29, zmm30, zmm31;             \
    const int64_t src_tkb_stride = params[5];                                  \
    const int64_t flt_ocb_stride = params[7];                                  \
    const int64_t dst_ocb_stride = params[6];                                  \
    const int64_t load_dst = params[8];                                        \
    const float *t_src = (const float *)(params[0]);                           \
    float *t_dst = (const float *)(params[1]);                                 \
    int64_t t = params[3];                                                     \
    do {                                                                       \
      WINOGRAD_B4F3_T14O32_KERNEL_AVX512_ITER_T(params, TLEN, t_src, t_dst);   \
      t_src += src_tkb_stride;                                                 \
      t_dst += TLEN * KERNEL_ONE_REG;                                          \
      t -= TLEN;                                                               \
    } while (t > 0);                                                           \
  }

#else
#define IMPLEMENT_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T14O32_FUNC(TLEN)          \
  void winograd_b4f3_gemm_kernel_fp32_avx512_o32_t##TLEN(                      \
      const int64_t *params) {                                                 \
    __m512 zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6, zmm7;                     \
    __m512 zmm8, zmm9, zmm10, zmm11, zmm12, zmm13, zmm14, zmm15;               \
    __m512 zmm16, zmm17, zmm18, zmm19, zmm20, zmm21, zmm22, zmm23;             \
    __m512 zmm24, zmm25, zmm26, zmm27, zmm28, zmm29, zmm30, zmm31;             \
    const int64_t src_tkb_stride = params[5];                                  \
    const int64_t flt_ocb_stride = params[7];                                  \
    const int64_t dst_ocb_stride = params[6];                                  \
    const int64_t load_dst = params[8];                                        \
    const float *t_src = (const float *)(params[0]);                           \
    float *t_dst = (const float *)(params[1]);                                 \
    int64_t t = params[3];                                                     \
    do {                                                                       \
      WINOGRAD_B4F3_T14O32_KERNEL_AVX512_ITER_T(params, TLEN, t_src, t_dst);   \
      t_src += src_tkb_stride;                                                 \
      t_dst += TLEN * KERNEL_ONE_REG;                                          \
      t -= TLEN;                                                               \
    } while (t > 0);                                                           \
  }
#endif

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
  const float *icb_flt = (const float *)(KERNEL_PARAMS[2]);                    \
  int64_t icb = KERNEL_PARAMS[4];                                              \
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

#define WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T31O16_FUNC(TLEN)                    \
  winograd_b4f3_gemm_kernel_fp32_avx512_o16_t##TLEN

#ifdef USE_INLINE_ASM
#define WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T31O16_ASM(TLEN)                     \
  winograd_b4f3_gemm_kernel_fp32_avx512_asm_o16_t##TLEN

#define DECLARE_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T31O16_ASM(TLEN)             \
  void winograd_b4f3_gemm_kernel_fp32_avx512_asm_o16_t##TLEN(                  \
      const int64_t *param)

DECLARE_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T31O16_ASM(1);
DECLARE_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T31O16_ASM(2);
DECLARE_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T31O16_ASM(3);
DECLARE_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T31O16_ASM(4);
DECLARE_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T31O16_ASM(5);
DECLARE_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T31O16_ASM(6);
DECLARE_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T31O16_ASM(7);
DECLARE_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T31O16_ASM(8);
DECLARE_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T31O16_ASM(9);
DECLARE_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T31O16_ASM(10);
DECLARE_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T31O16_ASM(11);
DECLARE_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T31O16_ASM(12);
DECLARE_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T31O16_ASM(13);
DECLARE_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T31O16_ASM(14);

#define IMPLEMENT_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T31O16_ASM(TLEN)           \
  void winograd_b4f3_gemm_kernel_fp32_avx512_asm_o16_t##TLEN(                  \
      const int64_t *param) {                                                  \
    __asm__ __volatile__(                                                      \
        ".equ P_BYTES, 8\n"                                                    \
        ".equ D_BYTES, 4\n"                                                    \
        ".equ CH_DT_BLK, 16\n"                                                 \
        ".equ SRC_IDX,      (0 * P_BYTES)\n"                                   \
        ".equ DST_IDX,      (1 * P_BYTES)\n"                                   \
        ".equ FLT_IDX,      (2 * P_BYTES)\n"                                   \
        ".equ TILES_IDX,    (3 * P_BYTES)\n"                                   \
        ".equ CHANNELS_IDX, (4 * P_BYTES)\n"                                   \
        ".equ SRC_TKB_STRIDE_IDX, (5 * P_BYTES)\n"                             \
        ".equ DST_OCB_STRIDE_IDX, (6 * P_BYTES)\n"                             \
        ".equ FLT_OCB_STRIDE_IDX, (7 * P_BYTES)\n"                             \
        ".equ LOAD_DST_IDX,       (8 * P_BYTES)\n"                             \
        ".equ T_LEN, %c[T_LEN]\n"                                              \
        "mov SRC_TKB_STRIDE_IDX(%[param]), %%r8\n"                             \
        "mov FLT_OCB_STRIDE_IDX(%[param]), %%r9\n"                             \
        "mov DST_OCB_STRIDE_IDX(%[param]), %%r11\n"                            \
        "mov LOAD_DST_IDX(%[param]), %%r12\n"                                  \
        "mov DST_IDX(%[param]), %%r13\n"                                       \
        "mov SRC_IDX(%[param]), %%r14\n"                                       \
        "mov TILES_IDX(%[param]), %%r15\n"                                     \
        "1:\n" /* label_init_session */                                        \
        "test %%r12, %%r12\n"                                                  \
        "jnz 2f\n" /* label_load_dst */                                        \
        ".if T_LEN > 0\n vpxord %%zmm0, %%zmm0, %%zmm0\n .endif\n"             \
        ".if T_LEN > 1\n vpxord %%zmm1, %%zmm1, %%zmm1\n .endif\n"             \
        ".if T_LEN > 2\n vpxord %%zmm2, %%zmm2, %%zmm2\n .endif\n"             \
        ".if T_LEN > 3\n vpxord %%zmm3, %%zmm3, %%zmm3\n .endif\n"             \
        ".if T_LEN > 4\n vpxord %%zmm4, %%zmm4, %%zmm4\n .endif\n"             \
        ".if T_LEN > 5\n vpxord %%zmm5, %%zmm5, %%zmm5\n .endif\n"             \
        ".if T_LEN > 6\n vpxord %%zmm6, %%zmm6, %%zmm6\n .endif\n"             \
        ".if T_LEN > 7\n vpxord %%zmm7, %%zmm7, %%zmm7\n .endif\n"             \
        ".if T_LEN > 8\n vpxord %%zmm8, %%zmm8, %%zmm8\n .endif\n"             \
        ".if T_LEN > 9\n vpxord %%zmm9, %%zmm9, %%zmm9\n .endif\n"             \
        ".if T_LEN > 10\n vpxord %%zmm10, %%zmm10, %%zmm10\n .endif\n"         \
        ".if T_LEN > 11\n vpxord %%zmm11, %%zmm11, %%zmm11\n .endif\n"         \
        ".if T_LEN > 12\n vpxord %%zmm12, %%zmm12, %%zmm12\n .endif\n"         \
        ".if T_LEN > 13\n vpxord %%zmm13, %%zmm13, %%zmm13\n .endif\n"         \
        ".if T_LEN > 14\n vpxord %%zmm14, %%zmm14, %%zmm14\n .endif\n"         \
        ".if T_LEN > 15\n vpxord %%zmm15, %%zmm15, %%zmm15\n .endif\n"         \
        ".if T_LEN > 16\n vpxord %%zmm16, %%zmm16, %%zmm16\n .endif\n"         \
        ".if T_LEN > 17\n vpxord %%zmm17, %%zmm17, %%zmm17\n .endif\n"         \
        ".if T_LEN > 18\n vpxord %%zmm18, %%zmm18, %%zmm18\n .endif\n"         \
        ".if T_LEN > 19\n vpxord %%zmm19, %%zmm19, %%zmm19\n .endif\n"         \
        ".if T_LEN > 20\n vpxord %%zmm20, %%zmm20, %%zmm20\n .endif\n"         \
        ".if T_LEN > 21\n vpxord %%zmm21, %%zmm21, %%zmm21\n .endif\n"         \
        ".if T_LEN > 22\n vpxord %%zmm22, %%zmm22, %%zmm22\n .endif\n"         \
        ".if T_LEN > 23\n vpxord %%zmm23, %%zmm23, %%zmm23\n .endif\n"         \
        ".if T_LEN > 24\n vpxord %%zmm24, %%zmm24, %%zmm24\n .endif\n"         \
        ".if T_LEN > 25\n vpxord %%zmm25, %%zmm25, %%zmm25\n .endif\n"         \
        ".if T_LEN > 26\n vpxord %%zmm26, %%zmm26, %%zmm26\n .endif\n"         \
        ".if T_LEN > 27\n vpxord %%zmm27, %%zmm27, %%zmm27\n .endif\n"         \
        ".if T_LEN > 28\n vpxord %%zmm28, %%zmm28, %%zmm28\n .endif\n"         \
        ".if T_LEN > 29\n vpxord %%zmm29, %%zmm29, %%zmm29\n .endif\n"         \
        ".if T_LEN > 30\n vpxord %%zmm30, %%zmm30, %%zmm30\n .endif\n"         \
        "jmp 3f\n"                                                             \
        "2:\n" /* label_load_dst */                                            \
        ".if T_LEN > 0\n vmovups (0 * CH_DT_BLK * D_BYTES)(%%r13), %%zmm0\n "  \
        ".endif\n"                                                             \
        ".if T_LEN > 1\n vmovups (1 * CH_DT_BLK * D_BYTES)(%%r13), %%zmm1\n "  \
        ".endif\n"                                                             \
        ".if T_LEN > 2\n vmovups (2 * CH_DT_BLK * D_BYTES)(%%r13), %%zmm2\n "  \
        ".endif\n"                                                             \
        ".if T_LEN > 3\n vmovups (3 * CH_DT_BLK * D_BYTES)(%%r13), %%zmm3\n "  \
        ".endif\n"                                                             \
        ".if T_LEN > 4\n vmovups (4 * CH_DT_BLK * D_BYTES)(%%r13), %%zmm4\n "  \
        ".endif\n"                                                             \
        ".if T_LEN > 5\n vmovups (5 * CH_DT_BLK * D_BYTES)(%%r13), %%zmm5\n "  \
        ".endif\n"                                                             \
        ".if T_LEN > 6\n vmovups (6 * CH_DT_BLK * D_BYTES)(%%r13), %%zmm6\n "  \
        ".endif\n"                                                             \
        ".if T_LEN > 7\n vmovups (7 * CH_DT_BLK * D_BYTES)(%%r13), %%zmm7\n "  \
        ".endif\n"                                                             \
        ".if T_LEN > 8\n vmovups (8 * CH_DT_BLK * D_BYTES)(%%r13), %%zmm8\n "  \
        ".endif\n"                                                             \
        ".if T_LEN > 9\n vmovups (9 * CH_DT_BLK * D_BYTES)(%%r13), %%zmm9\n "  \
        ".endif\n"                                                             \
        ".if T_LEN > 10\n vmovups (10 * CH_DT_BLK * D_BYTES)(%%r13), "         \
        "%%zmm10\n .endif\n"                                                   \
        ".if T_LEN > 11\n vmovups (11 * CH_DT_BLK * D_BYTES)(%%r13), "         \
        "%%zmm11\n .endif\n"                                                   \
        ".if T_LEN > 12\n vmovups (12 * CH_DT_BLK * D_BYTES)(%%r13), "         \
        "%%zmm12\n .endif\n"                                                   \
        ".if T_LEN > 13\n vmovups (13 * CH_DT_BLK * D_BYTES)(%%r13), "         \
        "%%zmm13\n .endif\n"                                                   \
        ".if T_LEN > 14\n vmovups (14 * CH_DT_BLK * D_BYTES)(%%r13), "         \
        "%%zmm14\n .endif\n"                                                   \
        ".if T_LEN > 15\n vmovups (15 * CH_DT_BLK * D_BYTES)(%%r13), "         \
        "%%zmm15\n .endif\n"                                                   \
        ".if T_LEN > 16\n vmovups (16 * CH_DT_BLK * D_BYTES)(%%r13), "         \
        "%%zmm16\n .endif\n"                                                   \
        ".if T_LEN > 17\n vmovups (17 * CH_DT_BLK * D_BYTES)(%%r13), "         \
        "%%zmm17\n .endif\n"                                                   \
        ".if T_LEN > 18\n vmovups (18 * CH_DT_BLK * D_BYTES)(%%r13), "         \
        "%%zmm18\n .endif\n"                                                   \
        ".if T_LEN > 19\n vmovups (19 * CH_DT_BLK * D_BYTES)(%%r13), "         \
        "%%zmm19\n .endif\n"                                                   \
        ".if T_LEN > 20\n vmovups (20 * CH_DT_BLK * D_BYTES)(%%r13), "         \
        "%%zmm20\n .endif\n"                                                   \
        ".if T_LEN > 21\n vmovups (21 * CH_DT_BLK * D_BYTES)(%%r13), "         \
        "%%zmm21\n .endif\n"                                                   \
        ".if T_LEN > 22\n vmovups (22 * CH_DT_BLK * D_BYTES)(%%r13), "         \
        "%%zmm22\n .endif\n"                                                   \
        ".if T_LEN > 23\n vmovups (23 * CH_DT_BLK * D_BYTES)(%%r13), "         \
        "%%zmm23\n .endif\n"                                                   \
        ".if T_LEN > 24\n vmovups (24 * CH_DT_BLK * D_BYTES)(%%r13), "         \
        "%%zmm24\n .endif\n"                                                   \
        ".if T_LEN > 25\n vmovups (25 * CH_DT_BLK * D_BYTES)(%%r13), "         \
        "%%zmm25\n .endif\n"                                                   \
        ".if T_LEN > 26\n vmovups (26 * CH_DT_BLK * D_BYTES)(%%r13), "         \
        "%%zmm26\n .endif\n"                                                   \
        ".if T_LEN > 27\n vmovups (27 * CH_DT_BLK * D_BYTES)(%%r13), "         \
        "%%zmm27\n .endif\n"                                                   \
        ".if T_LEN > 28\n vmovups (28 * CH_DT_BLK * D_BYTES)(%%r13), "         \
        "%%zmm28\n .endif\n"                                                   \
        ".if T_LEN > 29\n vmovups (29 * CH_DT_BLK * D_BYTES)(%%r13), "         \
        "%%zmm29\n .endif\n"                                                   \
        ".if T_LEN > 30\n vmovups (30 * CH_DT_BLK * D_BYTES)(%%r13), "         \
        "%%zmm30\n .endif\n"                                                   \
        "3:\n" /* label_compute_session */                                     \
        "mov %%r14, %%rax\n"                                                   \
        "mov FLT_IDX(%[param]), %%rbx\n"                                       \
        "mov CHANNELS_IDX(%[param]), %%r10\n"                                  \
        "cmp $CH_DT_BLK, %%r10\n"                                              \
        "jl 5f\n" /* label_ic_remain */                                        \
        "4:\n"    /* label_ic_body */                                          \
        ".align 16\n"                                                          \
        ".if T_LEN < 9\n"                                                      \
        ".irp IC,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15\n"                      \
        "vmovups (\\IC * CH_DT_BLK * D_BYTES)(%%rbx), %%zmm31\n"               \
        ".if T_LEN > 7\n prefetcht0 ((\\IC * CH_DT_BLK + CH_DT_BLK * "         \
        "CH_DT_BLK) * D_BYTES)(%%rbx)\n .endif\n"                              \
        ".if T_LEN > 0\n vfmadd231ps ((\\IC + 0 * CH_DT_BLK) * "               \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm0\n .endif\n"                  \
        ".if T_LEN > 1\n vfmadd231ps ((\\IC + 1 * CH_DT_BLK) * "               \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm1\n .endif\n"                  \
        ".if T_LEN > 2\n vfmadd231ps ((\\IC + 2 * CH_DT_BLK) * "               \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm2\n .endif\n"                  \
        ".if T_LEN > 3\n vfmadd231ps ((\\IC + 3 * CH_DT_BLK) * "               \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm3\n .endif\n"                  \
        ".if T_LEN > 4\n vfmadd231ps ((\\IC + 4 * CH_DT_BLK) * "               \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm4\n .endif\n"                  \
        ".if T_LEN > 5\n vfmadd231ps ((\\IC + 5 * CH_DT_BLK) * "               \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm5\n .endif\n"                  \
        ".if T_LEN > 6\n vfmadd231ps ((\\IC + 6 * CH_DT_BLK) * "               \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm6\n .endif\n"                  \
        ".if T_LEN > 7\n vfmadd231ps ((\\IC + 7 * CH_DT_BLK) * "               \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm7\n .endif\n"                  \
        ".if T_LEN > 8\n vfmadd231ps ((\\IC + 8 * CH_DT_BLK) * "               \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm8\n .endif\n"                  \
        ".if T_LEN > 9\n vfmadd231ps ((\\IC + 9 * CH_DT_BLK) * "               \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm9\n .endif\n"                  \
        ".if T_LEN > 10\n vfmadd231ps ((\\IC + 10 * CH_DT_BLK) * "             \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm10\n .endif\n"                 \
        ".if T_LEN > 11\n vfmadd231ps ((\\IC + 11 * CH_DT_BLK) * "             \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm11\n .endif\n"                 \
        ".if T_LEN > 12\n vfmadd231ps ((\\IC + 12 * CH_DT_BLK) * "             \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm12\n .endif\n"                 \
        ".if T_LEN > 13\n vfmadd231ps ((\\IC + 13 * CH_DT_BLK) * "             \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm13\n .endif\n"                 \
        ".if T_LEN > 14\n vfmadd231ps ((\\IC + 14 * CH_DT_BLK) * "             \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm14\n .endif\n"                 \
        ".if T_LEN > 15\n vfmadd231ps ((\\IC + 15 * CH_DT_BLK) * "             \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm15\n .endif\n"                 \
        ".if T_LEN > 16\n vfmadd231ps ((\\IC + 16 * CH_DT_BLK) * "             \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm16\n .endif\n"                 \
        ".if T_LEN > 17\n vfmadd231ps ((\\IC + 17 * CH_DT_BLK) * "             \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm17\n .endif\n"                 \
        ".if T_LEN > 18\n vfmadd231ps ((\\IC + 18 * CH_DT_BLK) * "             \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm18\n .endif\n"                 \
        ".if T_LEN > 19\n vfmadd231ps ((\\IC + 19 * CH_DT_BLK) * "             \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm19\n .endif\n"                 \
        ".if T_LEN > 20\n vfmadd231ps ((\\IC + 20 * CH_DT_BLK) * "             \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm20\n .endif\n"                 \
        ".if T_LEN > 21\n vfmadd231ps ((\\IC + 21 * CH_DT_BLK) * "             \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm21\n .endif\n"                 \
        ".if T_LEN > 22\n vfmadd231ps ((\\IC + 22 * CH_DT_BLK) * "             \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm22\n .endif\n"                 \
        ".if T_LEN > 23\n vfmadd231ps ((\\IC + 23 * CH_DT_BLK) * "             \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm23\n .endif\n"                 \
        ".if T_LEN > 24\n vfmadd231ps ((\\IC + 24 * CH_DT_BLK) * "             \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm24\n .endif\n"                 \
        ".if T_LEN > 25\n vfmadd231ps ((\\IC + 25 * CH_DT_BLK) * "             \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm25\n .endif\n"                 \
        ".if T_LEN > 26\n vfmadd231ps ((\\IC + 26 * CH_DT_BLK) * "             \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm26\n .endif\n"                 \
        ".if T_LEN > 27\n vfmadd231ps ((\\IC + 27 * CH_DT_BLK) * "             \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm27\n .endif\n"                 \
        ".if T_LEN > 28\n vfmadd231ps ((\\IC + 28 * CH_DT_BLK) * "             \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm28\n .endif\n"                 \
        ".if T_LEN > 29\n vfmadd231ps ((\\IC + 29 * CH_DT_BLK) * "             \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm29\n .endif\n"                 \
        ".if T_LEN > 30\n vfmadd231ps ((\\IC + 30 * CH_DT_BLK) * "             \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm30\n .endif\n"                 \
        ".endr\n"                                                              \
        "lea (CH_DT_BLK * CH_DT_BLK * D_BYTES)(%%rbx), %%rbx\n"                \
        "lea (T_LEN * CH_DT_BLK * D_BYTES)(%%rax), %%rax\n"                    \
        ".else\n" /* .if T_LEN < 9 */                                          \
        "mov $CH_DT_BLK, %%r9\n"                                               \
        "9:\n" /* label_ic */                                                  \
        "vmovups (0 * CH_DT_BLK * D_BYTES)(%%rbx), %%zmm31\n"                  \
        "prefetcht0 ((0 * CH_DT_BLK + CH_DT_BLK * CH_DT_BLK) * "               \
        "D_BYTES)(%%rbx)\n"                                                    \
        "lea (CH_DT_BLK * D_BYTES)(%%rbx), %%rbx\n"                            \
        ".if T_LEN > 0\n vfmadd231ps ((0 + 0 * CH_DT_BLK) * "                  \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm0\n .endif\n"                  \
        ".if T_LEN > 1\n vfmadd231ps ((0 + 1 * CH_DT_BLK) * "                  \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm1\n .endif\n"                  \
        ".if T_LEN > 2\n vfmadd231ps ((0 + 2 * CH_DT_BLK) * "                  \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm2\n .endif\n"                  \
        ".if T_LEN > 3\n vfmadd231ps ((0 + 3 * CH_DT_BLK) * "                  \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm3\n .endif\n"                  \
        ".if T_LEN > 4\n vfmadd231ps ((0 + 4 * CH_DT_BLK) * "                  \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm4\n .endif\n"                  \
        ".if T_LEN > 5\n vfmadd231ps ((0 + 5 * CH_DT_BLK) * "                  \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm5\n .endif\n"                  \
        ".if T_LEN > 6\n vfmadd231ps ((0 + 6 * CH_DT_BLK) * "                  \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm6\n .endif\n"                  \
        ".if T_LEN > 7\n vfmadd231ps ((0 + 7 * CH_DT_BLK) * "                  \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm7\n .endif\n"                  \
        ".if T_LEN > 8\n vfmadd231ps ((0 + 8 * CH_DT_BLK) * "                  \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm8\n .endif\n"                  \
        ".if T_LEN > 9\n vfmadd231ps ((0 + 9 * CH_DT_BLK) * "                  \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm9\n .endif\n"                  \
        ".if T_LEN > 10\n vfmadd231ps ((0 + 10 * CH_DT_BLK) * "                \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm10\n .endif\n"                 \
        ".if T_LEN > 11\n vfmadd231ps ((0 + 11 * CH_DT_BLK) * "                \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm11\n .endif\n"                 \
        ".if T_LEN > 12\n vfmadd231ps ((0 + 12 * CH_DT_BLK) * "                \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm12\n .endif\n"                 \
        ".if T_LEN > 13\n vfmadd231ps ((0 + 13 * CH_DT_BLK) * "                \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm13\n .endif\n"                 \
        ".if T_LEN > 14\n vfmadd231ps ((0 + 14 * CH_DT_BLK) * "                \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm14\n .endif\n"                 \
        ".if T_LEN > 15\n vfmadd231ps ((0 + 15 * CH_DT_BLK) * "                \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm15\n .endif\n"                 \
        ".if T_LEN > 16\n vfmadd231ps ((0 + 16 * CH_DT_BLK) * "                \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm16\n .endif\n"                 \
        ".if T_LEN > 17\n vfmadd231ps ((0 + 17 * CH_DT_BLK) * "                \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm17\n .endif\n"                 \
        ".if T_LEN > 18\n vfmadd231ps ((0 + 18 * CH_DT_BLK) * "                \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm18\n .endif\n"                 \
        ".if T_LEN > 19\n vfmadd231ps ((0 + 19 * CH_DT_BLK) * "                \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm19\n .endif\n"                 \
        ".if T_LEN > 20\n vfmadd231ps ((0 + 20 * CH_DT_BLK) * "                \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm20\n .endif\n"                 \
        ".if T_LEN > 21\n vfmadd231ps ((0 + 21 * CH_DT_BLK) * "                \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm21\n .endif\n"                 \
        ".if T_LEN > 22\n vfmadd231ps ((0 + 22 * CH_DT_BLK) * "                \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm22\n .endif\n"                 \
        ".if T_LEN > 23\n vfmadd231ps ((0 + 23 * CH_DT_BLK) * "                \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm23\n .endif\n"                 \
        ".if T_LEN > 24\n vfmadd231ps ((0 + 24 * CH_DT_BLK) * "                \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm24\n .endif\n"                 \
        ".if T_LEN > 25\n vfmadd231ps ((0 + 25 * CH_DT_BLK) * "                \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm25\n .endif\n"                 \
        ".if T_LEN > 26\n vfmadd231ps ((0 + 26 * CH_DT_BLK) * "                \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm26\n .endif\n"                 \
        ".if T_LEN > 27\n vfmadd231ps ((0 + 27 * CH_DT_BLK) * "                \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm27\n .endif\n"                 \
        ".if T_LEN > 28\n vfmadd231ps ((0 + 28 * CH_DT_BLK) * "                \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm28\n .endif\n"                 \
        ".if T_LEN > 29\n vfmadd231ps ((0 + 29 * CH_DT_BLK) * "                \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm29\n .endif\n"                 \
        ".if T_LEN > 30\n vfmadd231ps ((0 + 30 * CH_DT_BLK) * "                \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm30\n .endif\n"                 \
        "lea D_BYTES(%%rax), %%rax\n"                                          \
        "sub $1, %%r9\n"                                                       \
        "cmp $0, %%r9\n"                                                       \
        "jne 9b\n" /* label_ic  */                                             \
        "lea ((T_LEN - 1) * CH_DT_BLK * D_BYTES)(%%rax), %%rax\n"              \
        ".endif\n" /* .if T_LEN < 9 */                                         \
        "sub $CH_DT_BLK, %%r10\n"                                              \
        "cmp $CH_DT_BLK, %%r10\n"                                              \
        "jge 4b\n" /* label_ic_body */                                         \
        "cmp $0, %%r10\n"                                                      \
        "je 6f\n" /* label_finalize_session */                                 \
        "5:\n"    /* label_ic_remain */                                        \
        "vmovups (0 * CH_DT_BLK * D_BYTES)(%%rbx), %%zmm31\n"                  \
        "lea (CH_DT_BLK * D_BYTES)(%%rbx), %%rbx\n"                            \
        ".if T_LEN > 0\n vfmadd231ps ((0 + 0 * CH_DT_BLK) * "                  \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm0\n .endif\n"                  \
        ".if T_LEN > 1\n vfmadd231ps ((0 + 1 * CH_DT_BLK) * "                  \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm1\n .endif\n"                  \
        ".if T_LEN > 2\n vfmadd231ps ((0 + 2 * CH_DT_BLK) * "                  \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm2\n .endif\n"                  \
        ".if T_LEN > 3\n vfmadd231ps ((0 + 3 * CH_DT_BLK) * "                  \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm3\n .endif\n"                  \
        ".if T_LEN > 4\n vfmadd231ps ((0 + 4 * CH_DT_BLK) * "                  \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm4\n .endif\n"                  \
        ".if T_LEN > 5\n vfmadd231ps ((0 + 5 * CH_DT_BLK) * "                  \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm5\n .endif\n"                  \
        ".if T_LEN > 6\n vfmadd231ps ((0 + 6 * CH_DT_BLK) * "                  \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm6\n .endif\n"                  \
        ".if T_LEN > 7\n vfmadd231ps ((0 + 7 * CH_DT_BLK) * "                  \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm7\n .endif\n"                  \
        ".if T_LEN > 8\n vfmadd231ps ((0 + 8 * CH_DT_BLK) * "                  \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm8\n .endif\n"                  \
        ".if T_LEN > 9\n vfmadd231ps ((0 + 9 * CH_DT_BLK) * "                  \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm9\n .endif\n"                  \
        ".if T_LEN > 10\n vfmadd231ps ((0 + 10 * CH_DT_BLK) * "                \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm10\n .endif\n"                 \
        ".if T_LEN > 11\n vfmadd231ps ((0 + 11 * CH_DT_BLK) * "                \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm11\n .endif\n"                 \
        ".if T_LEN > 12\n vfmadd231ps ((0 + 12 * CH_DT_BLK) * "                \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm12\n .endif\n"                 \
        ".if T_LEN > 13\n vfmadd231ps ((0 + 13 * CH_DT_BLK) * "                \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm13\n .endif\n"                 \
        ".if T_LEN > 14\n vfmadd231ps ((0 + 14 * CH_DT_BLK) * "                \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm14\n .endif\n"                 \
        ".if T_LEN > 15\n vfmadd231ps ((0 + 15 * CH_DT_BLK) * "                \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm15\n .endif\n"                 \
        ".if T_LEN > 16\n vfmadd231ps ((0 + 16 * CH_DT_BLK) * "                \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm16\n .endif\n"                 \
        ".if T_LEN > 17\n vfmadd231ps ((0 + 17 * CH_DT_BLK) * "                \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm17\n .endif\n"                 \
        ".if T_LEN > 18\n vfmadd231ps ((0 + 18 * CH_DT_BLK) * "                \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm18\n .endif\n"                 \
        ".if T_LEN > 19\n vfmadd231ps ((0 + 19 * CH_DT_BLK) * "                \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm19\n .endif\n"                 \
        ".if T_LEN > 20\n vfmadd231ps ((0 + 20 * CH_DT_BLK) * "                \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm20\n .endif\n"                 \
        ".if T_LEN > 21\n vfmadd231ps ((0 + 21 * CH_DT_BLK) * "                \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm21\n .endif\n"                 \
        ".if T_LEN > 22\n vfmadd231ps ((0 + 22 * CH_DT_BLK) * "                \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm22\n .endif\n"                 \
        ".if T_LEN > 23\n vfmadd231ps ((0 + 23 * CH_DT_BLK) * "                \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm23\n .endif\n"                 \
        ".if T_LEN > 24\n vfmadd231ps ((0 + 24 * CH_DT_BLK) * "                \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm24\n .endif\n"                 \
        ".if T_LEN > 25\n vfmadd231ps ((0 + 25 * CH_DT_BLK) * "                \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm25\n .endif\n"                 \
        ".if T_LEN > 26\n vfmadd231ps ((0 + 26 * CH_DT_BLK) * "                \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm26\n .endif\n"                 \
        ".if T_LEN > 27\n vfmadd231ps ((0 + 27 * CH_DT_BLK) * "                \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm27\n .endif\n"                 \
        ".if T_LEN > 28\n vfmadd231ps ((0 + 28 * CH_DT_BLK) * "                \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm28\n .endif\n"                 \
        ".if T_LEN > 29\n vfmadd231ps ((0 + 29 * CH_DT_BLK) * "                \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm29\n .endif\n"                 \
        ".if T_LEN > 30\n vfmadd231ps ((0 + 30 * CH_DT_BLK) * "                \
        "D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm30\n .endif\n"                 \
        "lea D_BYTES(%%rax), %%rax\n"                                          \
        "sub $1, %%r10\n"                                                      \
        "cmp $0, %%r10\n"                                                      \
        "jne 5b\n" /* label_ic_remain */                                       \
        "6:\n"     /* label_finalize_session */                                \
        ".if T_LEN > 0\n vmovups %%zmm0, (0 * CH_DT_BLK * D_BYTES)(%%r13)\n "  \
        ".endif\n"                                                             \
        ".if T_LEN > 1\n vmovups %%zmm1, (1 * CH_DT_BLK * D_BYTES)(%%r13)\n "  \
        ".endif\n"                                                             \
        ".if T_LEN > 2\n vmovups %%zmm2, (2 * CH_DT_BLK * D_BYTES)(%%r13)\n "  \
        ".endif\n"                                                             \
        ".if T_LEN > 3\n vmovups %%zmm3, (3 * CH_DT_BLK * D_BYTES)(%%r13)\n "  \
        ".endif\n"                                                             \
        ".if T_LEN > 4\n vmovups %%zmm4, (4 * CH_DT_BLK * D_BYTES)(%%r13)\n "  \
        ".endif\n"                                                             \
        ".if T_LEN > 5\n vmovups %%zmm5, (5 * CH_DT_BLK * D_BYTES)(%%r13)\n "  \
        ".endif\n"                                                             \
        ".if T_LEN > 6\n vmovups %%zmm6, (6 * CH_DT_BLK * D_BYTES)(%%r13)\n "  \
        ".endif\n"                                                             \
        ".if T_LEN > 7\n vmovups %%zmm7, (7 * CH_DT_BLK * D_BYTES)(%%r13)\n "  \
        ".endif\n"                                                             \
        ".if T_LEN > 8\n vmovups %%zmm8, (8 * CH_DT_BLK * D_BYTES)(%%r13)\n "  \
        ".endif\n"                                                             \
        ".if T_LEN > 9\n vmovups %%zmm9, (9 * CH_DT_BLK * D_BYTES)(%%r13)\n "  \
        ".endif\n"                                                             \
        ".if T_LEN > 10\n vmovups %%zmm10, (10 * CH_DT_BLK * "                 \
        "D_BYTES)(%%r13)\n .endif\n"                                           \
        ".if T_LEN > 11\n vmovups %%zmm11, (11 * CH_DT_BLK * "                 \
        "D_BYTES)(%%r13)\n .endif\n"                                           \
        ".if T_LEN > 12\n vmovups %%zmm12, (12 * CH_DT_BLK * "                 \
        "D_BYTES)(%%r13)\n .endif\n"                                           \
        ".if T_LEN > 13\n vmovups %%zmm13, (13 * CH_DT_BLK * "                 \
        "D_BYTES)(%%r13)\n .endif\n"                                           \
        ".if T_LEN > 14\n vmovups %%zmm14, (14 * CH_DT_BLK * "                 \
        "D_BYTES)(%%r13)\n .endif\n"                                           \
        ".if T_LEN > 15\n vmovups %%zmm15, (15 * CH_DT_BLK * "                 \
        "D_BYTES)(%%r13)\n .endif\n"                                           \
        ".if T_LEN > 16\n vmovups %%zmm16, (16 * CH_DT_BLK * "                 \
        "D_BYTES)(%%r13)\n .endif\n"                                           \
        ".if T_LEN > 17\n vmovups %%zmm17, (17 * CH_DT_BLK * "                 \
        "D_BYTES)(%%r13)\n .endif\n"                                           \
        ".if T_LEN > 18\n vmovups %%zmm18, (18 * CH_DT_BLK * "                 \
        "D_BYTES)(%%r13)\n .endif\n"                                           \
        ".if T_LEN > 19\n vmovups %%zmm19, (19 * CH_DT_BLK * "                 \
        "D_BYTES)(%%r13)\n .endif\n"                                           \
        ".if T_LEN > 20\n vmovups %%zmm20, (20 * CH_DT_BLK * "                 \
        "D_BYTES)(%%r13)\n .endif\n"                                           \
        ".if T_LEN > 21\n vmovups %%zmm21, (21 * CH_DT_BLK * "                 \
        "D_BYTES)(%%r13)\n .endif\n"                                           \
        ".if T_LEN > 22\n vmovups %%zmm22, (22 * CH_DT_BLK * "                 \
        "D_BYTES)(%%r13)\n .endif\n"                                           \
        ".if T_LEN > 23\n vmovups %%zmm23, (23 * CH_DT_BLK * "                 \
        "D_BYTES)(%%r13)\n .endif\n"                                           \
        ".if T_LEN > 24\n vmovups %%zmm24, (24 * CH_DT_BLK * "                 \
        "D_BYTES)(%%r13)\n .endif\n"                                           \
        ".if T_LEN > 25\n vmovups %%zmm25, (25 * CH_DT_BLK * "                 \
        "D_BYTES)(%%r13)\n .endif\n"                                           \
        ".if T_LEN > 26\n vmovups %%zmm26, (26 * CH_DT_BLK * "                 \
        "D_BYTES)(%%r13)\n .endif\n"                                           \
        ".if T_LEN > 27\n vmovups %%zmm27, (27 * CH_DT_BLK * "                 \
        "D_BYTES)(%%r13)\n .endif\n"                                           \
        ".if T_LEN > 28\n vmovups %%zmm28, (28 * CH_DT_BLK * "                 \
        "D_BYTES)(%%r13)\n .endif\n"                                           \
        ".if T_LEN > 29\n vmovups %%zmm29, (29 * CH_DT_BLK * "                 \
        "D_BYTES)(%%r13)\n .endif\n"                                           \
        ".if T_LEN > 30\n vmovups %%zmm30, (30 * CH_DT_BLK * "                 \
        "D_BYTES)(%%r13)\n .endif\n"                                           \
        "sub $T_LEN, %%r15\n"                                                  \
        "cmp $0, %%r15\n"                                                      \
        "lea (T_LEN * CH_DT_BLK * D_BYTES)(%%r13), %%r13\n"                    \
        "lea (%%r14, %%r8, D_BYTES), %%r14\n"                                  \
        "jne 1b\n" /* label_init_session */                                    \
        :                                                                      \
        : [param] "r"(param), [T_LEN] "i"(TLEN)                                \
        : "cc", "rax", "rbx", "rcx", "rdx", "rsi", "r8", "r9", "r10", "r11",   \
          "r12", "r13", "r14", "r15", "zmm0", "zmm1", "zmm2", "zmm3", "zmm4",  \
          "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12",   \
          "zmm13", "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19",       \
          "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26",       \
          "zmm27", "zmm28", "zmm29", "zmm30", "zmm31", "memory");              \
  }

#define IMPLEMENT_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T31O16_FUNC(TLEN)          \
  void winograd_b4f3_gemm_kernel_fp32_avx512_o16_t##TLEN(                      \
      const int64_t *params) {                                                 \
    WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T31O16_ASM(TLEN)(params);                \
    return;                                                                    \
  }
#else
#define IMPLEMENT_WINO_B4F3_GEMM_KERNEL_FP32_AVX512_T31O16_FUNC(TLEN)          \
  void winograd_b4f3_gemm_kernel_fp32_avx512_o16_t##TLEN(                      \
      const int64_t *params) {                                                 \
    __m512 zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6, zmm7;                     \
    __m512 zmm8, zmm9, zmm10, zmm11, zmm12, zmm13, zmm14, zmm15;               \
    __m512 zmm16, zmm17, zmm18, zmm19, zmm20, zmm21, zmm22, zmm23;             \
    __m512 zmm24, zmm25, zmm26, zmm27, zmm28, zmm29, zmm30, zmm31;             \
    const int64_t src_tkb_stride = params[5];                                  \
    const int64_t load_dst = params[8];                                        \
    const float *t_src = (const float *)params[0];                             \
    float *t_dst = (float *)(params[1]);                                       \
    int64_t t = params[3];                                                     \
    do {                                                                       \
      WINOGRAD_B4F3_T31O16_KERNEL_AVX512_ITER_T(params, TLEN, t_src, t_dst);   \
      t_src += src_tkb_stride;                                                 \
      t_dst += TLEN * KERNEL_ONE_REG;                                          \
      t -= TLEN;                                                               \
    } while (t > 0);                                                           \
  }
#endif

void winograd_b4f3_srctrans_fp32_avx512(float *tile_buffer, const int64_t ih,
                                        const int64_t iw, const int64_t src_h,
                                        const int64_t src_w,
                                        const int64_t src_trans_ti_stride,
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