#ifndef COMMON_H_
#define COMMON_H_

#include <stdlib.h>

typedef u_int64_t uint64_t;

#define DIV_UP(A, B) (((A) + (B)-1) / (B))
#define ROUND(A, B) ((A) / (B) * (B))
#define ROUND_UP(A, B) (((A) + (B)-1) / (B) * (B))
#define MIN(A, B) (((A) < (B)) ? (A) : (B))
#define MAX(A, B) (((A) > (B)) ? (A) : (B))

#define X86_CACHELINE_BYTES 64

#ifdef __DEBUG
#define WINO_DEBUG(...)                                                        \
  do {                                                                         \
    fprintf(stderr, "[Debug ] %s %s(Line %d): ", __FILE__, __FUNCTION__,       \
            __LINE__);                                                         \
    fprintf(stderr, __VA_ARGS__);                                              \
  } while (0)
#else
#define WINO_DEBUG(...)
#endif

// opt params
typedef struct {
  // cvt filter
  int cvt_filter_precomputed;

  // padding
  int64_t padded_ic;
  int64_t padded_oc;

  // tiles
  int64_t num_tiles_h;
  int64_t num_tiles_w;
  int64_t num_tiles_b;
  int64_t num_tiles;

  // blocking
  int64_t ic_l2_blk;
  int64_t oc_l2_blk;
  int64_t tiles_l2_blk;

  // array length
  int64_t src_permute_len;
  int64_t dst_permute_len;
  int64_t cvt_flt_len;
  int override_gemm;
  int64_t blk_tile_in_len;
  int64_t blk_matmul_in_len;
  int64_t src_workspace_len;
  int64_t src_trans_len;
  int64_t gemm_out_len;
  int64_t blk_matmul_out_len;
  int64_t dst_trans_len;
  int64_t workspace_len;

  uint64_t temp_buffer_size;
  uint64_t work_buffer_size;
} WinogradOptParams;

typedef struct {
  int64_t b;
  int64_t th;
  int64_t tw;
} TileIndex;

// helper functions
inline TileIndex calculateTileIndex(WinogradOptParams param,
                                    const int64_t tileId) {
  TileIndex tile_index;
  const int64_t hw = tileId % param.num_tiles_b;
  tile_index.b = tileId / param.num_tiles_b;
  tile_index.th = hw / param.num_tiles_w;
  tile_index.tw = hw % param.num_tiles_w;

  return tile_index;
}

#endif