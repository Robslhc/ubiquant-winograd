#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#include "omp.h"

#include "wgb2f3/wgb2f3.h"
#include "wgb4f3/avx512/wgb4f3_avx512.h"

#ifdef __DEBUG
#define inline
#endif
#define LOOP_NUM 100

inline double timestamp() {
  struct timeval tv;
  gettimeofday(&tv, 0);
  return tv.tv_sec + tv.tv_usec * 1.e-6;
}

inline void no4k_aligned(long *num) {
  long flag = *num;
  if (flag % 4096 == 0)
    (*num) += 128;
}

void winograd_init(const int layer_num, const int Batch[], const int C[],
                   const int H[], const int W[], const int K[], long *ISTRIDE,
                   long *FSTRIDE, long *OSTRIDE) {
  int tmp;
  /* Compute max stride for input, filter and output. */
  long istride, fstride, ostride;
  istride = fstride = ostride = 0;

  for (int i = 0; i < layer_num; i++) {
    tmp = Batch[i] * (H[i] - 2) / 2 * (W[i] - 2) / 2 * C[i];
    if (tmp > istride)
      istride = tmp;

    tmp = C[i] * K[i];
    if (tmp > fstride)
      fstride = tmp;

    tmp = Batch[i] * (H[i] - 2) / 2 * (W[i] - 2) / 2 * K[i];
    if (tmp > ostride)
      ostride = tmp;
  }

  no4k_aligned(&istride);
  no4k_aligned(&fstride);
  no4k_aligned(&ostride);
  *ISTRIDE = istride;
  *FSTRIDE = fstride;
  *OSTRIDE = ostride;
}

int naive_conv(float *in, float *kn, float *out, const int N, const int C,
               const int H, const int W, const int K) {
  int inpos, knpos, outpos;

  int dimIn[4] = {N, C, H, W};
  int dimKn[4] = {K, C, 3, 3};
  int dimOut[4] = {N, K, H - 2, W - 2};

  int ingap[3] = {dimIn[1] * dimIn[2] * dimIn[3], dimIn[2] * dimIn[3],
                  dimIn[3]};
  int kngap[3] = {dimKn[1] * dimKn[2] * dimKn[3], dimKn[2] * dimKn[3],
                  dimKn[3]};
  int outgap[3] = {dimOut[1] * dimOut[2] * dimOut[3], dimOut[2] * dimOut[3],
                   dimOut[3]};

#pragma omp parallel for private(inpos, knpos, outpos)
  for (int inn = 0; inn < dimIn[0]; inn++)
    for (int knn = 0; knn < dimKn[0]; knn++)
      for (int inc = 0; inc < dimIn[1]; inc++) {
        for (int outh = 0; outh < dimOut[2]; outh++)
          for (int outw = 0; outw < dimOut[3]; outw++) {
            outpos =
                inn * outgap[0] + knn * outgap[1] + outh * outgap[2] + outw;
            for (int knh = 0; knh < dimKn[2]; knh++)
              for (int knw = 0; knw < dimKn[3]; knw++) {
                inpos = inn * ingap[0] + inc * ingap[1] +
                        (outh + knh) * ingap[2] + (outw + knw);
                // knpos = knn*kngap[0] + inc*kngap[1] + 8 - (knh*kngap[2] +
                // knw);
                knpos = knn * kngap[0] + inc * kngap[1] + knh * kngap[2] + knw;
                out[outpos] += in[inpos] * kn[knpos];
              }
          }
      }

  return 0;
}

void winograd_conv(const int layer_idx, const int validation_mode,
                   const int irows, const int icols, const int C, const int K,
                   const int batch, long *total_flops, double *total_time) {
  long i, j, n;
  const int outHeight = irows - 2;
  const int outWidth = icols - 2;
  const int sizeI = irows * icols;
  const int sizeF = 3 * 3;
  const int sizeO = outHeight * outWidth;
  const int P = batch * outHeight / 2 * outWidth / 2;

  int ret;

  float *image, *filter, *out;
  image = (float *)malloc(batch * C * sizeI * sizeof(float));
  assert(image != NULL);
  filter = (float *)malloc(K * C * sizeF * sizeof(float));
  assert(filter != NULL);
  out = (float *)malloc(batch * K * sizeO * sizeof(float));
  assert(out != NULL);
#ifndef WINO_B4
  float *U, *V, *M;
  U = (float *)aligned_alloc(64, sizeof(float) * 4 * 4 * K * C);
  V = (float *)aligned_alloc(64, sizeof(float) * 4 * 4 * C * P);
  M = (float *)aligned_alloc(64, sizeof(float) * 4 * 4 * K * P);
#endif

#pragma omp parallel for private(i)
  for (long i = 0; i < batch * C * sizeI; i++)
    image[i] = (float)(i % 10 + 1);
    // image[i] = rand()%10;
#pragma omp parallel for private(i)
  for (long i = 0; i < K * C * sizeF; i++)
    filter[i] = (float)(i / sizeF + 1);
    // filter[i] = rand()%10;

    // winconv 4x3 init params
#ifdef WINO_B4
  WinogradOptParams param =
      init_winconv_4x3_params(batch, C, K, irows, icols, 0);
  float *temp_buffer = (float *)malloc(param.temp_buffer_size);
  assert(temp_buffer != NULL);
  float *work_buffer = (float *)malloc(param.work_buffer_size);
  assert(work_buffer != NULL);
#endif

  // Warm up
#ifndef WINO_B4
  winconv_2x3(image, irows, icols, C, filter, K, batch, out, U, V, M);
#else
  winconv_4x3_avx512(param, image, irows, icols, C, filter, K, batch, out,
                     temp_buffer, work_buffer);
#endif
  if (validation_mode) { // Verify mode. Check the result
    float *out_ref = (float *)malloc(batch * K * sizeO * sizeof(float));
    memset(out_ref, 0, batch * K * sizeO * sizeof(float));

    naive_conv(image, filter, out_ref, batch, C, irows, icols, K);
    printf("Layer %-2d: (Channel Height Weight Filter Batch) = "
           "(%-3d %-3d %-3d %-3d %-3d) : ",
           layer_idx, C, irows, icols, K, batch);
    long n;
    for (n = 0; n < batch * sizeO * K; n++)
      if (fabs((out[n] - out_ref[n]) / out_ref[n]) > 1e-4) {
        printf(
            "Validation Failed ! winogradConv[%d] = %f || directConv[%d] = %f "
            "\n",
            n, out[n], n, out_ref[n]);
        break;
      }
    if (n == batch * sizeO * K)
      printf("Validation Passed !\n");
    free(out_ref);
  } else { // Benchmark mode
    double start_time = timestamp();
    for (int i = 0; i < LOOP_NUM; i++) {
#ifndef WINO_B4
      winconv_2x3(image, irows, icols, C, filter, K, batch, out, U, V, M);
#else
      winconv_4x3_avx512(param, image, irows, icols, C, filter, K, batch, out,
                         temp_buffer, work_buffer);
#endif
    }
    double end_time = timestamp();

    double elapse_time_all = end_time - start_time;

    double elapse_time = elapse_time_all / LOOP_NUM;
    *total_time += elapse_time;

    long nflops = (long)batch * K * C * (irows - 2) * (icols - 2) * 3 * 3 * 2;
    double gflops = (double)nflops * 1.0e-9 / elapse_time;
    *total_flops += nflops;
    printf("Layer %-2d:  Elapse time %lf ms. ( %7.2lf GFlops) \n", layer_idx,
           elapse_time * 1000, gflops);
  }
#ifndef WINO_B4
  free(U);
  free(V);
  free(M);
#else
  free(temp_buffer);
  free(work_buffer);
#endif
  free(image);
  free(filter);
  free(out);
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    printf("Usage: %s layer.conf [validation=0/1] \n", argv[0]);
    // printf("Please provided layer configs. Aborting\n");
    exit(-1);
  }
  FILE *input = fopen(argv[1], "r");
  if (!input) {
    printf("File open failed. Aborting...\n");
    exit(-1);
  }
  int validation_mode = 0; // 0 : benchmark mode, 1: validation mode
  if (argc > 2)
    validation_mode = atoi(argv[2]);

  int layer_num, batch;
  fscanf(input, "%d", &layer_num);
  if (layer_num <= 0) {
    printf("Invalid layer num %d. Aborting\n", layer_num);
    fclose(input);
    exit(1);
  }
  int *C_arr = (int *)malloc(sizeof(int) * layer_num);     // Channel
  int *H_arr = (int *)malloc(sizeof(int) * layer_num);     // Image Height
  int *W_arr = (int *)malloc(sizeof(int) * layer_num);     // Image Width
  int *K_arr = (int *)malloc(sizeof(int) * layer_num);     // Filters
  int *Batch_arr = (int *)malloc(sizeof(int) * layer_num); // Batch

  for (int l = 0; l < layer_num; ++l) {
    fscanf(input, "%d%d%d%d%d", &C_arr[l], &H_arr[l], &W_arr[l], &K_arr[l],
           &Batch_arr[l]);
  }
  fclose(input);

  // srand(time(NULL));
  srand(20210930);

  double total_time = 0.0;
  long total_flops = 0;

  // long istride;
  // long fstride;
  // long ostride;

  // winograd_init(layer_num, Batch_arr, C_arr, H_arr, W_arr, K_arr, &istride,
  //               &fstride, &ostride);

  for (int l = 0; l < layer_num; l++) {
    winograd_conv(l, validation_mode, H_arr[l], W_arr[l], C_arr[l], K_arr[l],
                  Batch_arr[l], &total_flops, &total_time);
  }

  if (!validation_mode)
    printf("Total elapse time: %lf. ( %7.2lf GFlops) \n", total_time,
           (double)total_flops * 1.0e-9 / total_time);

  if (C_arr)
    free(C_arr);
  if (H_arr)
    free(H_arr);
  if (W_arr)
    free(W_arr);
  if (K_arr)
    free(K_arr);
  if (Batch_arr)
    free(Batch_arr);
  return 0;
}
