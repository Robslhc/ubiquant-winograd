#ifndef TIMER_H_
#define TIMER_H_

#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#define CONVERT_TO_NCHW16_TIMER 0
#define CONVERT_FILTER_TIMER 1
#define WINO_COMPUTE_TIMER 2
#define CONVERT_FROM_NCHW16_TIMER 3
#define NUM_EVENTS 4

struct wino_timer_t {
  double records[NUM_EVENTS];
  double temp_points[NUM_EVENTS];
};

inline void wino_profile_tic(struct wino_timer_t *t, int event_id) {
  struct timeval tv;
  gettimeofday(&tv, 0);
  t->temp_points[event_id] = tv.tv_sec + tv.tv_usec * 1.e-6;
}

inline void wino_profile_toc(struct wino_timer_t *t, int event_id) {
  struct timeval tv;
  gettimeofday(&tv, 0);
  double end = tv.tv_sec + tv.tv_usec * 1.e-6;
  t->records[event_id] += end - t->temp_points[event_id];
}

inline void wino_profile_init(struct wino_timer_t *t) {
  memset(t->records, 0, NUM_EVENTS * sizeof(double));
}

inline void wino_profile_result(struct wino_timer_t *t, int loop_num) {
  const char *event_name_[NUM_EVENTS] = {"CONVERT_TO_NCHW16", "CONVERT_FILTER",
                                         "WINO_COMPUTE", "CONVERT_FROM_NCHW16"};

  double total_time = 0;
  for (int event = 0; event < NUM_EVENTS; ++event)
    total_time += t->records[event];
  printf("profile results\n");
  for (int event = 0; event < NUM_EVENTS; ++event) {
    printf("\t%s time: %lf ms ( %7.2f %% )\n", event_name_[event],
           t->records[event] / loop_num * 1000,
           t->records[event] / total_time * 100);
  }
}

#ifdef PROFILE
#define START_TIMER(TIMER, EVENT) wino_profile_tic(TIMER, EVENT)
#define END_TIMER(TIMER, EVENT) wino_profile_toc(TIMER, EVENT)
#else
#define START_TIMER(TIMER, EVENT)
#define END_TIMER(TIMER, EVENT)
#endif

#endif