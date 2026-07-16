/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <pthread.h>
#include <sched.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#include <stdint.h>
#include <errno.h>
#include <string.h>

#include "perf.h"

static int64_t phase __attribute__((aligned(64)));
static int64_t waiters __attribute__((aligned(64)));
static int64_t pingpong[16] __attribute__((aligned(64)));

/* active ping-pong buffer (two cache lines): points at the static array by
   default, or a per-pair mmap'd page in -L (NUMA-local) mode */
static volatile int64_t *pp;
static int numa_local = 0;

typedef struct {
  int tid;
  int core;
  int nthr;
  double avg;     /* average cycles per one-way hop */
  double avg_ns;  /* average nanoseconds per one-way hop */
} arg_t;

static void bind_to_core(int tid) {
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(tid, &set);
  int rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &set);
  switch(rc) {
  case EFAULT: printf("efault\n"); break;
  case EINVAL: printf("einval\n"); break;
  case ESRCH:  printf("esrch\n");  break;
  default: break;
  }
  assert(rc == 0);
}

static pthread_t threads[2] __attribute__((aligned(64)));
static arg_t argarray[2] __attribute__((aligned(64)));
static int w = 0;

/* iterations per measured pair; runtime-settable via -n (rounded to x64) */
static long ITERS = (1L<<20);

__thread int cycle_fd = -1;

static inline uint64_t read_cycles() {
  uint64_t c1 = 0;
  if(cycle_fd == -1) {
    cycle_fd = init_perf_counter(PERF_COUNT_HW_CPU_CYCLES);
  }
  ssize_t got = read(cycle_fd, &c1, sizeof(c1));
  if(got != sizeof(c1)) {
    perror("read(perf counter)");
  }
  assert(got == sizeof(c1));
  return c1;
}

static inline void close_cycles() {
  if(cycle_fd != -1) {
    close(cycle_fd);
    cycle_fd = -1;
  }
}

/* monotonic nanosecond clock: immune to NTP steps, vDSO-fast, ns resolution */
static inline uint64_t now_ns() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

/* In -L mode the tid-0 worker (pinned to core i) first-touches both cache
   lines, so the kernel's first-touch policy homes the page on core i's NUMA
   node. Called after bind_to_core, before the timed region. */
static inline void first_touch(int tid) {
  if(numa_local && tid == 0) {
    pp[0] = 1;   /* line 0 (used by all workers) */
    pp[8] = 0;   /* line 1 (used by the two-cache-line worker) */
  }
}

static inline void barrier(int64_t nthr) {
  volatile int64_t *v_phase = (volatile int64_t*)&phase;
  volatile int64_t *v_waiters = (volatile int64_t*)&waiters;
  int64_t phase_ = phase;

  int64_t left = __sync_sub_and_fetch(v_waiters, 1);
  assert(left >= 0);
  if(left == 0) {
    waiters = nthr;
#ifdef __amd64__
    asm volatile ("" ::: "memory");
#else
    __sync_synchronize();
#endif
    phase = (~phase_) & 1;
  }
  else {
    while(phase_ == *v_phase) {}
  }
}


static void *two_cl_worker(void *args_) {
  arg_t *args = (arg_t*)args_;
  int core = args->core;
  int tid = args->tid;
  int otid = (~tid)&1;
  long iters = ITERS;
  uint64_t c0,c1,t0,t1;
  volatile int64_t *m = &pp[8*tid];
  volatile int64_t *o = &pp[8*otid];

  bind_to_core(core);
  first_touch(tid);
  c0 = read_cycles();
  t0 = now_ns();

  barrier(2);
  /* a loop with a spin barrier */
  while(iters) {
    while(*m == 0) {}
    *m = 0;
    __sync_synchronize();
    *o = 1;
    --iters;
  }
  c1 = read_cycles();
  t1 = now_ns();
  args->avg = ((double)(c1-c0))/ITERS/2.0;
  args->avg_ns = ((double)(t1-t0))/ITERS/2.0;
  close_cycles();
  return NULL;
}


static void *one_cl_worker(void *args_) {
  arg_t *args = (arg_t*)args_;
  int core = args->core;
  int tid = args->tid;
  int otid = (~tid)&1;
  long iters = ITERS;
  uint64_t c0,c1,t0,t1;
  volatile int64_t *m = &pp[0];

  bind_to_core(core);
  first_touch(tid);
  c0 = read_cycles();
  t0 = now_ns();

  barrier(2);
  /* a loop with a spin barrier */
  while(iters) {
#pragma unroll
    for(int i = 0; i < 64; i++) {
      while(*m != tid) {}
      *m = otid;
#ifdef __aarch64__
      __sync_synchronize();
#endif
    }
    iters-=64;
  }
  c1 = read_cycles();
  t1 = now_ns();
  args->avg = ((double)(c1-c0))/ITERS/2.0;
  args->avg_ns = ((double)(t1-t0))/ITERS/2.0;
  close_cycles();
  return NULL;
}


/* CAS variant of one_cl_worker: hand off a single cache line by atomically
   flipping it from my tid to the other tid (matches nviennot bench 1). Uses relaxed ordering to match nviennot bench 1. */
static void *cas_cl_worker(void *args_) {
  arg_t *args = (arg_t*)args_;
  int core = args->core;
  int tid = args->tid;
  int otid = (~tid)&1;
  long iters = ITERS;
  uint64_t c0,c1,t0,t1;
  volatile int64_t *m = &pp[0];

  bind_to_core(core);
  first_touch(tid);
  c0 = read_cycles();
  t0 = now_ns();

  barrier(2);
  /* a loop with a spin barrier */
  while(iters) {
#pragma unroll
    for(int i = 0; i < 64; i++) {
      /* spin attempting to CAS my tid -> otid; succeeds once the other
         core has handed the line to me */
      int64_t expected = tid;
      while(!__atomic_compare_exchange_n(m, &expected, (int64_t)otid, 1, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) { expected = tid; }
    }
    iters-=64;
  }
  c1 = read_cycles();
  t1 = now_ns();
  args->avg = ((double)(c1-c0))/ITERS/2.0;
  args->avg_ns = ((double)(t1-t0))/ITERS/2.0;
  close_cycles();
  return NULL;
}


static void* (*workers[]) (void *) = {one_cl_worker, two_cl_worker, cas_cl_worker};

static void run_pair(int i, int j) {
  phase = 0;
  waiters = 2;
  argarray[0].core = i; argarray[0].tid = 0; argarray[0].nthr = 2;
  argarray[1].core = j; argarray[1].tid = 1; argarray[1].nthr = 2;
  argarray[0].avg = argarray[1].avg = 0.0;
  argarray[0].avg_ns = argarray[1].avg_ns = 0.0;

  if(numa_local) {
    /* fresh page per pair; do NOT touch here -- the tid-0 worker (pinned to
       core i) first-touches it so it homes on core i's NUMA node */
    pp = (volatile int64_t*)mmap(NULL, 4096, PROT_READ|PROT_WRITE,
                                 MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    assert(pp != MAP_FAILED);
  } else {
    pp = pingpong;
    pp[0] = 1;
    pp[8] = 0;
  }

  for(int k = 0; k < 2; k++)
    pthread_create(threads+k, NULL, workers[w], argarray+k);
  for(int k = 0; k < 2; k++)
    pthread_join(threads[k], NULL);

  if(numa_local)
    munmap((void*)pp, 4096);
}

static void usage(const char *prog) {
  fprintf(stderr,
    "usage: %s [-p A,B] [-w N] [-n ITERS] [-L] [-o FILE]\n"
    "  -p A,B   measure only core pair (A,B); may be repeated\n"
    "  -w N     worker (default 0):\n"
    "             0 = one cache line, load/store handoff\n"
    "             1 = two cache lines, single-writer/single-reader\n"
    "             2 = one cache line, compare-and-swap handoff\n"
    "  -n ITERS iterations per pair (default 1048576, rounded down to x64)\n"
    "  -L       NUMA-local line: first-touch the shared line on core A's node\n"
    "           (removes the home-node placement artifact)\n"
    "  -o FILE  cycles-matrix CSV for full sweep (default data.csv)\n"
    "  -h       show this help\n"
    "Results are ONE-WAY latency (half a ping-pong round-trip).\n"
    "With no -p, sweeps all core pairs, writing the cycles matrix to FILE\n"
    "and the nanosecond matrix to the same name with an _ns suffix.\n", prog);
}

typedef struct { int a, b; } pair_t;

int main(int argc, char *argv[]) {
  int n_cores = sysconf(_SC_NPROCESSORS_ONLN);
  const char *outfile = "data.csv";
  pair_t *pairs = NULL;
  int npairs = 0, cap = 0;
  int opt;

  while((opt = getopt(argc, argv, "p:w:n:Lo:h")) != -1) {
    switch(opt) {
    case 'p': {
      int a, b;
      if(sscanf(optarg, "%d,%d", &a, &b) != 2) {
        fprintf(stderr, "bad pair '%s' (expected A,B)\n", optarg);
        return 1;
      }
      if(npairs == cap) {
        cap = cap ? cap*2 : 8;
        pairs = (pair_t*)realloc(pairs, cap*sizeof(pair_t));
      }
      pairs[npairs].a = a;
      pairs[npairs].b = b;
      npairs++;
      break;
    }
    case 'w': w = atoi(optarg); break;
    case 'n': {
      long v = atol(optarg);
      if(v < 64) v = 64;
      ITERS = (v/64)*64;
      break;
    }
    case 'L': numa_local = 1; break;
    case 'o': outfile = optarg; break;
    case 'h': usage(argv[0]); return 0;
    default:  usage(argv[0]); return 1;
    }
  }

  w = w % (int)(sizeof(workers)/sizeof(workers[0]));
  printf("max_cores = %d, w = %d, iters = %ld, numa_local = %d\n",
         n_cores, w, ITERS, numa_local);

  /* explicit pair mode: measure only the requested pairs */
  if(npairs > 0) {
    printf("# core_a,core_b,cyc_a,cyc_b,ns_a,ns_b  (one-way)\n");
    for(int p = 0; p < npairs; p++) {
      int i = pairs[p].a, j = pairs[p].b;
      if(i < 0 || j < 0 || i >= n_cores || j >= n_cores) {
        fprintf(stderr, "skipping out-of-range pair %d,%d (have %d cores)\n",
                i, j, n_cores);
        continue;
      }
      run_pair(i, j);
      printf("%d,%d,%g,%g,%g,%g\n", i, j,
             argarray[0].avg, argarray[1].avg,
             argarray[0].avg_ns, argarray[1].avg_ns);
      fflush(stdout);
    }
    free(pairs);
    return 0;
  }

  /* full sweep: cycles and ns matrices */
  double *cyc = (double*)malloc(sizeof(double)*n_cores*n_cores);
  double *ns  = (double*)malloc(sizeof(double)*n_cores*n_cores);
  memset(cyc, 0, sizeof(double)*n_cores*n_cores);
  memset(ns,  0, sizeof(double)*n_cores*n_cores);

  for(int i = 0; i < n_cores; i++) {
    for(int j = i+1; j < n_cores; j++) {
      run_pair(i, j);
      cyc[i*n_cores + j] = argarray[0].avg;
      cyc[j*n_cores + i] = argarray[1].avg;
      ns[i*n_cores + j]  = argarray[0].avg_ns;
      ns[j*n_cores + i]  = argarray[1].avg_ns;
    }
    fprintf(stderr, "\rrow %d/%d done", i+1, n_cores);
  }
  fprintf(stderr, "\n");

  FILE *fp = fopen(outfile, "w");
  for(int i = 0; i < n_cores; i++) {
    for(int j = 0; j < n_cores; j++) {
      fprintf(fp, "%g", cyc[i*n_cores+j]);
      if(j != n_cores-1) fprintf(fp, ",");
    }
    fprintf(fp, "\n");
  }
  fclose(fp);

  /* derive ns filename: insert _ns before the extension if present */
  char nsname[1024];
  const char *dot = strrchr(outfile, '.');
  if(dot) {
    int base = (int)(dot - outfile);
    snprintf(nsname, sizeof(nsname), "%.*s_ns%s", base, outfile, dot);
  } else {
    snprintf(nsname, sizeof(nsname), "%s_ns", outfile);
  }
  fp = fopen(nsname, "w");
  for(int i = 0; i < n_cores; i++) {
    for(int j = 0; j < n_cores; j++) {
      fprintf(fp, "%g", ns[i*n_cores+j]);
      if(j != n_cores-1) fprintf(fp, ",");
    }
    fprintf(fp, "\n");
  }
  fclose(fp);
  printf("wrote %s (cycles) and %s (ns)\n", outfile, nsname);

  free(cyc);
  free(ns);
  return 0;
}
