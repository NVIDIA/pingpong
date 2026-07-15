/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <sched.h>

#include "perf.h"

int init_perf_counter(int config) {
  int fd;
  struct perf_event_attr pe;
  memset(&pe, 0, sizeof(pe));
  pe.type = PERF_TYPE_HARDWARE;
  pe.size = sizeof(pe);
  pe.config = config;
  pe.disabled = 1;
  pe.exclude_guest = 1;
  pe.inherit = 1;
  int cpu = sched_getcpu();
  fd = syscall(__NR_perf_event_open, &pe, 0, cpu, -1, 0);
  if(fd < 0) {
    perror("perf_event_open");
    fprintf(stderr, "perf counter initialization failed for counter %d\n", config);
  }
  assert(fd > -1);
  ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);
  return fd;
}
