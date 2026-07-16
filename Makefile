# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

OBJ = pingpong.o perf.o
EXE = pingpong
OPT = -O3 -MMD
CXXFLAGS = -g $(OPT)
DEP = $(OBJ:.o=.d)

.PHONY: all clean

all: $(EXE)

$(EXE) : $(OBJ)
	gcc $(CXXFLAGS) $(OBJ) $(LIBS) -o $(EXE) -lpthread

%.o: %.c
	gcc -MMD $(CXXFLAGS) -c $<


-include $(DEP)

clean:
	rm -rf $(EXE) $(OBJ) $(DEP)
