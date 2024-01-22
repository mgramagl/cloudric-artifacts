# SPDX-License-Identifier: BSD-3-Clause
# Copyright(c) 2017 Intel Corporation
#	This is an updated version starting from the service_cores example of DPDK
#		Author: Gines Garcia-Aviles
#				@GinesGarcia - Github
#		Institution: I2Cat Foundation

APP = lpu_models_app

CC = g++

# Flags
CFLAGS += -DALLOW_EXPERIMENTAL_API
CFLAGS += -O3
CFLAGS += $(WERROR_FLAGS)

# Source files
SRC_DIR := src
SRCS-y := $(wildcard $(SRC_DIR)/*.cpp)

CPPFLAGS := -Iinclude -I/usr/local/lib -L/opt/onnxruntime-linux-x64-1.16.3/lib -I/opt/onnxruntime-linux-x64-1.16.3/include -MMD -MP -std=c++17 -lm -lstdc++

all: shared
.PHONY: shared static
shared: build/$(APP)-shared
	ln -sf $(APP)-shared build/$(APP)
static: build/$(APP)-static
	ln -sf $(APP)-static build/$(APP)

build/$(APP)-shared: $(SRCS-y) Makefile $(PC_FILE) | build
	$(CC) $(CPPFLAGS) $(CFLAGS) $(SRCS-y) -o $@ $(LDFLAGS) $(LDFLAGS_SHARED) -lonnxruntime

build/$(APP)-static: $(SRCS-y) Makefile $(PC_FILE) | build
	$(CC) $(CPPFLAGS) $(CFLAGS) $(SRCS-y) -o $@ $(LDFLAGS) $(LDFLAGS_STATIC)

build:
	@mkdir -p $@

.PHONY: clean
clean:
	rm -f build/$(APP) build/$(APP)-static build/$(APP)-shared
	test -d build && rmdir -p build || true