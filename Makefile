CXX = g++
CPPFLAGS = -I./include
CXXFLAGS = -std=c++17 -Wall -Wextra -O3 -march=native -pthread
LDLIBS = -pthread

CUDA ?= 0
HIP ?= 0
CUDA_HOME ?= /opt/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
CUDA_ARCH ?= 89
CUDA_PRECISION ?= 32
CUDA_BATCH ?= 512
CUDA_FUSED ?= 1
CUDA_PROFILE ?= 0
CUDA_PINNED ?= 0
CUDA_OVERLAP_INIT ?= 1
CUDA_DEFER_RESIZE ?= 0
CUDA_ASYNC_OUTPUT ?= 1
CUDA_RESIZE_THREAD ?= 1
CUDA_STAGING_ROWS ?= 16

HIP_ARCH ?= gfx1201
HIP_PRECISION ?= 32
HIP_BATCH ?= 512
HIP_BATCH_LARGE ?= 256
HIP_FUSED ?= 1
HIP_PINNED ?= 0
HIP_PINNED_INPUT ?= 0
HIP_ASYNC_H2D ?= 1
HIP_PREWARM ?= 1
HIP_DIRECT_OUTPUT ?= 1
HIP_HUGEPAGE ?= 1
HIP_OVERLAP_INIT ?= 1
HIP_DEFER_RESIZE ?= 0
HIP_ASYNC_OUTPUT ?= 1
HIP_RESIZE_THREAD ?= 1
HIP_STAGING_ROWS ?= 16
HIP_PROFILE ?= 0
HIP_PREFIX ?= $(if $(ROCM_PREFIX),$(ROCM_PREFIX)/usr,$(CURDIR)/tmp/amd-gpu/root/usr)
HIPCC ?= $(HIP_PREFIX)/bin/hipcc
HIP_DEVICE_LIB_PATH ?= $(HIP_PREFIX)/lib/llvm-21/lib/clang/21/amdgcn/bitcode
HIP_LIBDIR = $(HIP_PREFIX)/lib/x86_64-linux-gnu
HIP_BUILD = tmp/build/hip-$(HIP_ARCH)-fp$(HIP_PRECISION)-b$(HIP_BATCH)-l$(HIP_BATCH_LARGE)-f$(HIP_FUSED)-p$(HIP_PROFILE)-h$(HIP_PINNED)-i$(HIP_PINNED_INPUT)-y$(HIP_ASYNC_H2D)-w$(HIP_PREWARM)-g$(HIP_DIRECT_OUTPUT)-u$(HIP_HUGEPAGE)-o$(HIP_OVERLAP_INIT)-d$(HIP_DEFER_RESIZE)-a$(HIP_ASYNC_OUTPUT)-t$(HIP_RESIZE_THREAD)-r$(HIP_STAGING_ROWS)

SRCS = $(wildcard src/*.cpp)
INCS = $(wildcard include/*.h)
BUILD = tmp/build/cpu
OBJS = $(patsubst src/%.cpp,$(BUILD)/%.o,$(SRCS))
EXEC = melspec

ifeq ($(CUDA)$(HIP),11)
$(error CUDA=1 and HIP=1 are mutually exclusive)
endif

ifeq ($(CUDA),1)
BUILD = tmp/build/cuda-$(CUDA_ARCH)-fp$(CUDA_PRECISION)-b$(CUDA_BATCH)-f$(CUDA_FUSED)-p$(CUDA_PROFILE)-h$(CUDA_PINNED)-o$(CUDA_OVERLAP_INIT)-d$(CUDA_DEFER_RESIZE)-a$(CUDA_ASYNC_OUTPUT)-t$(CUDA_RESIZE_THREAD)-r$(CUDA_STAGING_ROWS)
CPPFLAGS += -DMELSPEC_CUDA=1 -DMELSPEC_CUDA_PRECISION=$(CUDA_PRECISION)
CPPFLAGS += -DMELSPEC_CUDA_BATCH=$(CUDA_BATCH) -DMELSPEC_CUDA_FUSED=$(CUDA_FUSED)
CPPFLAGS += -DMELSPEC_CUDA_PINNED=$(CUDA_PINNED) -DMELSPEC_CUDA_OVERLAP_INIT=$(CUDA_OVERLAP_INIT)
CPPFLAGS += -DMELSPEC_CUDA_DEFER_RESIZE=$(CUDA_DEFER_RESIZE)
CPPFLAGS += -DMELSPEC_CUDA_ASYNC_OUTPUT=$(CUDA_ASYNC_OUTPUT)
CPPFLAGS += -DMELSPEC_CUDA_RESIZE_THREAD=$(CUDA_RESIZE_THREAD)
CPPFLAGS += -DMELSPEC_CUDA_STAGING_ROWS=$(CUDA_STAGING_ROWS)
ifeq ($(CUDA_PROFILE),1)
CPPFLAGS += -DMELSPEC_CUDA_PROFILE=1
else ifneq ($(CUDA_PROFILE),0)
$(error CUDA_PROFILE must be 0 or 1)
endif
CPPFLAGS += -I$(CUDA_HOME)/targets/x86_64-linux/include
OBJS += $(BUILD)/melspec_cuda.o
NVCCFLAGS = -std=c++17 -O3 -arch=sm_$(CUDA_ARCH) -Xcompiler -O3,-march=native,-pthread
LDLIBS += -L$(CUDA_HOME)/lib64 -Wl,-rpath,$(CUDA_HOME)/lib64 -lcufft -lcudart
else ifneq ($(CUDA),0)
$(error CUDA must be 0 or 1)
endif

ifeq ($(HIP),1)
BUILD = $(HIP_BUILD)
CPPFLAGS += -DMELSPEC_HIP=1 -I$(HIP_PREFIX)/include -I$(HIP_PREFIX)/include/hipfft
CPPFLAGS += -DMELSPEC_HIP_PRECISION=$(HIP_PRECISION) -DMELSPEC_HIP_BATCH=$(HIP_BATCH)
CPPFLAGS += -DMELSPEC_HIP_BATCH_LARGE=$(HIP_BATCH_LARGE)
CPPFLAGS += -DMELSPEC_HIP_FUSED=$(HIP_FUSED) -DMELSPEC_HIP_PINNED=$(HIP_PINNED)
CPPFLAGS += -DMELSPEC_HIP_PINNED_INPUT=$(HIP_PINNED_INPUT)
CPPFLAGS += -DMELSPEC_HIP_ASYNC_H2D=$(HIP_ASYNC_H2D)
CPPFLAGS += -DMELSPEC_HIP_PREWARM=$(HIP_PREWARM)
CPPFLAGS += -DMELSPEC_HIP_DIRECT_OUTPUT=$(HIP_DIRECT_OUTPUT)
CPPFLAGS += -DMELSPEC_HIP_HUGEPAGE=$(HIP_HUGEPAGE)
CPPFLAGS += -DMELSPEC_HIP_OVERLAP_INIT=$(HIP_OVERLAP_INIT)
CPPFLAGS += -DMELSPEC_HIP_DEFER_RESIZE=$(HIP_DEFER_RESIZE)
CPPFLAGS += -DMELSPEC_HIP_ASYNC_OUTPUT=$(HIP_ASYNC_OUTPUT)
CPPFLAGS += -DMELSPEC_HIP_RESIZE_THREAD=$(HIP_RESIZE_THREAD)
CPPFLAGS += -DMELSPEC_HIP_STAGING_ROWS=$(HIP_STAGING_ROWS)
ifeq ($(HIP_PROFILE),1)
CPPFLAGS += -DMELSPEC_HIP_PROFILE=1
else ifneq ($(HIP_PROFILE),0)
$(error HIP_PROFILE must be 0 or 1)
endif
OBJS += $(BUILD)/melspec_hip.o
HIPFLAGS = -std=c++17 -O3 --offload-arch=$(HIP_ARCH) \
  --rocm-path=$(HIP_PREFIX) --rocm-device-lib-path=$(HIP_DEVICE_LIB_PATH) \
  -march=native -pthread
LDLIBS += -L$(HIP_LIBDIR) -Wl,-rpath,$(HIP_LIBDIR) -lamdhip64 -lhipfft -lrocfft
else ifneq ($(HIP),0)
$(error HIP must be 0 or 1)
endif

ifneq ($(filter 0 1,$(CUDA_PINNED)),)
else
$(error CUDA_PINNED must be 0 or 1)
endif
ifneq ($(filter 0 1,$(CUDA_OVERLAP_INIT)),)
else
$(error CUDA_OVERLAP_INIT must be 0 or 1)
endif
ifneq ($(filter 0 1,$(CUDA_DEFER_RESIZE)),)
else
$(error CUDA_DEFER_RESIZE must be 0 or 1)
endif
ifneq ($(filter 0 1,$(CUDA_ASYNC_OUTPUT)),)
else
$(error CUDA_ASYNC_OUTPUT must be 0 or 1)
endif
ifneq ($(filter 0 1,$(CUDA_RESIZE_THREAD)),)
else
$(error CUDA_RESIZE_THREAD must be 0 or 1)
endif
ifeq ($(shell test "$(CUDA_STAGING_ROWS)" -gt 0 && echo yes),yes)
else
$(error CUDA_STAGING_ROWS must be positive)
endif
ifneq ($(filter 32 64,$(CUDA_PRECISION)),)
else
$(error CUDA_PRECISION must be 32 or 64)
endif
ifneq ($(filter 32 64,$(HIP_PRECISION)),)
else
$(error HIP_PRECISION must be 32 or 64)
endif
ifneq ($(filter-out 0 1,$(HIP_FUSED) $(HIP_PINNED) $(HIP_PINNED_INPUT) $(HIP_ASYNC_H2D) $(HIP_PREWARM) $(HIP_DIRECT_OUTPUT) $(HIP_HUGEPAGE) $(HIP_OVERLAP_INIT) $(HIP_DEFER_RESIZE) $(HIP_ASYNC_OUTPUT) $(HIP_RESIZE_THREAD) $(HIP_PROFILE)),)
$(error HIP booleans must be 0 or 1)
endif
ifeq ($(shell test "$(HIP_BATCH)" -gt 0 && test "$(HIP_BATCH_LARGE)" -gt 0 && test "$(HIP_STAGING_ROWS)" -gt 0 && echo yes),yes)
else
$(error HIP_BATCH, HIP_BATCH_LARGE and HIP_STAGING_ROWS must be positive)
endif

.PHONY: all clean test format hip-probe FORCE

all: $(EXEC)

$(EXEC): $(OBJS) | FORCE
	@mkdir -p $(dir $@)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(BUILD)/%.o: src/%.cpp $(INCS) Makefile
	@mkdir -p $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -o $@ -c $<

$(BUILD)/melspec_cuda.o: src/melspec_cuda.cu $(INCS) Makefile
	@mkdir -p $(BUILD)
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -MMD -MP -o $@ -c $<

$(BUILD)/melspec_hip.o: src/melspec_hip.cu $(INCS) Makefile
	@mkdir -p $(BUILD)
	$(HIPCC) $(CPPFLAGS) $(HIPFLAGS) -MMD -MP -o $@ -c $<

hip-probe: scripts/hip_probe.cpp Makefile
	@mkdir -p tmp/bin
	$(HIPCC) -std=c++17 -O3 --offload-arch=$(HIP_ARCH) \
	  --rocm-path=$(HIP_PREFIX) --rocm-device-lib-path=$(HIP_DEVICE_LIB_PATH) \
	  -I$(HIP_PREFIX)/include -I$(HIP_PREFIX)/include/hipfft \
	  scripts/hip_probe.cpp -o tmp/bin/hip_probe \
	  -L$(HIP_LIBDIR) -Wl,-rpath,$(HIP_LIBDIR) -lamdhip64 -lhipfft

-include $(OBJS:.o=.d)

FORCE:

clean:
	rm -f $(OBJS) $(OBJS:.o=.d) $(EXEC)

test: $(EXEC)
	python3 ./validate.py ./$(EXEC)

format: $(SRCS) $(INCS)
	clang-format -i $^
