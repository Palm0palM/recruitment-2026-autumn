CXX = g++
CPPFLAGS = -I./include
CXXFLAGS = -std=c++17 -Wall -Wextra -O3 -march=native -pthread
LDLIBS = -pthread

# CPU backend selection follows the frozen SIMD rules.
TARGET_MACROS = $(shell $(CXX) $(CXXFLAGS) -dM -E -x c++ /dev/null 2>/dev/null)
MELSPEC_FP32_BATCH ?= $(if $(findstring __AVX512F__,$(TARGET_MACROS)),16,$(if $(and $(findstring __AVX2__,$(TARGET_MACROS)),$(findstring __FMA__,$(TARGET_MACROS))),8,0))
ifneq ($(filter 0 8 16,$(MELSPEC_FP32_BATCH)),$(MELSPEC_FP32_BATCH))
$(error MELSPEC_FP32_BATCH must be 0, 8, or 16)
endif
CPPFLAGS += -DMELSPEC_FP32_BATCH=$(MELSPEC_FP32_BATCH)
CPU_THREADS ?= 16
ifeq ($(shell test "$(CPU_THREADS)" -ge 1 2>/dev/null && test "$(CPU_THREADS)" -le 1024 2>/dev/null && echo yes),)
$(error CPU_THREADS must be an integer in [1,1024])
endif
CPPFLAGS += -DMELSPEC_CPU_THREADS=$(CPU_THREADS)
CPU_PACK_REUSE ?= 1
CPU_HUGEPAGE ?= 1
ifneq ($(filter-out 0 1,$(CPU_PACK_REUSE) $(CPU_HUGEPAGE)),)
$(error CPU_PACK_REUSE and CPU_HUGEPAGE must be 0 or 1)
endif
CPPFLAGS += -DMELSPEC_PACK_REUSE=$(CPU_PACK_REUSE) -DMELSPEC_OUTPUT_HUGEPAGE=$(CPU_HUGEPAGE)

# CUDA backend uses the frozen V3 default configuration.
HIP ?= 0
HIP_ONLY ?= 0
CUDA ?= $(if $(filter 1,$(HIP)),0,1)
ifeq ($(CUDA)$(HIP),11)
$(error CUDA and HIP are mutually exclusive; build CPU+CUDA or CPU+HIP)
endif
ifneq ($(filter 0 1,$(HIP)),$(HIP))
$(error HIP must be 0 or 1)
endif
ifneq ($(filter 0 1,$(HIP_ONLY)),$(HIP_ONLY))
$(error HIP_ONLY must be 0 or 1)
endif
ifeq ($(HIP_ONLY)$(HIP),10)
$(error HIP_ONLY requires HIP=1)
endif
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

SRCS = $(wildcard src/*.cpp)
INCS = $(wildcard include/*.h)
BUILD = tmp/build/hybrid-cpu/fp$(MELSPEC_FP32_BATCH)-reuse$(CPU_PACK_REUSE)-hp$(CPU_HUGEPAGE)-threads$(CPU_THREADS)
OBJS = $(patsubst src/%.cpp,$(BUILD)/%.o,$(SRCS))
EXEC = melspec

ifeq ($(CUDA),1)
BUILD = tmp/build/hybrid-cuda-fp$(MELSPEC_FP32_BATCH)-reuse$(CPU_PACK_REUSE)-hp$(CPU_HUGEPAGE)-threads$(CPU_THREADS)-arch$(CUDA_ARCH)-p$(CUDA_PRECISION)-b$(CUDA_BATCH)-f$(CUDA_FUSED)-pr$(CUDA_PROFILE)-pin$(CUDA_PINNED)-o$(CUDA_OVERLAP_INIT)-d$(CUDA_DEFER_RESIZE)-a$(CUDA_ASYNC_OUTPUT)-t$(CUDA_RESIZE_THREAD)-r$(CUDA_STAGING_ROWS)
OBJS = $(patsubst src/%.cpp,$(BUILD)/%.o,$(SRCS))
CPPFLAGS += -DMELSPEC_WITH_CUDA=1 -DMELSPEC_CUDA_PRECISION=$(CUDA_PRECISION)
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
NVCCFLAGS = -std=c++17 -O3 -arch=sm_$(CUDA_ARCH) -Xcompiler -O3,-march=native,-pthread
LDLIBS += -L$(CUDA_HOME)/lib64 -Wl,-rpath,$(CUDA_HOME)/lib64 -lcufft -lcudart
OBJS += $(BUILD)/melspec_cuda.o
else ifneq ($(CUDA),0)
$(error CUDA must be 0 or 1)
endif

ifeq ($(HIP),1)
HIP_ARCH ?= gfx1201
HIP_PRECISION ?= 32
HIP_BATCH ?= 512
HIP_BATCH_LARGE ?= 256
HIP_FUSED ?= 1
HIP_PINNED ?= 0
HIP_PINNED_INPUT ?= 0
HIP_ASYNC_H2D ?= 1
HIP_PREWARM ?= 1
HIP_EAGER_PREWARM ?= $(HIP_ONLY)
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
ifeq ($(HIP_EAGER_PREWARM)$(HIP_ONLY),10)
$(error Eager HIP prewarm is allowed only for HIP_ONLY=1; CPU+HIP must stay lazy)
endif
ifeq ($(HIP_DIRECT_OUTPUT)$(HIP_ASYNC_OUTPUT),10)
$(error HIP_DIRECT_OUTPUT=1 requires HIP_ASYNC_OUTPUT=1; the legacy sync path copies Real, not OutReal)
endif
ifneq ($(filter-out 0 1,$(HIP_FUSED) $(HIP_PINNED) $(HIP_PINNED_INPUT) $(HIP_ASYNC_H2D) $(HIP_PREWARM) $(HIP_EAGER_PREWARM) $(HIP_DIRECT_OUTPUT) $(HIP_HUGEPAGE) $(HIP_OVERLAP_INIT) $(HIP_DEFER_RESIZE) $(HIP_ASYNC_OUTPUT) $(HIP_RESIZE_THREAD) $(HIP_PROFILE)),)
$(error HIP boolean options must be 0 or 1)
endif
ifeq ($(filter 32 64,$(HIP_PRECISION)),)
$(error HIP_PRECISION must be 32 or 64)
endif
ifeq ($(shell test "$(HIP_BATCH)" -gt 0 && test "$(HIP_BATCH_LARGE)" -gt 0 && test "$(HIP_STAGING_ROWS)" -gt 0 && echo yes),)
$(error HIP batch sizes and staging rows must be positive)
endif
BUILD = tmp/build/final-hip-$(HIP_ARCH)-cpu$(MELSPEC_FP32_BATCH)-reuse$(CPU_PACK_REUSE)-hp$(CPU_HUGEPAGE)-threads$(CPU_THREADS)-p$(HIP_PRECISION)-b$(HIP_BATCH)-l$(HIP_BATCH_LARGE)-f$(HIP_FUSED)-pin$(HIP_PINNED)-in$(HIP_PINNED_INPUT)-y$(HIP_ASYNC_H2D)-w$(HIP_PREWARM)-e$(HIP_EAGER_PREWARM)-only$(HIP_ONLY)-g$(HIP_DIRECT_OUTPUT)-u$(HIP_HUGEPAGE)-o$(HIP_OVERLAP_INIT)-d$(HIP_DEFER_RESIZE)-a$(HIP_ASYNC_OUTPUT)-t$(HIP_RESIZE_THREAD)-r$(HIP_STAGING_ROWS)-pr$(HIP_PROFILE)
OBJS = $(patsubst src/%.cpp,$(BUILD)/%.o,$(SRCS)) $(BUILD)/melspec_hip.o
CPPFLAGS += -DMELSPEC_WITH_HIP=1 -DMELSPEC_HIP_ONLY=$(HIP_ONLY)
CPPFLAGS += -DMELSPEC_HIP_PRECISION=$(HIP_PRECISION) -DMELSPEC_HIP_BATCH=$(HIP_BATCH) -DMELSPEC_HIP_BATCH_LARGE=$(HIP_BATCH_LARGE)
CPPFLAGS += -DMELSPEC_HIP_FUSED=$(HIP_FUSED) -DMELSPEC_HIP_PINNED=$(HIP_PINNED) -DMELSPEC_HIP_PINNED_INPUT=$(HIP_PINNED_INPUT)
CPPFLAGS += -DMELSPEC_HIP_ASYNC_H2D=$(HIP_ASYNC_H2D) -DMELSPEC_HIP_PREWARM=$(HIP_PREWARM) -DMELSPEC_HIP_EAGER_PREWARM=$(HIP_EAGER_PREWARM)
CPPFLAGS += -DMELSPEC_HIP_DIRECT_OUTPUT=$(HIP_DIRECT_OUTPUT) -DMELSPEC_HIP_HUGEPAGE=$(HIP_HUGEPAGE)
CPPFLAGS += -DMELSPEC_HIP_OVERLAP_INIT=$(HIP_OVERLAP_INIT) -DMELSPEC_HIP_DEFER_RESIZE=$(HIP_DEFER_RESIZE)
CPPFLAGS += -DMELSPEC_HIP_ASYNC_OUTPUT=$(HIP_ASYNC_OUTPUT) -DMELSPEC_HIP_RESIZE_THREAD=$(HIP_RESIZE_THREAD) -DMELSPEC_HIP_STAGING_ROWS=$(HIP_STAGING_ROWS)
CPPFLAGS += -I$(HIP_PREFIX)/include -I$(HIP_PREFIX)/include/hipfft
ifeq ($(HIP_PROFILE),1)
CPPFLAGS += -DMELSPEC_HIP_PROFILE=1
endif
HIPFLAGS = -std=c++17 -O3 --offload-arch=$(HIP_ARCH) --rocm-path=$(HIP_PREFIX) --rocm-device-lib-path=$(HIP_DEVICE_LIB_PATH) -march=native -pthread
LDLIBS += -L$(HIP_LIBDIR) -Wl,-rpath,$(HIP_LIBDIR) -lamdhip64 -lhipfft -lrocfft
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

.PHONY: all clean test format FORCE

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

-include $(OBJS:.o=.d)

FORCE:

clean:
	rm -rf tmp/build $(EXEC)

test: $(EXEC)
	python3 ./validate.py ./$(EXEC)

format: $(SRCS) $(INCS)
	clang-format -i $^
