CXX = g++
CPPFLAGS = -I./include
CXXFLAGS = -std=c++17 -Wall -Wextra -O3 -march=native -pthread
LDLIBS = -pthread

CUDA ?= 1
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
CUDA_BACKGROUND_INIT ?= 0
CUDA_GPU_INPUT_CONVERT ?= 0
CUDA_STAGING_ROWS ?= 16

SRCS = $(wildcard src/*.cpp)
INCS = $(wildcard include/*.h)
BUILD = tmp/build/cpu
OBJS = $(patsubst src/%.cpp,$(BUILD)/%.o,$(SRCS))
EXEC = melspec

ifeq ($(CUDA),1)
BUILD = tmp/build/cuda-$(CUDA_ARCH)-fp$(CUDA_PRECISION)-b$(CUDA_BATCH)-f$(CUDA_FUSED)-p$(CUDA_PROFILE)-h$(CUDA_PINNED)-o$(CUDA_OVERLAP_INIT)-d$(CUDA_DEFER_RESIZE)-a$(CUDA_ASYNC_OUTPUT)-t$(CUDA_RESIZE_THREAD)-g$(CUDA_BACKGROUND_INIT)-i$(CUDA_GPU_INPUT_CONVERT)-r$(CUDA_STAGING_ROWS)
CPPFLAGS += -DMELSPEC_CUDA=1 -DMELSPEC_CUDA_PRECISION=$(CUDA_PRECISION)
CPPFLAGS += -DMELSPEC_CUDA_BATCH=$(CUDA_BATCH) -DMELSPEC_CUDA_FUSED=$(CUDA_FUSED)
CPPFLAGS += -DMELSPEC_CUDA_PINNED=$(CUDA_PINNED) -DMELSPEC_CUDA_OVERLAP_INIT=$(CUDA_OVERLAP_INIT)
CPPFLAGS += -DMELSPEC_CUDA_DEFER_RESIZE=$(CUDA_DEFER_RESIZE)
CPPFLAGS += -DMELSPEC_CUDA_ASYNC_OUTPUT=$(CUDA_ASYNC_OUTPUT)
CPPFLAGS += -DMELSPEC_CUDA_RESIZE_THREAD=$(CUDA_RESIZE_THREAD)
CPPFLAGS += -DMELSPEC_CUDA_BACKGROUND_INIT=$(CUDA_BACKGROUND_INIT)
CPPFLAGS += -DMELSPEC_CUDA_GPU_INPUT_CONVERT=$(CUDA_GPU_INPUT_CONVERT)
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
ifneq ($(filter 0 1,$(CUDA_BACKGROUND_INIT)),)
else
$(error CUDA_BACKGROUND_INIT must be 0 or 1)
endif
ifneq ($(filter 0 1,$(CUDA_GPU_INPUT_CONVERT)),)
else
$(error CUDA_GPU_INPUT_CONVERT must be 0 or 1)
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

-include $(OBJS:.o=.d)

FORCE:

clean:
	rm -f $(OBJS) $(OBJS:.o=.d) $(EXEC)

test: $(EXEC)
	python3 ./validate.py ./$(EXEC)

format: $(SRCS) $(INCS)
	clang-format -i $^
