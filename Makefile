CXX = g++
CPPFLAGS = -I./include
CXXFLAGS = -std=c++17 -Wall -Wextra -O3 -march=native -pthread
LDLIBS = -pthread

CUDA ?= 0
CUDA_HOME ?= /opt/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
CUDA_ARCH ?= 89
CUDA_PRECISION ?= 32
CUDA_BATCH ?= 256
CUDA_FUSED ?= 1
CUDA_PROFILE ?= 0
CUDA_PINNED ?= 0
CUDA_OVERLAP_INIT ?= 0

SRCS = $(wildcard src/*.cpp)
INCS = $(wildcard include/*.h)
BUILD = tmp/build/cpu
OBJS = $(patsubst src/%.cpp,$(BUILD)/%.o,$(SRCS))
EXEC = melspec

ifeq ($(CUDA),1)
BUILD = tmp/build/cuda-$(CUDA_ARCH)-fp$(CUDA_PRECISION)-b$(CUDA_BATCH)-f$(CUDA_FUSED)-p$(CUDA_PROFILE)-h$(CUDA_PINNED)-o$(CUDA_OVERLAP_INIT)
CPPFLAGS += -DMELSPEC_CUDA=1 -DMELSPEC_CUDA_PRECISION=$(CUDA_PRECISION)
CPPFLAGS += -DMELSPEC_CUDA_BATCH=$(CUDA_BATCH) -DMELSPEC_CUDA_FUSED=$(CUDA_FUSED)
CPPFLAGS += -DMELSPEC_CUDA_PINNED=$(CUDA_PINNED) -DMELSPEC_CUDA_OVERLAP_INIT=$(CUDA_OVERLAP_INIT)
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
ifneq ($(filter 32 64,$(CUDA_PRECISION)),)
else
$(error CUDA_PRECISION must be 32 or 64)
endif

.PHONY: all clean test format

all: $(EXEC)

$(EXEC): $(OBJS)
	@mkdir -p $(dir $@)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(BUILD)/%.o: src/%.cpp $(INCS) Makefile
	@mkdir -p $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -o $@ -c $<

$(BUILD)/melspec_cuda.o: src/melspec_cuda.cu $(INCS) Makefile
	@mkdir -p $(BUILD)
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -MMD -MP -o $@ -c $<

-include $(OBJS:.o=.d)

clean:
	rm -f $(OBJS) $(OBJS:.o=.d) $(EXEC)

test: $(EXEC)
	python3 ./validate.py ./$(EXEC)

format: $(SRCS) $(INCS)
	clang-format -i $^
