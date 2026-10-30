CXX = g++
CPPFLAGS = -I./include
CXXFLAGS = -std=c++17 -Wall -Wextra -O3 -march=native -pthread
LDLIBS = -pthread

# Select the fastest available CPU backend at build time. Explicit overrides:
#   make MELSPEC_FP32_BATCH=16   # AVX-512F FP32 Batch16
#   make MELSPEC_FP32_BATCH=8    # AVX2+FMA FP32 Batch8
#   make MELSPEC_FP32_BATCH=0    # strict FP64
TARGET_MACROS = $(shell $(CXX) $(CXXFLAGS) -dM -E -x c++ /dev/null 2>/dev/null)
MELSPEC_FP32_BATCH ?= $(if $(findstring __AVX512F__,$(TARGET_MACROS)),16,$(if $(and $(findstring __AVX2__,$(TARGET_MACROS)),$(findstring __FMA__,$(TARGET_MACROS))),8,0))
ifneq ($(filter 0 8 16,$(MELSPEC_FP32_BATCH)),$(MELSPEC_FP32_BATCH))
$(error MELSPEC_FP32_BATCH must be 0, 8, or 16)
endif
CPPFLAGS += -DMELSPEC_FP32_BATCH=$(MELSPEC_FP32_BATCH)

SRCS = $(wildcard src/*.cpp)
INCS = $(wildcard include/*.h)
BUILD = tmp/build/fp$(MELSPEC_FP32_BATCH)
OBJS = $(patsubst src/%.cpp,$(BUILD)/%.o,$(SRCS))
EXEC = melspec

.PHONY: all clean test format

all: $(EXEC)

$(EXEC): $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(BUILD)/%.o: src/%.cpp $(INCS) Makefile
	@mkdir -p $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -o $@ -c $<

-include $(OBJS:.o=.d)

clean:
	rm -f $(OBJS) $(OBJS:.o=.d) $(EXEC)

test: $(EXEC)
	python3 ./validate.py ./$(EXEC)

format: $(SRCS) $(INCS)
	clang-format -i $^
