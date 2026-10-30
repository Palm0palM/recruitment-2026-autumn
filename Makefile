CXX = g++
CPPFLAGS = -I./include
CXXFLAGS = -std=c++17 -Wall -Wextra -O3 -march=native -pthread
LDLIBS = -pthread
FFTW_PREFIX ?=
FFTW_PLAN ?= estimate
FFT_MODE ?= r2c
FRAME_BLOCK ?= 1
MEL_PROJECTION ?= sparse
CPU_THREADS ?= 16
OUTPUT_OVERLAP ?= 0
FUSE_POWER_MEL ?= 0
CPPFLAGS += -DMELSPEC_FRAME_BLOCK=$(FRAME_BLOCK) -DMELSPEC_MAX_THREADS=$(CPU_THREADS)
CPPFLAGS += -DMELSPEC_OUTPUT_OVERLAP=$(OUTPUT_OVERLAP) -DMELSPEC_FUSE_POWER_MEL=$(FUSE_POWER_MEL)
ifeq ($(MEL_PROJECTION),moments)
CPPFLAGS += -DMELSPEC_MOMENTS=1
else ifneq ($(MEL_PROJECTION),sparse)
$(error MEL_PROJECTION must be sparse or moments)
endif
ifeq ($(FFT_MODE),r2c)
CPPFLAGS += -DMELSPEC_FFTW_R2C=1
else ifeq ($(FFT_MODE),c2c)
CPPFLAGS += -DMELSPEC_FFTW_R2C=0
else
$(error FFT_MODE must be c2c or r2c)
endif
ifeq ($(FFTW_PLAN),measure)
CPPFLAGS += -DMELSPEC_FFTW_MEASURE=1
else ifneq ($(FFTW_PLAN),estimate)
$(error FFTW_PLAN must be estimate or measure)
endif
ifneq ($(FFTW_PREFIX),)
CPPFLAGS += -I$(FFTW_PREFIX)/include
LDLIBS += $(FFTW_PREFIX)/lib/libfftw3.a -lm
else
CPPFLAGS += $(shell pkg-config --cflags fftw3)
LDLIBS += $(shell pkg-config --libs fftw3)
endif

SRCS = $(wildcard src/*.cpp)
INCS = $(wildcard include/*.h)
BUILD = tmp/build/$(FFT_MODE)-$(FFTW_PLAN)-b$(FRAME_BLOCK)-$(MEL_PROJECTION)-t$(CPU_THREADS)-ow$(OUTPUT_OVERLAP)-f$(FUSE_POWER_MEL)
OBJS = $(patsubst src/%.cpp,$(BUILD)/%.o,$(SRCS))
EXEC = melspec

.PHONY: all clean test format

all: $(EXEC)

$(EXEC): $(OBJS)
	@mkdir -p $(dir $@)
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
