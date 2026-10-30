CXX = g++
CPPFLAGS = -I./include
CXXFLAGS = -std=c++17 -Wall -Wextra -O3 -march=native -pthread
LDLIBS = -pthread
FFTW_PREFIX ?=
FFTW_PLAN ?= estimate
FFT_MODE ?= r2c
FFT_PRECISION ?= double
FFT_POWER ?= float
ifeq ($(FFT_PRECISION),float)
ifneq ($(FFT_MODE),r2c)
$(error FFT_PRECISION=float requires FFT_MODE=r2c)
endif
CPPFLAGS += -DMELSPEC_FFTW_FLOAT=1
FFTW_LIB = fftw3f
ifeq ($(FFT_POWER),double)
CPPFLAGS += -DMELSPEC_FFTWF_DOUBLE_POWER=1
else ifneq ($(FFT_POWER),float)
$(error FFT_POWER must be double or float)
endif
else ifeq ($(FFT_PRECISION),double)
FFTW_LIB = fftw3
else
$(error FFT_PRECISION must be double or float)
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
LDLIBS += $(FFTW_PREFIX)/lib/lib$(FFTW_LIB).a -lm
else
CPPFLAGS += $(shell pkg-config --cflags $(FFTW_LIB))
LDLIBS += $(shell pkg-config --libs $(FFTW_LIB))
endif

SRCS = $(wildcard src/*.cpp)
INCS = $(wildcard include/*.h)
BUILD = tmp/build/$(FFT_MODE)-$(FFTW_PLAN)-$(FFT_PRECISION)-$(FFT_POWER)
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
