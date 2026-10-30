CXX = g++
CPPFLAGS = -I./include
CXXFLAGS = -std=c++17 -Wall -Wextra -O3 -march=native -pthread
LDLIBS = -pthread
FFTW_PREFIX ?=
FFTW_PLAN ?= estimate
FFT_MODE ?= r2c
FFT_PRECISION ?= double
FFT_POWER ?= float
FFTW_WISDOM ?= 0
ifeq ($(FFTW_WISDOM),1)
ifneq ($(FFT_PRECISION),float)
$(error FFTW_WISDOM=1 currently requires FFT_PRECISION=float)
endif
ifneq ($(FFTW_PLAN),estimate)
$(error FFTW_WISDOM=1 requires FFTW_PLAN=estimate: runtime measurement is forbidden)
endif
CPPFLAGS += -DMELSPEC_FFTWF_WISDOM=1
else ifneq ($(FFTW_WISDOM),0)
$(error FFTW_WISDOM must be 0 or 1)
endif
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

SRCS = $(filter-out src/fftw_float_wisdom.cpp,$(wildcard src/*.cpp))
ifeq ($(FFTW_WISDOM),1)
SRCS += src/fftw_float_wisdom.cpp
endif
INCS = $(wildcard include/*.h)
BUILD = tmp/build/$(FFT_MODE)-$(FFTW_PLAN)-$(FFT_PRECISION)-$(FFT_POWER)-wisdom$(FFTW_WISDOM)
OBJS = $(patsubst src/%.cpp,$(BUILD)/%.o,$(SRCS))
EXEC = melspec

.PHONY: all clean test format wisdom-generator

all: $(EXEC)

# Independent offline tool. Never linked into melspec and never reads PCM.
wisdom-generator:
ifeq ($(FFT_PRECISION),float)
	@mkdir -p tmp/bin
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -DMELSPEC_FFTWF_WISDOM_GENERATOR=1 \
		src/fftw_float_wisdom.cpp scripts/fftw_float_wisdom_generate.cpp \
		-o tmp/bin/fftwf-wisdom-generate $(LDLIBS)
else
	$(error wisdom-generator requires FFT_PRECISION=float and a matching FFTW_PREFIX)
endif

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
