CXX = g++
CPPFLAGS = -I./include
CXXFLAGS = -std=c++17 -Wall -Wextra -O3 -march=native -pthread
LDLIBS = -pthread
FFTW_PREFIX ?=
FFTW_PLAN ?= estimate
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
BUILD = tmp/build/c2c-$(FFTW_PLAN)
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
