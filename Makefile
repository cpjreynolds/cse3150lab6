PROJECT=lab6

# macos symlinks gcc/g++ to apple clang so this is here to ensure we get an
# actual GNU compiler.
#
# Apple Clang is a dumpster fire for C++ support.
uname_S := $(shell sh -c 'uname -s')
ifeq ($(uname_S),Darwin)
	# homebrew will install the latest version which is currently 13
	CXX=g++-13
else
	CXX=g++
endif

GCC := $(findstring GCC,$(shell sh -c '$(CXX) --version | head -1'))

# cant indent commands without an associated rule so just leave the whole thing
# unindented.
ifeq ($(uname_S),Darwin)
ifneq ($(GCC),GCC)
$(error macos compilation requires GCC)
endif
else
# this shouldn't happen outside of macos but just in case throw out a warning
ifneq ($(GCC),GCC)
$(warning project is tested with GCC. your mileage may vary.)
endif
endif

OPTOPTS=-ffast-math -Ofast -funroll-loops

SIMDOPTS=-mavx

ifdef NOAVX2
	NOAVX512=1
endif

ifndef NOAVX2
	SIMDOPTS+=-mavx2
endif

ifndef NOAVX512
	SIMDOPTS+=-mavx512f
endif

CXXFLAGS=-Wall --std=gnu++23 ${OPTOPTS} ${SIMDOPTS}

# to make the assembly more concise
ASMEXTRA=-fverbose-asm

# fix for libunwind issue on macos
ifeq ($(uname_S),Darwin)
CXXFLAGS += -Wl,-ld_classic
endif

# testing target
TESTTARGET=$(PROJECT)test.out
# runnable target
RUNTARGET=$(PROJECT).out

# all source files including test
SOURCES:=$(wildcard *.cpp)
OBJECTS:=$(SOURCES:.cpp=.o)
ASMFILES:=$(SOURCES:.cpp=.s)

# only the regular main file
#RSOURCES:=$(filter-out %.test.cpp,$(SOURCES))
# only the testing main file
#TSOURCES:=$(filter-out lab2.cpp,$(SOURCES))

.PHONY: all clean check run leaks asm

all: $(RUNTARGET)

check: $(TESTTARGET)
	./$(TESTTARGET)

run: $(RUNTARGET)
	./$(RUNTARGET)

asm: clean $(ASMFILES)

%.s: %.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(ASMEXTRA) -S $< -o $@

$(TESTTARGET): $(SOURCES)
	$(CXX) $(CPPFLAGS) -DTESTING $(CXXFLAGS) $^ -o $@

$(RUNTARGET): $(SOURCES)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ -o $@

leaks: $(RUNTARGET) $(TESTTARGET)
	leaks -atExit -quiet -- ./$(RUNTARGET)
	leaks -atExit -quiet -- ./$(TESTTARGET)

clean:
	rm -rf \
		$(OBJECTS)					\
		$(RUNTARGET)				\
		$(RUNTARGET:.out=.out.dSYM)	\
		$(TESTTARGET)				\
		$(TESTTARGET:.out=.out.dSYM)\
		$(ASMFILES)
