TEST      := mpiGraph

OMPI_PATH ?= /opt/hpc/software/mpi/hpcx/v2.4.0/ompi
ROCM_PATH ?= /opt/rocm

CXX       := g++
CXXFLAGS  := -O3 -march=znver1 -mtune=znver1
INCLUDES  := -I. -I$(OMPI_PATH)/include
LINK      := g++
LIBRARIES := -L. -L$(OMPI_PATH)/lib -lmpi
LDFLAGS   := -fPIC -Wl,-rpath=\$$ORIGIN

ifeq ($(rocm),1)
 CXXFLAGS += -D_USE_ROCM_
 INCLUDES += -I$(ROCM_PATH)/include
LIBRARIES += -L$(ROCM_PATH)/lib -L$(ROCM_PATH)/lib64 -lhip_hcc
endif

.PHONY: all
all: build

.PHONY: build
build: $(TEST)

%.o: %.c
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ -c $<

$(TEST): %: %.o
	$(LINK) $(LDFLAGS) -o $@ $+ $(LIBRARIES)

.PHONY: clean
clean:
	rm -f *.o $(TEST)
