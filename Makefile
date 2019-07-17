TEST      := mpiGraph

OMPI_PATH ?=
ROCM_PATH ?=

CXX       := g++
CXXFLAGS  := -O2
INCLUDES  := -I. -I$(OMPI_PATH)/include
LINK      := g++
LIBRARIES := -L. -L$(OMPI_PATH)/lib -lmpi
LDFLAGS   := -fPIC -Wl,-rpath=\$$ORIGIN -Wl,-rpath=$(OMPI_PATH)/lib

ifeq ($(rocm),1)
 CXXFLAGS += -D_USE_ROCM_
 INCLUDES += -I$(ROCM_PATH)/include
LIBRARIES += -L$(ROCM_PATH)/lib -L$(ROCM_PATH)/lib64 -lhip_hcc
  LDFLAGS += -Wl,-rpath=$(ROCM_PATH)/lib -Wl,-rpath=$(ROCM_PATH)/lib64
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
