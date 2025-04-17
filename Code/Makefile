CC			:= gcc
NVCC		:= nvcc

DEFINES		:=

#Any value other than "double" will result in floats being used.

ifeq ($(type),double)
# ifdef double
TYPE		:= -DUSE_DOUBLE
else
TYPE		:= 
endif


CSTD		:= -std=c11
# COPT		:= -g
COPT		:= -O3 -march=native -funroll-loops #-fopenmp
CWARN		:= -Wextra -Wall 
CFLAGS		:= $(CWARN) $(COPT) $(DEFINES) $(TYPE) 

NVSTD		:= -Wno-deprecated-gpu-targets
NVSTD		+= -gencode arch=compute_35,code=sm_35 
NVSTD		+= -gencode arch=compute_75,code=sm_75
# NVOPT		:= -g -G
NVOPT		:= --use_fast_math -O4
NVCCFLAGS	:= $(NVSTD) $(NVOPT) --compiler-options="$(CFLAGS)"

LDFLAGS		:= -lm #-lefence
INCLUDES	:= 

EXECS 		:= assignment2

COBJS		:= cl_arguments.o matrix_utils.o timing.o iteration.o average.o 
CUOBJS		:= device_props.o average_gpu.o iteration_gpu.o
OBJS		:= $(COBJS) $(CUOBJS) $(EXECS:=.o)

CDEPS		:= $(COBJS:.o=.d) $(EXECS:=.d)
CUDEPS		:= $(CUOBJS:.o=.d)
DEPS		:= $(CDEPS) $(CUDEPS)

.PHONY: clean all

all: $(EXECS)

$(EXECS): %: %.o $(COBJS) $(CUOBJS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

$(COBJS) $(EXECS:=.o): %.o: %.c %.d
	$(CC) $(CFLAGS) $(CSTD) $(INCLUDES) -c $< -o $@ $(LDFLAGS)

$(CUOBJS): %.o: %.cu %.d
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@ $(LDFLAGS)

$(CDEPS): %.d: %.c
	$(CC) $(CSTD) -MM $^ -MF $@ 

$(CUDEPS): %.d: %.cu
	$(NVCC) -MM $^ -MF $@ 	

clean:
	rm -f $(OBJS)
	rm -f $(DEPS)
	rm -f $(EXECS)

ifneq ($(MAKECMDGOALS),clean)
-include $(DEPS)
endif

