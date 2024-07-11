CC := nvcc
MKL_LOAD_FLAGS := -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl -lcublas
MKL_CMP_FLAGS := -I"${MKLROOT}/include" -I"/path/to/cublas/include"
CFLAGS := $(MKL_CMP_FLAGS) $(MKL_LOAD_FLAGS) -O3 -lstdc++ -pg

NVCCFLAGS=-O3 $(GENCODE_FLAGS) -Xcompiler -march=native
NVCCFLAGS+= -c --std=c++03
NVCCFLAGS+= -DWITH_CUBLAS -I $(CUDA_ROOT)/include

LDFLAGS	= -L$(CUDA_ROOT)/lib64 -lcudart
LDFLAGS+= -lcublas -L $(CUDA_ROOT)/lib64 -Xcompiler \"-Wl,-rpath,$(CUDA_ROOT)/lib64\"

LIBRARY := blas_actors.o support.o
SRC := cpu_ipm.o blas_test.o gpu_ipm.o
PRODUCT := cpu_ipm.out blas_test.out gpu_ipm.out cublas_test.out
OBJ = $(SRC) $(LIBRARY)

all: $(PRODUCT)

library: $(LIBRARY)

clean::
		rm -f $(OBJ) $(PRODUCT) *.txt *.out *.o

%.o : %.cu
	$(CC) $(CFLAGS) $(LDFLAGS) $(NVCCFLAGS) -c $< -o $@

%.out : %.cu $(LIBRARY)
		$(CC) $(CFLAGS) $(LDFLAGS)  $< $(LIBRARY) -o $@

profiling::
		./cpu_ipm.out
		gprof -a cpu_ipm.out gmon.out > profiling_result.txt

cpu_ipm::
		./cpu_ipm.out > cpu_imp.log

test_cublas::
		sbatch cuda.sbatch

gpu_ipm::
		sbatch ipm.sbatch
