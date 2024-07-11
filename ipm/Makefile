CC := gcc
MKL_LOAD_FLAGS := -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
MKL_CMP_FLAGS := -I"${MKLROOT}/include"
CFLAGS := $(MKL_CMP_FLAGS) $(MKL_LOAD_FLAGS) -o3 -lstdc++

LIBRARY := blas_actors.o
SRC := cpu_ipm.o blas_test.o
PRODUCT := cpu_ipm.out blas_test.out
OBJ = $(SRC) $(LIBRARY)

all: $(PRODUCT)

library: $(LIBRARY)

clean::
		rm -f $(OBJ) $(PRODUCT)

%.o : %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

%.out : %.cpp $(LIBRARY)
		$(CC) $(CFLAGS) $< $(LIBRARY) -o $@
