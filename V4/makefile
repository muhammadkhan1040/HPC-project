NVCC = nvcc
CFLAGS = -Wall -O3 -march=native -ffast-math -funroll-loops -fopenmp
NVCCFLAGS = -O3 --use_fast_math -lineinfo --default-stream per-thread

# Add architecture-specific optimization flags specifically for Tensor Cores
# Ensure Tensor Core support (compute capability 7.0 or higher)
ARCH = -gencode arch=compute_70,code=sm_70 \
       -gencode arch=compute_75,code=sm_75 \
       -gencode arch=compute_80,code=sm_80 \
       -gencode arch=compute_86,code=sm_86

# Add special flags for Tensor Core operations
TC_FLAGS = --ptxas-options=-v -D_TENSORCORE -std=c++14

# Add flags for dynamic parallelism
DP_FLAGS = -rdc=true -lcudadevrt

EXE = nn.exe
SRC = v4.cu

all: $(EXE) run

$(EXE): $(SRC)
	$(NVCC) $(NVCCFLAGS) $(ARCH) $(TC_FLAGS) $(DP_FLAGS) -Xptxas=-v,-O3 -Xcompiler -O3,-march=native,-ffast-math,-funroll-loops,-fopenmp -o $(EXE) $(SRC) -lm

run: $(EXE)
	./$(EXE)

clean:
	rm -f $(EXE)

profile: $(EXE)
	nvprof --metrics achieved_occupancy,sm_efficiency,tensor_precision_fu_utilization ./$(EXE)

tensor_profile: $(EXE)
	nvprof --metrics tensor_precision_fu_utilization,sm_efficiency,dram_read_throughput,dram_write_throughput,flop_count_hp ./$(EXE)