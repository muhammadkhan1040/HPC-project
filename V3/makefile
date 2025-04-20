NVCC = nvcc
CFLAGS = -Wall -O3 -march=native -ffast-math -funroll-loops -fopenmp
NVCCFLAGS = -O3 --use_fast_math -lineinfo --default-stream per-thread

# Add architecture-specific optimization flags
# For maximum performance, use the compute capability of your specific GPU
# Common values: 60 (Pascal), 70 (Volta), 75 (Turing), 80/86 (Ampere)
ARCH = -gencode arch=compute_60,code=sm_60 \
       -gencode arch=compute_70,code=sm_70 \
       -gencode arch=compute_75,code=sm_75 \
       -gencode arch=compute_80,code=sm_80

EXE = nn.exe
SRC = v3.cu

all: $(EXE) run

$(EXE): $(SRC)
	$(NVCC) $(NVCCFLAGS) $(ARCH) -Xptxas=-v,-O3 -Xcompiler -O3,-march=native,-ffast-math,-funroll-loops,-fopenmp -o $(EXE) $(SRC) -lm

run: $(EXE)
	./$(EXE)

clean:
	rm -f $(EXE)

profile: $(EXE)
	nvprof --metrics achieved_occupancy ./$(EXE)