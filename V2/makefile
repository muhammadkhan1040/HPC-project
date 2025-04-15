NVCC = nvcc
CFLAGS = -Wall -O2
NVCCFLAGS = -O2

EXE = nn.exe
SRC = v2.cu

all: $(EXE) run

$(EXE): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $(EXE) $(SRC) -lm

run: $(EXE)
	./$(EXE)

clean:
	rm -f $(EXE)