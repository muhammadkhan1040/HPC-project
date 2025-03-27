CC = gcc
CFLAGS = -Wall -O2 -pg
LDFLAGS = -lm -pg

EXE = nn.exe
SRC = nn.c

all: $(EXE) profile

$(EXE): $(SRC)
	$(CC) $(CFLAGS) -o $(EXE) $(SRC) $(LDFLAGS)

profile: $(EXE)
	./$(EXE)  # Run the executable to generate gmon.out
	gprof $(EXE) > gprof_output.txt
	cat gprof_output.txt

clean:
	rm -f $(EXE) gmon.out gprof_output.txt

