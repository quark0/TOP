CC = g++
CFLAGS += -I./tools/eigen/ -I./tools
CFLAGS += -std=c++11 
CFLAGS += -Wall
CFLAGS += -O3 

all: train

Top.o: Top.cc
	$(CC) $(CFLAGS) -c Top.cc
main.o: main.cc
	$(CC) $(CFLAGS) -c main.cc
problem.o: problem.cc
	$(CC) $(CFLAGS) -c problem.cc
train: main.o problem.o Top.o
	$(CC) main.o problem.o Top.o -o train
clean:
	rm -f *.o
