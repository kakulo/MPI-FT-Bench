CXX = mpicxx
# use -xmic-avx512 instead of -xHost for Intel Xeon Phi platforms
#OPTFLAGS = -O0 -fopenmp -DPRINT_DIST_STATS -DUSE_MPI_RMA # -DDEBUG_PRINTF #-DPRINT_EXTRA_NEDGES #-DUSE_MPI_RMA -DUSE_MPI_ACCUMULATE #-DUSE_32_BIT_GRAPH #-DDEBUG_PRINTF #-DUSE_MPI_RMA #-DPRINT_LCG_DOUBLE_LOHI_RANDOM_NUMBERS#-DUSE_MPI_RMA #-DPRINT_LCG_DOUBLE_RANDOM_NUMBERS #-DPRINT_RANDOM_XY_COORD
OPTFLAGS = -O0 -g -L/g/g90/guo7/mylibs/FTI/lib64/ -lfti # -DDEBUG_PRINTF #-DPRINT_EXTRA_NEDGES #-DUSE_MPI_RMA -DUSE_MPI_ACCUMULATE #-DUSE_32_BIT_GRAPH #-DDEBUG_PRINTF #-DUSE_MPI_RMA #-DPRINT_LCG_DOUBLE_LOHI_RANDOM_NUMBERS#-DUSE_MPI_RMA #-DPRINT_LCG_DOUBLE_RANDOM_NUMBERS #-DPRINT_RANDOM_XY_COORD
#-DUSE_MPI_SENDRECV
#-DUSE_MPI_COLLECTIVES
# use export ASAN_OPTIONS=verbosity=1 to check ASAN output
#SNTFLAGS = -std=c++11 -fopenmp -fsanitize=address -O1 -fno-omit-frame-pointer
SNTFLAGS = -std=c++11 -O0
#CXXFLAGS = -std=c++11 -g $(OPTFLAGS)
CXXFLAGS = -std=c++11 -O0 -g  -I/g/g90/guo7/mylibs/FTI/include/  $(OPTFLAGS)

OBJ = main.o ulfm-util.o
CPK_OBJ = ../libcheckpoint/checkpoint.o
TARGET = miniVite

all: $(TARGET)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $^

$(TARGET):  $(OBJ) $(CPK_OBJ)
	$(CXX) $^ $(OPTFLAGS) -o $@

.PHONY: clean

clean:
	rm -rf *~ $(OBJ) $(TARGET) check_* tmp_* *core
