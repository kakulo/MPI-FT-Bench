# Makefile for src-parallel

SHELL = /bin/bash

### CoMD can be built in either double or single precision and with or
### without MPI.  Select desired precision and MPI here.

# double precision (ON/OFF)
DOUBLE_PRECISION = ON
# MPI for parallel (ON/OFF)
DO_MPI = ON
# Write Checkpoint (ON/OFF). 
# Compile with DO_CP ON will checkpoint application state
# after 10 (default printRate value) steps.
DO_CP = OFF
# Read Checkpoint (ON/OFF).
# Compile with DO_RESTART ON will load the last checkpoint
DO_TIMER = ON

### Set your desired C compiler and any necessary flags.  Note that CoMD
### uses some c99 features.  You can also set flags for optimization and
### specify paths to include files that the compiler can't find on its
### own.  If you need any -L or -l switches to get C standard libraries
### (such as -lm for the math library) put them in C_LIB.
CC = mpicc
CFLAGS = -std=c11 -D_GNU_SOURCE
OPTFLAGS = -O3
INCLUDES =
C_LIB = -lm

FTI_LIB = -L/g/g90/guo7/mylibs/FTI/lib64/ -lfti
FTI_INC = -I/g/g90/guo7/mylibs/FTI/include/


### If you need to specify include paths, library paths, or link flags
### for MPI, put them here.  Put both -L and -l switches into MPI_LIB.
MPI_LIB =
MPI_INCLUDE =

### A place to specify any other include or library switches your
### platform requires.
OTHER_LIB =
OTHER_INCLUDE =




#########################################
### Below here, it is pitch black.  
### You are likely to be eaten by a grue.
##########################################

# clear all suffixes
.SUFFIXES:
# list only those that we use 
.SUFFIXES: .c .o

.PHONY: DEFAULT clean distclean depend

BIN_DIR=./

# Check for double precision
ifeq ($(DOUBLE_PRECISION), ON)
CFLAGS += -DDOUBLE
else
CFLAGS += -DSINGLE
endif
# Check for checkpoint write
ifeq ($(DO_CP), ON)
CFLAGS += -DDO_CP
endif
# Check for TIMER
ifeq ($(DO_TIMER), ON)
CFLAGS += -DTIMER
endif
# Check for checkpoint read
ifeq ($(DO_RESTART), ON)
OPTFLAGS += -DRESTART
endif
ifeq ($(PROCFI), 1)
OPTFLAGS += -DPROCFI=1
endif
ifeq ($(NODEFI), 1)
OPTFLAGS += -DNODEFI=1
endif
# Set executable name and add includes & libraries for MPI if needed.
CoMD_VARIANT = CoMD-serial
ifeq ($(DO_MPI), ON)
CoMD_VARIANT = CoMD-mpi
INCLUDES += ${MPI_INCLUDE}
CFLAGS += -DDO_MPI
LDFLAGS += ${MPI_LIB}
LDFLAGS += ${FTI_LIB}
endif
CoMD_EXE = ${BIN_DIR}/${CoMD_VARIANT}

LDFLAGS += ${C_LIB} ${OTHER_LIB}
CFLAGS  += ${OPTFLAGS} ${INCLUDES} ${OTHER_INCLUDE}
CFLAGS += ${FTI_INC}

SOURCES=$(wildcard *.c)
OBJECTS=$(SOURCES:.c=.o) ../libcheckpoint/checkpoint.o


DEFAULT: ${CoMD_EXE}

%.o: %.c
	${CC} ${CFLAGS} -c $< -o $@

${CoMD_EXE}: ${BIN_DIR} CoMD_info.h ${OBJECTS} 
	${CC} ${CFLAGS} -o ${CoMD_EXE} ${OBJECTS} ${LDFLAGS}

CoMD_info.h: Makefile
	./generate_info_header ${CoMD_VARIANT} "$(CC)" "$(CFLAGS)" "$(LDFLAGS)"

${BIN_DIR}:
	@if [ ! -d ${BIN_DIR} ]; then mkdir -p ${BIN_DIR} ; fi

clean:
	rm -f *.o CoMD_info.h *yaml .depend *core

cl:
	rm -f *yaml check*

distclean: clean
	rm -f ${CoMD_EXE} .depend.bak
	rm -rf html latex

.depend: $(SOURCES)
	@touch .depend
	@$(MAKE) --no-print-directory depend

depend:
	@echo "Rebuilding dependencies..."
	@$(MAKE) CoMD_info.h
	@makedepend -f .depend -Y. --$(CFLAGS)-- $(SOURCES) 2> /dev/null


-include .depend
